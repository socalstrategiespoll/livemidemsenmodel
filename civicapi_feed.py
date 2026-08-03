"""
civicAPI live feed for the Michigan model.

Endpoint:  https://civicapi.org/api/v2/race/{race_id}
Race:      84778  (2026 Michigan US Senate Democratic Primary)
Auth:      none. Attribution required for non-personal use, so credit civicapi.org
           anywhere this output is published.

FOUR THINGS THE PAYLOAD FORCED INTO THIS DESIGN

1. THE FEED CARRIES NO VOTE-MODE INFORMATION.
   Each county returns name, type, percent_reporting, and a candidate array with
   raw votes. Nothing about absentee versus Election Day. There is no field to
   read and no field coming. Mode inference is not a convenience here, it is the
   only path, and mode_calibration is what keeps it honest.

2. percent_reporting IS PRECINCTS, NOT VOTES.
   Do not feed it to the model as completeness. In Michigan the gap between the two
   is large and systematic: absentee counting boards are not precincts, so a county
   can post 60% of its votes at 5% precincts reporting, or sit at 90% precincts with
   the entire absentee batch still outstanding. Completeness is derived from counted
   votes against the projected county total instead. percent_reporting is carried
   through as a diagnostic only.

3. THE RACE HAS MORE THAN TWO CANDIDATES.
   McMorrow is in the field. Every margin in this model is defined two-candidate,
   El-Sayed against Stevens with everyone else dropped, because that is how the
   baselines were built. The client enforces that explicitly rather than letting a
   third candidate silently contaminate the denominator.

4. CORS IS DISABLED ON THE API.
   This has to run server-side. A browser frontend cannot call civicAPI directly, so
   the loop below is the thing that runs, and the frontend reads its JSON output.
"""

import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

from michigan_primary_model import build_michigan_county_data
from vote_method_split import build_vote_method_table
from mode_calibration import VoteFeed
import hierarchical_model as hm


API_BASE = "https://civicapi.org/api/v2"
MICHIGAN_SENATE_DEM_PRIMARY = 84778

EL_SAYED_KEYS = ("el-sayed", "elsayed", "el sayed")
STEVENS_KEYS = ("stevens",)

POLL_INTERVAL_SECONDS = 60
REQUEST_TIMEOUT = 15
MAX_RETRIES = 4


# ---------------------------------------------------------------------------
# County name matching
# ---------------------------------------------------------------------------

def normalize_county(name: str) -> str:
    """
    Reduce a county name to a matching key.

    Handles the cases that actually bite in Michigan: 'St. Clair' against
    'st_clair' or 'Saint Clair', 'Grand Traverse' against 'grand_traverse', and a
    trailing 'County' if the feed ever adds one.
    """
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\bcounty\b", " ", text)
    text = re.sub(r"\bsaint\b", "st", text)
    text = text.replace(".", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def build_county_lookup(counties: pd.DataFrame) -> dict:
    return {normalize_county(c): c for c in counties["county"]}


# ---------------------------------------------------------------------------
# Fetching and parsing
# ---------------------------------------------------------------------------

def fetch_race(race_id: int = MICHIGAN_SENATE_DEM_PRIMARY,
               timeout: int = REQUEST_TIMEOUT,
               max_retries: int = MAX_RETRIES,
               session=None) -> dict:
    """
    GET a race payload, retrying on transient failure with backoff.

    Raises on exhaustion. Callers on election night should catch and keep the last
    good snapshot rather than crashing the loop.
    """
    if requests is None:
        raise RuntimeError("requests is not installed: pip install requests")

    url = "{}/race/{}".format(API_BASE, race_id)
    getter = session.get if session is not None else requests.get
    last_error = None

    for attempt in range(max_retries):
        try:
            response = getter(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network, HTTP, or JSON decode
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError("civicAPI fetch failed after {} attempts: {}".format(
        max_retries, last_error))


def _match_candidate(name: str, keys: tuple) -> bool:
    lowered = str(name).lower()
    return any(k in lowered for k in keys)


def extract_two_candidate(candidate_list: list) -> tuple:
    """
    Pull El-Sayed and Stevens votes out of a candidate array.

    Everyone else is dropped. Returns (el_sayed, stevens, other, matched_names).
    """
    el_sayed = stevens = other = 0
    matched = {"el_sayed": None, "stevens": None}

    for entry in candidate_list or []:
        name = entry.get("name", "")
        votes = int(entry.get("votes") or 0)
        if _match_candidate(name, EL_SAYED_KEYS):
            el_sayed += votes
            matched["el_sayed"] = name
        elif _match_candidate(name, STEVENS_KEYS):
            stevens += votes
            matched["stevens"] = name
        else:
            other += votes

    return el_sayed, stevens, other, matched


def parse_payload(payload: dict, counties: pd.DataFrame) -> dict:
    """
    Turn a civicAPI race payload into county-level two-candidate vote counts.

    Returns dict with statewide totals, per-county records, and any county names in
    the feed that failed to match the model's 83.
    """
    lookup = build_county_lookup(counties)

    state_es, state_st, state_other, matched_names = extract_two_candidate(
        payload.get("candidates"))

    records = {}
    unmatched = []

    for _slug, region in (payload.get("region_results") or {}).items():
        if str(region.get("type", "")).lower() not in ("county", ""):
            continue
        raw_name = region.get("name", _slug)
        key = normalize_county(raw_name)
        county = lookup.get(key)
        if county is None:
            unmatched.append(raw_name)
            continue

        es, st, other, _ = extract_two_candidate(region.get("candidates"))
        if es + st <= 0:
            continue

        records[county] = {
            "el_sayed": es,
            "stevens": st,
            "other": other,
            "percent_precincts": region.get("percent_reporting"),
        }

    return {
        "election_name": payload.get("election_name"),
        "last_updated": payload.get("last_updated"),
        "percent_precincts_statewide": payload.get("percent_reporting"),
        "state_el_sayed": state_es,
        "state_stevens": state_st,
        "state_other": state_other,
        "candidate_names": matched_names,
        "counties": records,
        "unmatched": unmatched,
    }


# ---------------------------------------------------------------------------
# Live loop
# ---------------------------------------------------------------------------

class LiveRunner:
    """
    Polls civicAPI, maintains the batch history, and reruns the projection.

    Snapshot history is persisted to disk after every poll so a crash or restart
    mid-evening does not throw away the batch structure that mode_calibration
    depends on. Losing that history costs you the gap estimate.
    """

    def __init__(self,
                 race_id: int = MICHIGAN_SENATE_DEM_PRIMARY,
                 counties: pd.DataFrame = None,
                 method_table: pd.DataFrame = None,
                 state_path: str = "feed_state.json",
                 output_path: str = "projection.json",
                 n_sims: int = 20000):
        self.race_id = race_id
        self.counties = counties if counties is not None else build_michigan_county_data()
        if method_table is None:
            method_table, _ = build_vote_method_table(self.counties)
        self.base_table = method_table
        self.state_path = state_path
        self.output_path = output_path
        self.n_sims = n_sims

        self.feed = VoteFeed()
        self.last_payload = None
        self._load_state()

    # -- persistence -------------------------------------------------------

    def _load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as handle:
                stored = json.load(handle)
            self.feed.snapshots = {
                county: [tuple(pair) for pair in history]
                for county, history in stored.get("snapshots", {}).items()
            }
            self.feed.pct_reporting = dict(stored.get("pct_reporting", {}))
        except Exception:
            pass  # corrupt state should never stop the night

    def _save_state(self) -> None:
        try:
            with open(self.state_path, "w") as handle:
                json.dump({"snapshots": self.feed.snapshots,
                           "pct_reporting": self.feed.pct_reporting,
                           "saved_at": datetime.now(timezone.utc).isoformat()},
                          handle)
        except Exception:
            pass

    # -- one cycle ---------------------------------------------------------

    def poll(self) -> dict:
        """Fetch once, fold into the feed, and return the parsed payload."""
        payload = fetch_race(self.race_id)
        parsed = parse_payload(payload, self.counties)

        for county, record in parsed["counties"].items():
            self.feed.update(county, record["el_sayed"], record["stevens"],
                             pct_reporting=record.get("percent_precincts"))

        self.last_payload = parsed
        self._save_state()
        return parsed

    def project(self, seed: int = None) -> dict:
        """Run the model against the current feed state."""
        return hm.simulate(self.counties, self.base_table,
                           reported=None, feed=self.feed,
                           n_sims=self.n_sims, seed=seed)

    def emit(self, result: dict, parsed: dict) -> dict:
        """Write a JSON snapshot for a frontend to read."""
        calibration = result.get("calibration") or {}
        turnout_cal = result.get("turnout_calibration") or {}
        output = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": "civicapi.org",
            "race_id": self.race_id,
            "election_name": parsed.get("election_name"),
            "feed_last_updated": parsed.get("last_updated"),
            "counted": {
                "el_sayed": parsed.get("state_el_sayed"),
                "stevens": parsed.get("state_stevens"),
                "other": parsed.get("state_other"),
                "pct_of_projected_turnout": round(result["pct_counted"] * 100, 2),
                "pct_precincts_reporting": parsed.get("percent_precincts_statewide"),
                "projected_turnout": result.get("projected_turnout"),
                "turnout_vs_prior": round(turnout_cal.get("pooled_ratio", 1.0), 3),
            },
            "projection": {
                "el_sayed_win_probability": round(result["el_sayed_win_probability"], 4),
                "median_margin": round(result["median_margin"], 2),
                "interval_50": [round(result["margin_ci_50_lower"], 2),
                                round(result["margin_ci_50_upper"], 2)],
                "interval_90": [round(result["margin_ci_lower"], 2),
                                round(result["margin_ci_upper"], 2)],
                "el_sayed_votes": int(result["el_sayed_median_votes"]),
                "stevens_votes": int(result["stevens_median_votes"]),
            },
            "diagnostics": {
                "counties_reporting": result["n_reported"],
                "implied_state_shift": round(result["implied_state_shift"], 2),
                "mode_gap_multiplier": round(calibration.get("kappa_mean", 1.0), 3),
                "mode_gap_multiplier_sd": round(calibration.get("kappa_sd", 0.0), 3),
                "counties_calibrating_gap": calibration.get("n_counties_used", 0),
                "joint_passes": result.get("joint_passes"),
                "unmatched_counties": parsed.get("unmatched", []),
            },
            "regional_shift": {k: round(v, 2) for k, v
                               in result["region_posterior_shift"].items()},
        }
        try:
            with open(self.output_path, "w") as handle:
                json.dump(output, handle, indent=2)
        except Exception:
            pass
        return output

    # -- loop --------------------------------------------------------------

    def run(self, interval: int = POLL_INTERVAL_SECONDS,
            max_cycles: int = None, verbose: bool = True) -> None:
        cycle = 0
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            try:
                parsed = self.poll()
                result = self.project()
                output = self.emit(result, parsed)

                if verbose:
                    proj = output["projection"]
                    diag = output["diagnostics"]
                    print("[{}] {:.1f}% counted | El-Sayed {:+.1f} "
                          "[{:+.1f}, {:+.1f}] | win {:.1%} | gap x{:.2f} ({} cty)".format(
                              datetime.now().strftime("%H:%M:%S"),
                              output["counted"]["pct_of_projected_turnout"],
                              proj["median_margin"],
                              proj["interval_90"][0], proj["interval_90"][1],
                              proj["el_sayed_win_probability"],
                              diag["mode_gap_multiplier"],
                              diag["counties_calibrating_gap"]))
                    if parsed["unmatched"]:
                        print("   unmatched counties: {}".format(parsed["unmatched"]))

            except Exception as exc:
                print("[{}] cycle failed, keeping last state: {}".format(
                    datetime.now().strftime("%H:%M:%S"), exc))

            if max_cycles is None or cycle < max_cycles:
                time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live civicAPI feed for the MI model")
    parser.add_argument("--race-id", type=int, default=MICHIGAN_SENATE_DEM_PRIMARY)
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL_SECONDS)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--sims", type=int, default=20000)
    parser.add_argument("--state", default="feed_state.json")
    parser.add_argument("--output", default="projection.json")
    parser.add_argument("--once", action="store_true", help="single poll then exit")
    args = parser.parse_args()

    runner = LiveRunner(race_id=args.race_id, state_path=args.state,
                        output_path=args.output, n_sims=args.sims)
    runner.run(interval=args.interval,
               max_cycles=1 if args.once else args.cycles)
