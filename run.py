"""
Live runner entrypoint.

Two modes, both supported from the same file:

    python run.py --once      one cycle, exit. For a Render cron job.
    python run.py --loop      poll forever. For a Render background worker.

WHICH ONE TO USE

    --loop is the better option and it is not close. A cron job pays container cold
    start every single minute, which on Render runs longer than the model itself, and
    it forces a full state round trip to R2 on both ends of every cycle. The loop
    starts once, holds the feed in memory, and writes state only as a crash
    insurance policy. The model takes about 0.6 seconds at 20,000 sims, so the loop
    spends the rest of the minute idle.

    --once exists because cron is what you asked for and because it is genuinely the
    right shape for a backfill or a manual re-run.

ONE CYCLE

    load state from R2 -> fetch civicAPI -> fold into VoteFeed -> recalibrate
    turnout -> recalibrate the mode gap -> joint mode/shift fit -> correlated Monte
    Carlo -> write projection.json and an archived copy -> save state
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone

from michigan_primary_model import build_michigan_county_data
from vote_method_split import build_vote_method_table
from mode_calibration import VoteFeed
from civicapi_feed import fetch_race, parse_payload, MICHIGAN_SENATE_DEM_PRIMARY
import hierarchical_model as hm

from store import get_store, STATE_KEY, PROJECTION_KEY, history_key


RACE_ID = int(os.environ.get("RACE_ID", MICHIGAN_SENATE_DEM_PRIMARY))
N_SIMS = int(os.environ.get("N_SIMS", 20000))
INTERVAL = int(os.environ.get("POLL_INTERVAL", 60))
ARCHIVE = os.environ.get("ARCHIVE_HISTORY", "1") == "1"


def load_feed(store) -> VoteFeed:
    feed = VoteFeed()
    stored = store.get_json(STATE_KEY)
    if not stored:
        return feed
    feed.snapshots = {
        county: [tuple(pair) for pair in history]
        for county, history in stored.get("snapshots", {}).items()
    }
    feed.pct_reporting = dict(stored.get("pct_reporting", {}))
    return feed


def save_feed(store, feed: VoteFeed) -> None:
    store.put_json(STATE_KEY, {
        "snapshots": feed.snapshots,
        "pct_reporting": feed.pct_reporting,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })


def build_output(result: dict, parsed: dict, race_id: int) -> dict:
    calibration = result.get("calibration") or {}
    turnout_cal = result.get("turnout_calibration") or {}
    diagnostics = turnout_cal.get("diagnostics")

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "civicapi.org",
        "attribution": "Election results from civicAPI (civicapi.org)",
        "race_id": race_id,
        "election_name": parsed.get("election_name"),
        "feed_last_updated": parsed.get("last_updated"),
        "counted": {
            "el_sayed": parsed.get("state_el_sayed"),
            "stevens": parsed.get("state_stevens"),
            "other": parsed.get("state_other"),
            "pct_of_projected_turnout": round(result["pct_counted"] * 100, 2),
            "pct_precincts_reporting": parsed.get("percent_precincts_statewide"),
        },
        "turnout": {
            "projected": result.get("projected_turnout"),
            "prior": int(turnout_cal.get("prior_statewide_total", 0)) or None,
            "pooled_ratio": round(turnout_cal.get("pooled_ratio", 1.0), 3),
            "counties_recalibrated": 0 if diagnostics is None else int(len(diagnostics)),
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
            "candidate_names": parsed.get("candidate_names"),
        },
        "regional_shift": {k: round(v, 2) for k, v
                           in result["region_posterior_shift"].items()},
    }


def run_cycle(store, counties, base_table, feed: VoteFeed, race_id: int) -> dict:
    payload = fetch_race(race_id)
    parsed = parse_payload(payload, counties)

    for county, record in parsed["counties"].items():
        feed.update(county, record["el_sayed"], record["stevens"],
                    pct_reporting=record.get("percent_precincts"))

    result = hm.simulate(counties, base_table, reported=None, feed=feed,
                         n_sims=N_SIMS)
    output = build_output(result, parsed, race_id)

    store.put_json(PROJECTION_KEY, output, public=True)
    if ARCHIVE:
        store.put_json(history_key(), output, public=True)
    save_feed(store, feed)

    return output


def log(output: dict) -> None:
    proj = output["projection"]
    diag = output["diagnostics"]
    names = diag.get("candidate_names") or {}

    # These two checks are the whole pre-flight. If you are deploying straight to
    # Render without a local run, this log line is the only place you will ever see
    # them, so they print loudly and unconditionally rather than being buried in the
    # JSON.
    if not names.get("el_sayed") or not names.get("stevens"):
        print("!! CANDIDATE MATCH FAILED: el_sayed={!r} stevens={!r} -- fix "
              "EL_SAYED_KEYS / STEVENS_KEYS in civicapi_feed.py".format(
                  names.get("el_sayed"), names.get("stevens")), flush=True)
    else:
        print("   matched: {} vs {}".format(
            names["el_sayed"], names["stevens"]), flush=True)

    if diag.get("unmatched_counties"):
        print("!! UNMATCHED COUNTIES: {} -- fix normalize_county() in "
              "civicapi_feed.py".format(diag["unmatched_counties"]), flush=True)

    print("[{}] {:.1f}% counted | {} cty | El-Sayed {:+.1f} [{:+.1f}, {:+.1f}] "
          "| win {:.1%} | gap x{:.2f} | turnout {:,}".format(
              datetime.now().strftime("%H:%M:%S"),
              output["counted"]["pct_of_projected_turnout"],
              diag["counties_reporting"],
              proj["median_margin"],
              proj["interval_90"][0], proj["interval_90"][1],
              proj["el_sayed_win_probability"],
              diag["mode_gap_multiplier"],
              output["turnout"]["projected"] or 0),
          flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=INTERVAL)
    parser.add_argument("--race-id", type=int, default=RACE_ID)
    args = parser.parse_args()

    store = get_store()
    backend = store.describe()
    print("state backend: {}".format(backend), flush=True)
    if backend.startswith("local:"):
        print("!! STATE IS NOT PERSISTING. On a cron job every container is fresh, "
              "so batch history is lost every run and the mode gap will sit at 1.00 "
              "all night while the projection still renders. Fix the R2_* env vars.",
              flush=True)

    counties = build_michigan_county_data()
    base_table, _ = build_vote_method_table(counties)
    feed = load_feed(store)
    print("loaded {} counties with history".format(len(feed.snapshots)), flush=True)

    if args.once or not args.loop:
        try:
            log(run_cycle(store, counties, base_table, feed, args.race_id))
            return 0
        except Exception:
            traceback.print_exc()
            return 1

    while True:
        started = time.time()
        try:
            log(run_cycle(store, counties, base_table, feed, args.race_id))
        except Exception as exc:
            # Never exit the loop on a bad cycle. A civicAPI hiccup or a malformed
            # payload should cost one update, not the rest of the night.
            print("[{}] cycle failed, keeping last good state: {}".format(
                datetime.now().strftime("%H:%M:%S"), exc), flush=True)
            traceback.print_exc()
        time.sleep(max(1.0, args.interval - (time.time() - started)))


if __name__ == "__main__":
    sys.exit(main())
