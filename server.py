# Render web service: polls civicAPI on a background thread and serves the projection.
#
# WHY THIS AND NOT A CRON JOB
#
#     A cron job cannot do this job. Two reasons, and the second is fatal:
#
#     1. Render destroys the container after every cron run. mode_calibration learns
#        the real early-vs-Election-Day gap from the SEQUENCE of batches each county
#        reports, so wiping state every minute means every county looks like it
#        reported once in a single dump. That is exactly the case where the gap is
#        unidentified. It silently reverts to the assumed 21.04 points and stays there
#        all night while the projection keeps rendering as if nothing is wrong.
#
#     2. A cron job has nowhere to publish. There is no URL, so there is nothing for a
#        website to connect to. You would need external storage purely to bridge the
#        gap between the job and the site.
#
#     A web service solves both by existing continuously. The poller thread holds the
#     VoteFeed in memory across cycles, and the HTTP thread serves whatever the last
#     completed cycle produced. No object storage, no Worker, no credentials.
#
# DESIGN NOTES
#
#     Stdlib only. No Flask, no gunicorn. That is deliberate: gunicorn with more than
#     one worker would spawn one poller per worker, and they would fight over the API
#     and produce inconsistent projections. A single-process threading server makes
#     that mistake impossible rather than merely documented.
#
#     The poller never lets an exception escape. A civicAPI hiccup, a malformed
#     payload, or a transient network failure costs one update and nothing more. The
#     previous projection stays served with its own timestamp so the site can show how
#     stale it is.
#
#     State is in memory. If Render restarts the service you lose batch history, which
#     costs you the gap calibration until counties finish counting again. Attach a
#     Render persistent disk mounted at /var/data and set STATE_DIR to it if you want
#     that covered.
import json
import os
import threading
import time
import traceback

import numpy as np

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from michigan_primary_model import build_michigan_county_data
from vote_method_split import build_vote_method_table
from mode_calibration import VoteFeed
from civicapi_feed import fetch_race, parse_payload, MICHIGAN_SENATE_DEM_PRIMARY
import hierarchical_model as hm


PORT = int(os.environ.get("PORT", 10000))
RACE_ID = int(os.environ.get("RACE_ID", MICHIGAN_SENATE_DEM_PRIMARY))
N_SIMS = int(os.environ.get("N_SIMS", 20000))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 60))
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", 2000))
STATE_DIR = os.environ.get("STATE_DIR", "")


class ModelState:
    """Everything the poller produces and the HTTP handler reads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.projection = None
        self.history = []          # trimmed list of past cycles
        self.error = None
        self.cycles = 0
        self.started_at = datetime.now(timezone.utc).isoformat()

    def publish(self, output: dict) -> None:
        with self.lock:
            self.projection = output
            self.history.append({
                "updated_at": output["updated_at"],
                "median_margin": output["projection"]["median_margin"],
                "win_probability": output["projection"]["el_sayed_win_probability"],
                "interval_90": output["projection"]["interval_90"],
                "pct_counted": output["counted"]["pct_of_projected_turnout"],
                "counties_reporting": output["diagnostics"]["counties_reporting"],
                "mode_gap_multiplier": output["diagnostics"]["mode_gap_multiplier"],
                "projected_turnout": output["turnout"]["projected"],
            })
            if len(self.history) > HISTORY_LIMIT:
                self.history = self.history[-HISTORY_LIMIT:]
            self.error = None
            self.cycles += 1

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message

    def snapshot(self) -> tuple:
        with self.lock:
            return self.projection, list(self.history), self.error, self.cycles


STATE = ModelState()


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
            # The actual simulated distribution, thinned to 61 percentiles. The site
            # draws its density curve from this rather than assuming a normal shape
            # around the median, which matters because the posterior is genuinely
            # skewed when a few large counties are partly counted.
            "margin_percentiles": [
                round(float(v), 2) for v in
                np.percentile(result["margins"], np.arange(1, 100, 1.65))
            ],
        },
        "counties": build_county_table(result, parsed),
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


def build_county_table(result: dict, parsed: dict) -> list:
    """
    Per-county rows covering ALL 83, not just the ones reporting.

    The maps need every county every cycle: the results map has to draw the ones
    with no results as no-results rather than omitting them, and the remainder map
    is mostly about counties that have not finished.
    """
    table = result["method_table"].set_index("county")
    shifts = result["county_posterior_mean_shift"]
    remainder = result["remainder_baseline"]
    remaining = result["remaining_votes"]
    rem_early = result["remaining_early"]
    rem_ed = result["remaining_ed"]
    theta = result["county_theta"]
    reported = parsed.get("counties", {})

    rows = []
    for county in table.index:
        row = table.loc[county]
        total = float(row["total_votes"])
        record = reported.get(county)
        counted = (record["el_sayed"] + record["stevens"]) if record else 0

        margin = None
        if counted > 0:
            margin = round((record["el_sayed"] - record["stevens"]) / counted * 100.0, 1)

        left = float(remaining.get(county, total))
        rem_margin = float(remainder.get(county, row["blended_margin"]))

        # Where the county lands once the remainder is added in at its projected
        # margin. This is the number the projection is actually built on.
        if counted > 0 and left >= 0:
            net = (record["el_sayed"] - record["stevens"]) + left * rem_margin / 100.0
            final = round(net / max(counted + left, 1.0) * 100.0, 1)
        else:
            final = round(rem_margin, 1)

        rows.append({
            "county": county,
            "reporting": counted > 0,
            "el_sayed": record["el_sayed"] if record else 0,
            "stevens": record["stevens"] if record else 0,
            "votes": counted,
            "margin": margin,
            "expected_blended": round(float(row["blended_margin"]), 1),
            "vs_expected": None if margin is None else round(margin - float(row["blended_margin"]), 1),
            "shift": round(float(shifts.get(county, 0.0)), 1),
            "pct_precincts": record.get("percent_precincts") if record else None,
            "pct_of_projected": round(counted / max(total, 1.0) * 100, 1),
            "projected_total": int(total),
            "remaining": int(round(left)),
            "remaining_early": int(round(float(rem_early.get(county, 0.0)))),
            "remaining_ed": int(round(float(rem_ed.get(county, 0.0)))),
            "remainder_margin": round(rem_margin, 1),
            "early_margin": round(float(row["early_margin"]), 1),
            "ed_margin": round(float(row["ed_margin"]), 1),
            "theta": round(float(theta.get(county, 1.0)), 2),
            "projected_final": final,
        })

    rows.sort(key=lambda r: (-r["votes"], -r["projected_total"]))
    return rows


def save_state(feed: VoteFeed) -> None:
    if not STATE_DIR:
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(os.path.join(STATE_DIR, "feed_state.json"), "w") as handle:
            json.dump({"snapshots": feed.snapshots,
                       "pct_reporting": feed.pct_reporting}, handle)
    except Exception:
        pass


def load_state(feed: VoteFeed) -> None:
    if not STATE_DIR:
        return
    path = os.path.join(STATE_DIR, "feed_state.json")
    try:
        with open(path) as handle:
            stored = json.load(handle)
        feed.snapshots = {c: [tuple(p) for p in h]
                          for c, h in stored.get("snapshots", {}).items()}
        feed.pct_reporting = dict(stored.get("pct_reporting", {}))
        print("restored {} counties from {}".format(len(feed.snapshots), path),
              flush=True)
    except Exception:
        pass


def poller() -> None:
    """Background loop. Never exits."""
    counties = build_michigan_county_data()
    base_table, _ = build_vote_method_table(counties)
    feed = VoteFeed()
    load_state(feed)

    print("poller started: race {} every {}s, {} sims".format(
        RACE_ID, POLL_INTERVAL, N_SIMS), flush=True)

    while True:
        started = time.time()
        try:
            payload = fetch_race(RACE_ID)
            parsed = parse_payload(payload, counties)

            for county, record in parsed["counties"].items():
                feed.update(county, record["el_sayed"], record["stevens"],
                            pct_reporting=record.get("percent_precincts"))

            result = hm.simulate(counties, base_table, reported=None, feed=feed,
                                 n_sims=N_SIMS)
            output = build_output(result, parsed, RACE_ID)
            STATE.publish(output)
            save_state(feed)

            names = output["diagnostics"].get("candidate_names") or {}
            if not names.get("el_sayed") or not names.get("stevens"):
                print("!! CANDIDATE MATCH FAILED: el_sayed={!r} stevens={!r} -- fix "
                      "EL_SAYED_KEYS / STEVENS_KEYS in civicapi_feed.py".format(
                          names.get("el_sayed"), names.get("stevens")), flush=True)
            else:
                print("   matched: {} vs {}".format(
                    names["el_sayed"], names["stevens"]), flush=True)
            if output["diagnostics"]["unmatched_counties"]:
                print("!! UNMATCHED COUNTIES: {} -- fix normalize_county() in "
                      "civicapi_feed.py".format(
                          output["diagnostics"]["unmatched_counties"]), flush=True)

            proj = output["projection"]
            print("[{}] {:.1f}% counted | {} cty | El-Sayed {:+.1f} "
                  "[{:+.1f}, {:+.1f}] | win {:.1%} | gap x{:.2f} | turnout {:,}".format(
                      datetime.now().strftime("%H:%M:%S"),
                      output["counted"]["pct_of_projected_turnout"],
                      output["diagnostics"]["counties_reporting"],
                      proj["median_margin"], proj["interval_90"][0],
                      proj["interval_90"][1], proj["el_sayed_win_probability"],
                      output["diagnostics"]["mode_gap_multiplier"],
                      output["turnout"]["projected"] or 0), flush=True)

        except Exception as exc:
            STATE.fail(str(exc))
            print("[{}] cycle failed, serving last good projection: {}".format(
                datetime.now().strftime("%H:%M:%S"), exc), flush=True)
            traceback.print_exc()

        time.sleep(max(1.0, POLL_INTERVAL - (time.time() - started)))


class Handler(BaseHTTPRequestHandler):

    def _send(self, body, status=200, content_type="application/json"):
        encoded = (body if isinstance(body, bytes)
                   else json.dumps(body).encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        # Permissive CORS so the site can live anywhere: Pages, a custom domain,
        # or a local file while you are iterating on it.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        projection, history, error, cycles = STATE.snapshot()

        if path in ("/", "/health"):
            return self._send({
                "ok": True,
                "cycles": cycles,
                "started_at": STATE.started_at,
                "last_error": error,
                "has_projection": projection is not None,
            })

        if path == "/api/projection":
            if projection is None:
                return self._send(
                    {"error": "no projection yet", "last_error": error}, status=503)
            return self._send(projection)

        if path == "/api/history":
            return self._send({"count": len(history), "cycles": history})

        return self._send({"error": "not found"}, status=404)

    def log_message(self, *args):
        return  # suppress per-request noise; the poller owns the log


def main():
    thread = threading.Thread(target=poller, daemon=True)
    thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("serving on :{}".format(PORT), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
