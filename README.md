# Michigan Senate Democratic Primary — Live Model

County-level Bayesian live election-night model for the 2026 Michigan US Senate
Democratic Primary (El-Sayed vs. Stevens), fed by civicAPI race `84778`.

Results from [civicAPI](https://civicapi.org).

## How it fits together

```
civicAPI  ──►  Render web service  ──►  Cloudflare Pages
 (poll)         (model + JSON API)        (the site)
```

One backend service. It polls civicAPI on a background thread, runs the model, and
serves the result over HTTP. The site reads that URL directly.

**This is a web service, not a cron job.** A cron container is destroyed after every
run, which wipes the batch history `mode_calibration` needs to learn the real
early-vs-Election-Day gap, and it has no URL for a site to read. The web service
solves both by staying alive.

## Files

| File | Does |
|---|---|
| `server.py` | background poller + JSON API. The entrypoint |
| `civicapi_feed.py` | API client, payload parsing, county name matching |
| `michigan_primary_model.py` | 83-county baselines, turnout scaling |
| `vote_method_split.py` | splits each county into early and Election Day margins |
| `vote_mode_inference.py` | infers which mode a reported batch is |
| `mode_calibration.py` | learns the real mode gap from finished counties |
| `turnout_calibration.py` | rewrites turnout from live percent reporting |
| `hierarchical_model.py` | correlated shift covariance and Monte Carlo |
| `web/` | the static site |

## Endpoints

| Route | Returns |
|---|---|
| `/health` | uptime, cycle count, last error |
| `/api/projection` | the current projection, county table, diagnostics |
| `/api/history` | one compact record per cycle since start |

CORS is open, so the site can be hosted anywhere.

## One cycle

```
fetch civicAPI
  → fold into VoteFeed (cumulative snapshots differenced into batches)
  → recalibrate turnout from percent reporting
  → recalibrate the early/Election Day gap from finished counties
  → joint mode/shift fit
  → correlated Monte Carlo, 20,000 sims
  → publish
```

About 0.6 seconds of compute per cycle.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `RACE_ID` | civicAPI race | `84778` |
| `N_SIMS` | Monte Carlo draws | `20000` |
| `POLL_INTERVAL` | seconds between cycles | `60` |
| `STATE_DIR` | optional disk path so batch history survives a restart | unset |

## Known limitations

- **civicAPI carries no vote-mode field.** Absentee versus Election Day is inferred,
  never observed.
- **`percent_reporting` counts precincts, not votes.** In Michigan those diverge
  early, because AV counting boards are not precincts in every county. Turnout
  calibration clamps and ramps for exactly this reason.
- **Margins are two-candidate.** McMorrow is dropped from the denominator, matching
  how the baselines were built.
- **Margin alone does not identify vote mode.** Volume bounds and finished counties
  do the real work.
- **State is in memory.** A restart costs the gap calibration until counties finish
  counting again. Set `STATE_DIR` to a mounted disk to avoid that.
