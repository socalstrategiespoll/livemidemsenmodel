# Michigan Senate Democratic Primary — Live Election-Night Model

County-level Bayesian live model for the 2026 Michigan US Senate Democratic Primary
(El-Sayed vs. Stevens), fed by civicAPI race `84778`.

Election results from [civicAPI](https://civicapi.org).

## Architecture

```
civicAPI  ──►  Render worker  ──►  Cloudflare R2  ──►  Cloudflare Worker  ──►  Pages site
  (poll)        (run model)         (private)           (public read)          (browser)
```

Three deliberate choices in that chain:

- **R2 sits in the middle, not the repo.** The runner does not commit results to git.
  Every-minute commits would bury the history and Pages would rebuild on each one.
- **The bucket is private.** Only the Worker reads it, so runner credentials never
  reach the browser and `feed_state.json` is not internet-reachable.
- **The Worker exists because civicAPI has CORS disabled.** The site cannot call the
  results API directly; something server-side has to.

## Repository layout

```
model/                   the model itself, importable as a package
  michigan_primary_model.py   83-county baselines, turnout scaling
  vote_method_split.py        early vs Election Day margin split
  vote_mode_inference.py      infers which mode a dump is
  mode_calibration.py         learns the realized mode gap from finished counties
  turnout_calibration.py      rewrites turnout from live percent reporting
  hierarchical_model.py       correlated shift covariance + Monte Carlo
  civicapi_feed.py            API client, parsing, county matching
runner/
  run.py                 entrypoint: --once for cron, --loop for a worker
  store.py               R2 backend with local fallback
worker/                  Cloudflare Worker read layer
web/                     static site for Cloudflare Pages
render.yaml              Render blueprint (worker and cron both defined)
```

## One cycle

```
load state from R2
  → fetch civicAPI
  → fold into VoteFeed (cumulative snapshots, differenced into batches)
  → recalibrate turnout from percent reporting
  → recalibrate the early/ED gap from finished counties
  → joint mode/shift fit
  → correlated Monte Carlo, 20,000 sims
  → write projection.json + archived copy
  → save state
```

Roughly 0.6 seconds of compute at 20,000 simulations, plus network.

## Run it

```bash
pip install -r requirements.txt

python run.py --once            # one cycle
python run.py --loop            # poll forever
```

With no R2 credentials set it falls back to `./state` on local disk and says so.

### Environment

| Variable | Purpose | Default |
|---|---|---|
| `RACE_ID` | civicAPI race | `84778` |
| `N_SIMS` | Monte Carlo draws | `20000` |
| `POLL_INTERVAL` | seconds between cycles in loop mode | `60` |
| `ARCHIVE_HISTORY` | keep every cycle for the post-mortem | `1` |
| `R2_ACCOUNT_ID` | Cloudflare account | — |
| `R2_ACCESS_KEY_ID` | R2 token | — |
| `R2_SECRET_ACCESS_KEY` | R2 token | — |
| `R2_BUCKET` | bucket name | — |

## Deploy

**1. R2 bucket.** Create one in the Cloudflare dashboard. Keep it private. Generate an
R2 API token with object read and write.

**2. Render.** Point a new Blueprint at this repo. `render.yaml` defines both a worker
and a cron job — enable one. Set the four `R2_*` variables in the dashboard.

Use the worker. A cron job pays container cold start every minute, which on Render
runs longer than the model does, and it forces a full state round trip to R2 on both
ends of every cycle. The loop starts once and holds the feed in memory.

**3. Cloudflare Worker.**

```bash
cd worker
npx wrangler deploy
```

Confirm the `bucket_name` in `wrangler.toml` matches `R2_BUCKET`.

**4. Cloudflare Pages.** Point Pages at this repo with `web/` as the output directory
and no build command. Set `API_BASE` in `web/app.js` to your Worker URL.

## Why state persistence is not optional

Render cron containers are ephemeral. Losing `feed_state.json` between runs does not
crash anything, which is what makes it dangerous — the projection still renders, just
wrong.

`mode_calibration` learns the early/Election Day gap from the *sequence* of batches a
county reports. Without history every county looks like it reported once in a single
dump, which is exactly the case where the gap multiplier is unidentified. It silently
falls back to the assumed 21.04 points and stays there. Turnout calibration degrades
the same way.

## Known limitations

- **civicAPI carries no vote-mode field.** Absentee versus Election Day is inferred,
  never observed. `mode_calibration` is what keeps that honest.
- **`percent_reporting` counts precincts, not votes.** In Michigan those diverge
  badly early, because AV counting boards are not precincts in every county. Turnout
  calibration clamps and ramps for this reason. Watch the `raw_ratio` spread.
- **Margins are two-candidate.** McMorrow is on the ballot and is dropped from the
  denominator, matching how the baselines were built.
- **Margin alone does not identify vote mode.** Volume bounds and finished counties
  do the real work. Pass explicit `mode=` or `theta=` wherever you have ground truth.
