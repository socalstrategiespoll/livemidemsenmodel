# Deploy: step by step

Two things to stand up.

1. **Render** runs the model and serves it at a URL. About 15 minutes.
2. **Cloudflare Pages** serves the site, which reads that URL. About 5 minutes.

No local setup, no terminal, no API tokens, no object storage. Everything happens in
three browser tabs: GitHub, Render, Cloudflare.

Everything is flat. No folders anywhere, because GitHub's web uploader does not
preserve them.

---

## Part 1 — Clean up the repo

Your repo currently has files from the cron version that no longer exist. Delete
them or the repo will be confusing later. It will still deploy either way, but do it
now while it is quick.

For each file below: click it in GitHub, click the **trash can** icon at the top
right of the file view, scroll down, click **Commit changes**.

- `run.py`
- `store.py`
- `FIX.md`
- `DEPLOYMENT.md`

`run.py` was the cron entrypoint. `server.py` replaces it.

---

## Part 2 — Upload the new files

### 2.1 Unzip

Unzip `mi-senate-model-web.zip`. You get 14 files, no folders.

### 2.2 Upload

1. Go to `github.com/<you>/socalstrategiespoll/upload/main`
   (or **Add file → Upload files** from the repo page)
2. Select all 14 files in your unzipped folder
3. Drag them onto the upload area
4. Commit message: `Web service and site`
5. Click **Commit changes**

Files with the same name overwrite cleanly. You do not need to delete anything first
except the four in Part 1.

### 2.3 Check

Your repo should now contain exactly these 14 files and nothing else:

```
README.md
app.js
civicapi_feed.py
hierarchical_model.py
index.html
michigan_primary_model.py
mode_calibration.py
render.yaml
requirements.txt
server.py
style.css
turnout_calibration.py
vote_method_split.py
vote_mode_inference.py
```

No folders. If you see any, something went sideways in the upload.

---

## Part 3 — Render

### 3.1 Create the service

1. Log in at **dashboard.render.com**
2. Click **New +** in the top right
3. Choose **Blueprint**
4. If Render has not seen your GitHub yet, it prompts you to connect. Authorize it and
   grant access to `socalstrategiespoll`
5. Select the `socalstrategiespoll` repo
6. Render reads `render.yaml` and shows **one** service: `mi-senate-model`
7. Blueprint name: anything. `mi-senate-model` is fine
8. Click **Apply** (or **Create**, depending on your dashboard version)

**If Render says it cannot find `render.yaml`**, the file is not at the repo root.
Go back to Part 2.

### 3.2 There is nothing to configure

No environment variables. No secrets. The blueprint sets everything the service
needs. Skip straight to watching it build.

### 3.3 Watch the first build

Open the service, then the **Logs** tab.

The build takes **three to five minutes** the first time, because pip is compiling
and installing numpy, pandas, and scipy. Subsequent deploys are much faster since the
layer is cached.

Once the build finishes you want to see, in this order:

```
poller started: race 84778 every 60s, 20000 sims
serving on :10000
   matched: Abdul El-Sayed vs Haley Stevens
[HH:MM:SS] 0.0% counted | 0 cty | El-Sayed +14.6 [+3.7, +25.4] | win 98.8% | gap x1.00 | turnout 1,419,960
```

and then that last line repeating once a minute.

Before polls close, `0.0% counted | 0 cty` is correct. There are no results yet, so
the model is showing you its pre-election baseline.

### 3.4 Read the log carefully. This is your only pre-flight.

Since you are not testing locally, these lines are how you find problems. Three
things to look for.

**`serving on :10000`** — the HTTP server is up. Without this line the service will
fail Render's health check and get restarted in a loop.

**`matched: Abdul El-Sayed vs Haley Stevens`** — the model found both candidates in
the civicAPI payload.

If you instead see:

```
!! CANDIDATE MATCH FAILED: el_sayed=None stevens=None -- fix EL_SAYED_KEYS / STEVENS_KEYS in civicapi_feed.py
```

the feed spells a name differently than the matcher expects. Fix it on GitHub, no
local setup needed:

1. Open `civicapi_feed.py` in your repo
2. Click the **pencil** icon
3. Near the top find:
   ```python
   EL_SAYED_KEYS = ("el-sayed", "elsayed", "el sayed")
   STEVENS_KEYS = ("stevens",)
   ```
4. Add the actual spelling from the error message, lowercase, as another entry
5. **Commit changes** directly to `main`

Render redeploys automatically in a few minutes.

**No `!! UNMATCHED COUNTIES:` line.** If that line appears, a county name in the feed
did not match the model's 83 and is being silently dropped from the projection. Fix
the same way, by editing `normalize_county()` in `civicapi_feed.py`.

### 3.5 Get your URL

At the top of the service page Render shows the URL, something like:

```
https://mi-senate-model.onrender.com
```

**Copy it.** You need it in Part 4.

### 3.6 Test it in your browser

Open these three, one at a time:

| URL | What you should see |
|---|---|
| `<your-url>/health` | `{"ok": true, "cycles": 3, ...}` with cycles counting up |
| `<your-url>/api/projection` | a large JSON blob with `projection`, `counties`, `diagnostics` |
| `<your-url>/api/history` | a list of past cycles |

If `/api/projection` returns `{"error": "no projection yet"}`, the first cycle has
not finished. Wait a minute and reload.

### 3.7 A note on the plan

`render.yaml` specifies the `starter` plan rather than free. This matters: Render
spins free services down after a period without HTTP traffic, and a spun-down service
stops polling. You would come back to a model that has been asleep for an hour with
no batch history.

If you want to check pricing or change it, do that in the service's **Settings**.

---

## Part 4 — Cloudflare Pages

### 4.1 Point the site at Render

1. In GitHub, open `app.js`
2. Click the **pencil** icon
3. Line 5 reads:
   ```js
   const API_BASE = "https://mi-senate-model.onrender.com";
   ```
4. Replace that with your actual Render URL from 3.5. **No trailing slash.**
5. **Commit changes** to `main`

### 4.2 Create the Pages project

1. In the Cloudflare dashboard, go to **Workers & Pages** in the left sidebar
2. Click **Create**
3. Choose the **Pages** tab
4. Click **Connect to Git**
5. Authorize GitHub if prompted, and grant access to `socalstrategiespoll`
6. Select the repo
7. Click **Begin setup**

### 4.3 Build settings

This is the part people get wrong. Set exactly this:

| Setting | Value |
|---|---|
| Production branch | `main` |
| Framework preset | **None** |
| Build command | **leave completely empty** |
| Build output directory | **`/`** |
| Root directory | leave as `/` |

There is no build. The site is three plain files and Cloudflare just copies them.

Because everything is flat, the output directory is the repo root. The `.py` files
get copied alongside `index.html`. They are never linked to and do no harm.

8. Click **Save and Deploy**

Takes under a minute.

### 4.4 Open the site

Cloudflare gives you a URL like `https://socalstrategiespoll.pages.dev`.

**What you should see:** the status pill in the top right reads **live** in teal with
a timestamp. The headline shows El-Sayed with a margin, the distribution curve renders
underneath it, and the county table says "No counties reporting yet."

**If the pill says "reconnecting":** open the browser console (F12, Console tab).

- A **CORS error** means `API_BASE` in `app.js` is wrong. Recheck 4.1
- A **404 or connection refused** means the Render service is down. Check its logs
- **"waiting for first results"** in the pill means Render is up but has not completed
  a cycle. Wait a minute

### 4.5 Optional: custom domain

In the Pages project, go to **Custom domains** → **Set up a custom domain**. If the
domain is already on Cloudflare, DNS is configured for you.

---

## Part 5 — Before polls close

Two checks, both quick.

**The service is still awake.** Open `<render-url>/health`. `cycles` should be
roughly one per minute since it started. If it is far lower, the service has been
sleeping or restarting.

**The site updates on its own.** Leave the site open for a minute. The timestamp in
the top right should tick forward without you reloading.

---

## Part 6 — Election night

### What the page is telling you

**The distribution is the point.** The curve is drawn from the 20,000 simulations the
model actually ran, not a bell curve fitted to the median. The dark line is the
median, the shaded bands are the 50% and 90% intervals, and the dashed line is a tie.
Early in the night that curve is wide, and that width is the honest answer.

**The county table** shows each county's reported margin against what the model
expected and the swing between them. That is the view worth reading at 9pm, because
it tells you whether a surprise is one county or the whole state.

**Model state** shows what has been learned versus assumed. The number to watch is
**Early vs Election Day gap**. It starts at ×1.00 and only moves once counties finish
counting and there is batch structure to learn from. If it settles well away from
1.00, your +8 early-vote assumption was off. **Counties setting that gap** tells you
how many are informing it. Below about five, it is mostly prior.

### What not to do

**Do not read the win probability as a call early.** At 20% reporting the model is
largely handing your prior back to you. The mode inference bias runs against whoever
is overperforming, so early numbers understate a real surge.

**Do not restart the Render service** unless you have to. State is held in memory, so
a restart throws away the batch history and the gap calibration resets until counties
finish counting again.

### If something breaks

**Feed goes down.** Nothing to do. Each cycle catches its own exceptions and keeps
serving the last good projection. The site's timestamp shows how stale it is, and the
pill turns orange after three minutes.

**Need to change a parameter.** Edit the file on GitHub, commit to `main`. Render
redeploys in a few minutes. You lose in-memory state, so weigh that against the fix.

| Constant | File | Controls |
|---|---|---|
| `SEQUENTIAL_MIXING` | `mode_calibration.py` | how hard "early clears first" bites |
| `FULL_TRUST_PCT` | `turnout_calibration.py` | when implied turnout replaces the prior |
| `CLAMP` | `turnout_calibration.py` | how far implied turnout may stray |
| `SIGMA_STATE` | `hierarchical_model.py` | width of the whole uncertainty band |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Render cannot find `render.yaml` | file not at repo root | redo Part 2 |
| Build fails on numpy or scipy | Python version | confirm `PYTHON_VERSION` is `3.12.3` in the service Environment |
| No `serving on` line, service restarts | server did not bind | check logs for a traceback above it |
| `!! CANDIDATE MATCH FAILED` | feed spells names differently | edit `EL_SAYED_KEYS` / `STEVENS_KEYS` on GitHub |
| `!! UNMATCHED COUNTIES` | county name variant | edit `normalize_county()` on GitHub |
| `/api/projection` says no projection yet | first cycle not done | wait 60 seconds |
| Site pill stuck on "reconnecting" | wrong `API_BASE`, or Render down | check browser console for CORS vs 404 |
| Site shows nothing but the header | Pages output directory wrong | should be `/`, not `web` |
| `cycles` in `/health` far below one per minute | service sleeping or restarting | check the plan, check logs for crashes |
| Gap stuck at ×1.00 late in the night | no batch history, or single-dump counties | check **Counties setting that gap** first |

---

## Timeline

| When | Do |
|---|---|
| Now | Parts 1 through 4 |
| Any time before 7pm ET | Part 5 |
| 8:00pm ET | Polls close, results start |
| Through the night | Part 6 |
