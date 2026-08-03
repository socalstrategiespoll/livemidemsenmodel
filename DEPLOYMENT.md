# Deployment: GitHub upload + Render cron

No local setup, no git commands, no terminal. Everything happens in three browser
tabs: GitHub, Cloudflare, Render.

About 25 minutes. Website comes later.

---

## What you need

- GitHub account
- Cloudflare account (free tier)
- Render account (connect it to GitHub when prompted)
- The `mi-senate-model-repo.zip` file, unzipped on your desktop

No civicAPI key. The API is open.

---

## The one thing that is not optional

You have to do Step 2 (Cloudflare R2) even though it feels like it belongs to the
website work.

Render cron gives you a brand new container every run and an ephemeral filesystem.
Nothing written to disk survives to the next minute. That matters because
`mode_calibration` learns the real early-vs-Election-Day gap from the *sequence* of
batches each county reports. With no persistent state, every county looks like it
reported once, in a single dump, which is exactly the case where the gap cannot be
identified. It falls back to the assumed 21.04 points and stays there all night.

Nothing breaks visibly. The projection still renders. It is just wrong in a way you
would not notice.

So: R2 first, then Render.

---

## Step 1. Get the files onto GitHub

### 1.1 Unzip

Unzip `mi-senate-model-repo.zip` somewhere you can find it. You should have a folder
containing `model/`, `runner/`, `web/`, `worker/`, `render.yaml`,
`requirements.txt`, `README.md`, `DEPLOYMENT.md`.

### 1.2 Create an empty repo

1. Go to **github.com/new**
2. Repository name: `mi-senate-model`
3. Visibility: **Private** is fine (Render reads private repos once authorized)
4. **Do not** check "Add a README file"
5. **Do not** add a .gitignore or license
6. Click **Create repository**

You land on a page that says "Quick setup". Ignore the command-line instructions.

### 1.3 Upload

1. On that page, click the link **uploading an existing file**
   (or go to `github.com/<you>/mi-senate-model/upload/main`)
2. Open your unzipped folder in a file browser
3. Select **everything inside** the folder, the `model` folder, the `runner` folder,
   `web`, `worker`, `render.yaml`, `requirements.txt`, `README.md`, `DEPLOYMENT.md`
4. Drag all of it onto the GitHub upload area

GitHub's uploader accepts folders and preserves the directory structure. Do not drag
the outer folder itself, drag its contents, or everything ends up nested one level
too deep and Render will not find `render.yaml`.

5. Commit message: `Michigan Senate primary live model`
6. Click **Commit changes**

### 1.4 Check the structure

Your repo root should look exactly like this:

```
model/
runner/
web/
worker/
DEPLOYMENT.md
README.md
render.yaml
requirements.txt
```

If instead you see a single folder like `mi-senate-model/` containing everything,
you dragged the wrong thing. Delete the repo and redo 1.3, dragging the contents.

If `.gitignore` did not come across because your file browser hides dotfiles, that is
harmless. It only matters for local work, which you are not doing.

---

## Step 2. Cloudflare R2

### 2.1 Create the bucket

1. Log in at **dash.cloudflare.com**
2. In the left sidebar find **R2** (may be nested under Storage, or listed as
   "R2 Object Storage")
3. First time only: Cloudflare asks for a payment method even on the free tier. Free
   covers 10 GB. This project uses a few megabytes.
4. Click **Create bucket**
5. Name: `mi-senate-model`
6. Location: **Automatic**
7. Click **Create bucket**

Leave public access **disabled**. Nothing needs to read this bucket from the browser
yet, and when the website comes later it will read through a Worker rather than
directly.

### 2.2 Copy your Account ID

On the R2 page, find **Account ID** in the right sidebar. A 32-character hex string.

It is also in the dashboard URL: `dash.cloudflare.com/<account-id>/r2`

### 2.3 Create an API token

1. From the R2 page, click **Manage API Tokens** (right sidebar or an API tab)
2. Click **Create API Token**
3. Name: `mi-senate-model-runner`
4. Permissions: **Object Read & Write**
5. Specify bucket: pick `mi-senate-model`, not "All buckets"
6. Click **Create API Token**

You now see **Access Key ID** and **Secret Access Key**.

**Copy the secret now.** It is shown once and cannot be retrieved. If you lose it,
delete the token and make a new one.

### 2.4 Park all four values

Paste these somewhere for Step 3:

```
R2_ACCOUNT_ID        = <from 2.2>
R2_ACCESS_KEY_ID     = <from 2.3>
R2_SECRET_ACCESS_KEY = <from 2.3>
R2_BUCKET            = mi-senate-model
```

Watch for trailing spaces or newlines when you paste. That is the single most common
way this goes wrong.

---

## Step 3. Render

### 3.1 Create the Blueprint

1. Log in at **dashboard.render.com**
2. **New +** (top right) → **Blueprint**
3. Connect GitHub if prompted, and grant access to `mi-senate-model`
4. Select the repo
5. Render reads `render.yaml` and shows one service: **mi-model-cron**
6. Blueprint name: anything
7. Click **Apply** / **Create**

If Render says it cannot find `render.yaml`, your files are nested one folder too
deep. Go back to 1.4.

### 3.2 Add the four secrets

Open the **mi-model-cron** service → **Environment**.

The blueprint already set `PYTHON_VERSION`, `RACE_ID`, `N_SIMS`, and
`ARCHIVE_HISTORY`. Add these four:

| Key | Value |
|---|---|
| `R2_ACCOUNT_ID` | from 2.2 |
| `R2_ACCESS_KEY_ID` | from 2.3 |
| `R2_SECRET_ACCESS_KEY` | from 2.3 |
| `R2_BUCKET` | `mi-senate-model` |

**Save Changes.** Render rebuilds.

### 3.3 Trigger a run and read the log

The cron schedule is `* * * * *`, so it fires at the top of the next minute. You can
also force one immediately: on the service page, look for **Trigger Run** or
**Run Now**.

Open **Logs**. The first build takes two to four minutes (pip installing numpy,
pandas, scipy). Then every run prints something like:

```
state backend: r2:mi-senate-model
loaded 0 counties with history
   matched: Abdul El-Sayed vs Haley Stevens
[HH:MM:SS] 0.0% counted | 0 cty | El-Sayed +14.6 [+3.7, +25.4] | win 98.8% | gap x1.00 | turnout 1,419,960
```

Since you are not testing locally, this log is your entire pre-flight. Three lines
matter.

---

## Step 4. Read the log carefully. This replaces the local test.

### 4.1 `state backend:`

**Want:** `state backend: r2:mi-senate-model`

**Bad:** `state backend: local:./state`, followed by a loud warning block.

If you see the local one, one of the four R2 values is wrong. Recheck for trailing
whitespace. Do not proceed until this reads `r2:`. On a cron job this is the failure
that quietly ruins the night.

### 4.2 `matched:`

**Want:** `matched: Abdul El-Sayed vs Haley Stevens` (whatever the feed actually
calls them)

**Bad:** `!! CANDIDATE MATCH FAILED: el_sayed=None stevens=...`

If it failed, the feed spells a name differently than the matcher expects. Fix it in
GitHub directly, no local setup needed:

1. Go to `civicapi_feed.py` in your repo
2. Click the pencil icon to edit
3. Near the top find:
   ```python
   EL_SAYED_KEYS = ("el-sayed", "elsayed", "el sayed")
   STEVENS_KEYS = ("stevens",)
   ```
4. Add the actual spelling from the error, lowercase
5. Commit directly to `main`

Render redeploys automatically within a few minutes.

### 4.3 `!! UNMATCHED COUNTIES:`

If this line appears, a county name in the feed did not match the model's 83. Those
counties are being silently dropped from the projection.

Fix the same way: edit `civicapi_feed.py` on GitHub, extend
`normalize_county()` to handle the spelling, commit to `main`.

If the line is absent, all counties matched.

---

## Step 5. Confirm R2 is receiving writes

Go back to the Cloudflare R2 dashboard, open `mi-senate-model`, click into the object
list.

Within two minutes you should see:

- `projection.json`
- `feed_state.json`
- `history/` with a file appearing every minute

**If the bucket is empty**, the job is writing to a container that evaporates. Return
to 3.2.

You can download `projection.json` from the dashboard to eyeball the full output.
That is how you inspect results until the website exists.

---

## Step 6. The check that has to wait until results start

At around 8:30pm ET, once counties are actually reporting, open the Render logs and
find a recent run. You want:

```
loaded N counties with history
```

**N must be greater than zero.** Before polls close it will read 0, which tells you
nothing, because there are no results yet. After results start, a 0 means state is
not persisting between runs and mode calibration is blind.

This is the single most important check in the whole guide, and it cannot be done
early.

---

## Step 7. Watching it run

Everything is in the log line and in `projection.json`.

**`gap x1.00`** is the mode gap multiplier. It starts at 1.00 and only moves once
counties finish counting and there is batch structure to learn from. If it settles
well away from 1.00, your +8 early-vote assumption was off. If it is still exactly
1.00 late in the night with many counties finished, check Step 6.

**`turnout`** is the live-recalibrated statewide projection, starting from 1,419,960.
Open `projection.json` for `turnout.pooled_ratio`. Roughly 0.9 to 1.4 means precinct
reporting is tracking vote share. Pinned at 2.50 means it is hitting the clamp, which
is expected in Michigan early on because absentee counting boards are not precincts.
Leave the clamp alone.

**Do not read the win probability as a call early.** At 20% reporting the model is
largely handing your prior back to you. The mode inference bias runs against whoever
is overperforming, so early numbers understate a real surge.

**A failed cycle is not a problem.** Each run catches its own exceptions, logs them,
and keeps the last good state. A civicAPI hiccup costs one update.

---

## Changing a parameter mid-night

Edit the file on GitHub, commit to `main`, Render picks it up on the next build.
State is in R2, so you lose nothing.

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
| Render cannot find `render.yaml` | files nested one folder deep | redo Step 1.3, drag folder *contents* |
| `state backend: local:./state` | R2 env var wrong | recheck all four, look for trailing whitespace |
| `!! CANDIDATE MATCH FAILED` | feed spells names differently | edit `EL_SAYED_KEYS` / `STEVENS_KEYS` on GitHub |
| `!! UNMATCHED COUNTIES` | county name variant | extend `normalize_county()` on GitHub |
| R2 bucket empty | state not persisting | see Step 3.2 |
| `loaded 0 counties` after 8:30pm | state not persisting | see Step 3.2 |
| `gap x1.00` late with many counties done | no batch history, or single-dump counties | check Step 6 first |
| `pooled_ratio` at 2.50 | precinct reporting diverging from votes | expected early. Leave it |
| Build fails on numpy/scipy | Python version | confirm `PYTHON_VERSION` is `3.12.3` |

---

## Timeline

| Time (ET) | Do |
|---|---|
| Now | Steps 1 through 5 |
| Any time before 7pm | Confirm the log still shows `r2:` and `matched:` each run |
| 8:00pm | Polls close |
| 8:30pm | **Step 6.** N must be greater than zero |
| Overnight | Watch `gap` and `turnout` |
| Later | Website, reading from this same bucket |
