# Fixing the flattened upload

GitHub's web uploader dropped every file at the repository root instead of keeping
the folders. Rather than fight it, this version of the code expects a flat layout.
Every `.py` sits at the root and the entrypoint is `python run.py --once`.

## Step A. Delete the three junk files

In your repo, click each file, then the trash icon, then **Commit changes**:

- `__init__.py`
- `__init__ (1).py`
- `download`

The two `__init__.py` files were Python package markers. There are no packages any
more, so they do nothing. `download` is an artifact of the upload.

## Step B. Upload the corrected files

1. Go to `github.com/<you>/socalstrategiespoll/upload/main`
2. Unzip `mi-senate-model-flat.zip`
3. Drag **all 12 files** onto the upload area
4. Commit message: `Flat layout`
5. **Commit changes**

Same-named files overwrite cleanly. Nine of the twelve are unchanged; `run.py`,
`store.py`, and `render.yaml` are the ones that actually differ.

## Step C. What the repo should look like

```
civicapi_feed.py
DEPLOYMENT.md
hierarchical_model.py
michigan_primary_model.py
mode_calibration.py
README.md
render.yaml
requirements.txt
run.py
store.py
turnout_calibration.py
vote_method_split.py
vote_mode_inference.py
```

Plus `app.js`, `index.html`, `index.js`, `style.css`, and `wrangler.toml` from your
first upload. Those are website files and do nothing yet. Leave them; they get sorted
out when we build the site.

Nothing else. No folders.

## Step D. Continue

Pick the guide back up at **Step 2, Cloudflare R2**. Everything from there is
unchanged except one detail: the Render command is now `python run.py --once`
rather than `python -m runner.run --once`, and `render.yaml` already says so.
