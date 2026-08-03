"""
State and artifact storage for the live runner.

WHY THIS MODULE EXISTS AT ALL

    Render cron jobs get a fresh container every run and an ephemeral filesystem.
    Nothing written to disk survives to the next invocation. That is fatal here, and
    not in an obvious way: the projection would still render, it would just be wrong.

    mode_calibration learns the early/Election Day gap from the SEQUENCE of batches a
    county reports. Lose the batch history and every county looks like it reported
    once, in a single dump, which is exactly the case where kappa is unidentified.
    The gap silently falls back to the assumed 21.04 points and stays there all
    night. Turnout calibration degrades the same way.

    So state has to live somewhere the container does not. This module puts it in
    Cloudflare R2, which is also where the public projection.json goes, so the site
    and the state share one backend.

BACKENDS

    R2Store     - S3-compatible, used in production. Needs boto3 and four env vars.
    LocalStore  - plain files, used for development and as an automatic fallback so
                  a missing credential degrades to "works on my machine" instead of
                  crashing the night.
"""

import json
import os
from datetime import datetime, timezone


STATE_KEY = "feed_state.json"
PROJECTION_KEY = "projection.json"
HISTORY_KEY_PREFIX = "history/"


class LocalStore:
    """Filesystem backend. Development, and the fallback when R2 is unconfigured."""

    def __init__(self, root: str = "./state"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        path = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_json(self, key: str):
        try:
            with open(self._path(key)) as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def put_json(self, key: str, payload: dict, public: bool = False) -> None:
        with open(self._path(key), "w") as handle:
            json.dump(payload, handle, indent=2)

    def describe(self) -> str:
        return "local:{}".format(self.root)


class R2Store:
    """
    Cloudflare R2 over the S3 API.

    Env vars:
        R2_ACCOUNT_ID
        R2_ACCESS_KEY_ID
        R2_SECRET_ACCESS_KEY
        R2_BUCKET
    """

    def __init__(self):
        import boto3  # imported lazily so LocalStore works without it

        account = os.environ["R2_ACCOUNT_ID"]
        self.bucket = os.environ["R2_BUCKET"]
        self.client = boto3.client(
            "s3",
            endpoint_url="https://{}.r2.cloudflarestorage.com".format(account),
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def get_json(self, key: str):
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response["Body"].read())
        except Exception:
            return None

    def put_json(self, key: str, payload: dict, public: bool = False) -> None:
        # Short max-age keeps the site fresh without hammering the origin. The site
        # polls faster than this; Cloudflare's edge absorbs the difference.
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
            CacheControl="public, max-age=20" if public else "no-store",
        )

    def describe(self) -> str:
        return "r2:{}".format(self.bucket)


def get_store():
    """
    Return R2Store when fully configured, LocalStore otherwise.

    Falling back rather than raising is deliberate. A typo in one env var should not
    take the model offline on election night; it should degrade to local state and
    say so loudly in the logs.
    """
    required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    if all(os.environ.get(name) for name in required):
        try:
            return R2Store()
        except Exception as exc:
            print("WARNING: R2 unavailable, falling back to local state: {}".format(exc))
    else:
        missing = [n for n in required if not os.environ.get(n)]
        print("WARNING: R2 not configured (missing {}), using local state".format(
            ", ".join(missing)))
    return LocalStore(os.environ.get("STATE_DIR", "./state"))


def history_key(timestamp: datetime = None) -> str:
    """Timestamped key so every cycle's projection is archived for the post-mortem."""
    stamp = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return "{}{}.json".format(HISTORY_KEY_PREFIX, stamp)
