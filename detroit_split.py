# Detroit feed and Wayne County decomposition.
#
# WHY THIS IS THE HIGHEST-VALUE FEED IN THE MODEL
#
#     Wayne is 21% of the projected statewide vote and it is not one electorate.
#     Detroit and the western and downriver suburbs vote very differently, so a
#     partial Wayne count is a biased sample of Wayne, and until now the model
#     could only defend against that with a variance penalty (COUNTY_HETEROGENEITY
#     in vote_mode_inference). That penalty says "distrust partial Wayne." It never
#     says which part of Wayne reported.
#
#     Detroit reports separately, so subtracting it turns a guess into arithmetic:
#
#         non-Detroit Wayne = Wayne (civicAPI) - Detroit (this feed)
#
#     Both halves then get their own baseline, their own observed shift, and their
#     own remaining vote, and Wayne's remainder is the volume-weighted blend.
#
# AND IT CARRIES VOTE MODE, WHICH civicAPI DOES NOT
#
#     The Detroit results page breaks every candidate out by Early Voting,
#     Election Day, Pre-Process Absentee, and Absentee Election Day. Per Wilson,
#     everything that is not Election Day counts as early. That means Detroit needs
#     no theta inference at all — the mode split is observed, not estimated, which
#     is worth more than any amount of prior tuning.
#
#     Note that "Absentee Election Day" is absentee ballots processed on Election
#     Day. They are absentee ballots and belong in the early bucket.
#
# GETTING THE ENDPOINT
#
#     The results site is a single-page app, so the JSON is fetched separately and
#     is not in the page source. In the browser: DevTools -> Network -> filter XHR
#     -> reload the results page -> copy the request URL that returns candidate
#     data. Set it as DETROIT_API_URL and this module handles the rest.
#
#     The parser deliberately does not assume a schema. It walks the JSON looking
#     for objects that carry a candidate name and per-method vote counts, so it
#     should survive whatever nesting Enhanced Voting uses. Run this file directly
#     to dump the shape if it comes back empty.

import json
import os
import re

# --- Wayne composition ------------------------------------------------------

DETROIT_SHARE_OF_WAYNE = 0.39     # Detroit's share of Wayne's total vote
DETROIT_MARGIN = -20.0            # El-Sayed margin in Detroit (Stevens +20)
WAYNE_EX_DETROIT_MARGIN = 26.0    # El-Sayed margin in the rest of Wayne

# 0.39 x -20 + 0.61 x +26 = +8.06, but Wayne's baseline in the model is +8.70.
# With this True the non-Detroit margin is nudged (to +27.05) so Wayne still lands
# on its baseline and the statewide topline is untouched. Set it False to take the
# two sub-margins literally and let Wayne fall to +8.06, which costs about 0.13
# points statewide.
RECONCILE_TO_WAYNE_BASELINE = True

# --- Feed -------------------------------------------------------------------

DETROIT_API_URL = os.environ.get("DETROIT_API_URL", "").strip()
REQUEST_TIMEOUT = 12

EL_SAYED_KEYS = ("el-sayed", "elsayed", "el sayed")
STEVENS_KEYS = ("stevens",)

# Everything that is not Election Day is early.
ELECTION_DAY_METHODS = ("election day",)
NOT_ELECTION_DAY_OVERRIDES = ("absentee election day",)


def classify_method(label: str) -> str:
    """
    Map a vote-method column to 'ed', 'early', or 'skip'.

    Two traps here. 'Absentee Election Day' contains the words 'election day' but
    is an absentee ballot, so it is checked first and routed to early. And a feed
    may name the key 'electionDay' with no separator, so matching is done against
    a space-stripped form as well — otherwise Election Day silently lands in the
    early bucket, which is the single worst way this could fail.

    Anything that looks like a total is skipped rather than bucketed, so a Total
    Votes column cannot be double counted.
    """
    text = re.sub(r"[^a-z ]+", " ", str(label).lower())
    text = " ".join(text.split())
    squashed = text.replace(" ", "")

    if "total" in squashed:
        return "skip"
    for override in NOT_ELECTION_DAY_OVERRIDES:
        if override in text or override.replace(" ", "") in squashed:
            return "early"
    for ed in ELECTION_DAY_METHODS:
        if ed in text or ed.replace(" ", "") in squashed:
            return "ed"
    return "early"


def _matches(name: str, keys) -> bool:
    lowered = str(name).lower()
    return any(k in lowered for k in keys)


def _walk(node, out):
    """
    Recursively hunt for candidate-shaped objects.

    A node qualifies if it has something name-like matching a candidate and any
    numeric field whose key looks like a vote method or a total.
    """
    if isinstance(node, list):
        for item in node:
            _walk(item, out)
        return
    if not isinstance(node, dict):
        return

    name = None
    for key in ("name", "candidateName", "ballotOptionName", "title", "label",
                "choiceName", "option"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break

    if name and (_matches(name, EL_SAYED_KEYS) or _matches(name, STEVENS_KEYS)):
        who = "el_sayed" if _matches(name, EL_SAYED_KEYS) else "stevens"
        out.setdefault(who, {"name": name, "early": 0, "ed": 0, "total": 0})

        # Per-method counts may be nested under a list, or flattened as sibling
        # keys. Handle both.
        methods = None
        for key in ("voteMethods", "methods", "voteTypes", "results", "breakdown",
                    "votesByMethod"):
            if isinstance(node.get(key), list):
                methods = node[key]
                break

        if methods:
            for m in methods:
                if not isinstance(m, dict):
                    continue
                label = next((m[k] for k in ("name", "method", "type", "label",
                                             "voteMethod")
                              if isinstance(m.get(k), str)), "")
                votes = next((m[k] for k in ("votes", "count", "total", "value")
                              if isinstance(m.get(k), (int, float))), None)
                if votes is None:
                    continue
                bucket = classify_method(label)
                if bucket != "skip":
                    out[who][bucket] += int(votes)
        else:
            for key, value in node.items():
                if not isinstance(value, (int, float)):
                    continue
                if re.search(r"early|absentee|election\s*day", str(key), re.I):
                    bucket = classify_method(key)
                    if bucket != "skip":
                        out[who][bucket] += int(value)

        for key in ("totalVotes", "total", "votes"):
            if isinstance(node.get(key), (int, float)):
                out[who]["total"] = max(out[who]["total"], int(node[key]))

    for value in node.values():
        _walk(value, out)


def parse_detroit(payload: dict) -> dict:
    """
    Pull El-Sayed and Stevens out of a Detroit results payload, split by mode.

    Returns {'el_sayed': {...}, 'stevens': {...}} with early/ed/total counts, or
    an empty dict if nothing matched.
    """
    out = {}
    _walk(payload, out)

    # If the per-method walk found nothing but a total exists, treat the total as
    # unallocated rather than silently reporting zeros.
    for who, rec in out.items():
        if rec["early"] + rec["ed"] == 0 and rec["total"] > 0:
            rec["unallocated"] = rec["total"]
        else:
            rec["total"] = max(rec["total"], rec["early"] + rec["ed"])
    return out


def fetch_detroit(url: str = None, timeout: int = REQUEST_TIMEOUT) -> dict:
    """GET and parse the Detroit feed. Returns {} when unconfigured."""
    target = url or DETROIT_API_URL
    if not target:
        return {}
    import requests
    response = requests.get(target, timeout=timeout,
                            headers={"Accept": "application/json"})
    response.raise_for_status()
    return parse_detroit(response.json())


# --- Wayne decomposition ----------------------------------------------------

def wayne_sub_baselines() -> tuple:
    """(detroit_margin, rest_of_wayne_margin) after optional reconciliation."""
    if not RECONCILE_TO_WAYNE_BASELINE:
        return DETROIT_MARGIN, WAYNE_EX_DETROIT_MARGIN
    from michigan_primary_model import build_michigan_county_data
    c = build_michigan_county_data()
    wayne = float(c.loc[c.county == "Wayne", "margin"].iloc[0])
    rest = (wayne - DETROIT_SHARE_OF_WAYNE * DETROIT_MARGIN) / (1 - DETROIT_SHARE_OF_WAYNE)
    return DETROIT_MARGIN, rest


def wayne_remainder(detroit: dict,
                    wayne_counted_el_sayed: int,
                    wayne_counted_stevens: int,
                    wayne_total_projected: float,
                    wayne_early_pool: float,
                    wayne_ed_pool: float) -> dict:
    """
    Split Wayne into Detroit and non-Detroit and project the remainder.

    Args:
        detroit: parse_detroit() output
        wayne_counted_*: Wayne totals from civicAPI, which INCLUDE Detroit
        wayne_total_projected: the model's projected Wayne turnout
        wayne_early_pool / wayne_ed_pool: Wayne's projected mode pools

    Returns a dict with the blended remainder margin for Wayne plus the pieces,
    or {'available': False} when there is nothing usable.
    """
    if not detroit or "el_sayed" not in detroit or "stevens" not in detroit:
        return {"available": False, "reason": "no Detroit data"}

    d_es = detroit["el_sayed"]
    d_st = detroit["stevens"]
    d_counted = d_es["total"] + d_st["total"]
    if d_counted <= 0:
        return {"available": False, "reason": "Detroit reporting zero"}

    wayne_counted = wayne_counted_el_sayed + wayne_counted_stevens

    # CONSISTENCY GATE.
    #
    # Detroit and civicAPI refresh independently, so Detroit will sometimes be
    # ahead. That is not a double count — Detroit votes never enter the totals,
    # which come only from civicAPI — but it does break the decomposition. If
    # Detroit reports 46,000 votes and civicAPI's Wayne still shows 0, the split
    # concludes the suburbs are entirely outstanding and hands back a
    # suburb-weighted remainder, while simulate() applies that remainder to the
    # whole county including the Stevens-heavy Detroit votes it does not yet know
    # about. Wayne gets inflated.
    #
    # So the override is only offered when civicAPI's Wayne count can actually
    # contain the Detroit count. Otherwise Wayne falls back to county-level
    # inference until the feeds line up, which is a lost enhancement rather than
    # a wrong number.
    tolerance = wayne_counted * 0.01 + 200      # absorbs refresh jitter
    if wayne_counted <= 0:
        return {"available": False, "reason": "civicAPI Wayne has not reported yet",
                "detroit_counted": d_counted, "wayne_counted": 0}
    if d_counted > wayne_counted + tolerance:
        return {"available": False,
                "reason": "feeds out of step: Detroit ahead of civicAPI",
                "detroit_counted": d_counted, "wayne_counted": wayne_counted}

    d_margin_base, rest_margin_base = wayne_sub_baselines()

    # Projected sizes of each half
    d_total = wayne_total_projected * DETROIT_SHARE_OF_WAYNE
    rest_total = wayne_total_projected - d_total

    # Observed
    d_margin = (d_es["total"] - d_st["total"]) / d_counted * 100.0
    rest_counted = wayne_counted - d_counted
    rest_margin = None
    if rest_counted > 0:
        rest_es = wayne_counted_el_sayed - d_es["total"]
        rest_st = wayne_counted_stevens - d_st["total"]
        if rest_es >= 0 and rest_st >= 0:
            rest_margin = (rest_es - rest_st) / rest_counted * 100.0
        else:
            # Within tolerance on the total but negative on a candidate. Treat the
            # suburbs as not yet reporting rather than inventing a margin.
            rest_counted = 0

    # Shift each half toward what it is actually doing, damped by completeness.
    # A half that has barely reported keeps its baseline.
    d_done = min(d_counted / max(d_total, 1.0), 1.0)
    d_proj = d_margin_base + (d_margin - d_margin_base) * d_done

    if rest_margin is None:
        rest_proj, rest_done = rest_margin_base, 0.0
    else:
        rest_done = min(rest_counted / max(rest_total, 1.0), 1.0)
        rest_proj = rest_margin_base + (rest_margin - rest_margin_base) * rest_done

    d_left = max(d_total - d_counted, 0.0)
    rest_left = max(rest_total - max(rest_counted, 0), 0.0)

    # Conserve volume against the simulator. simulate() computes Wayne's remaining
    # vote as (projected total - counted), and the two halves must sum to exactly
    # that. They can drift apart when Detroit's reported count disagrees with the
    # 39% share assumption — if Detroit has posted less than 39% of what Wayne has
    # counted, the split would otherwise claim Detroit vote is still outstanding
    # that Wayne says is already in. Rescaling keeps the Detroit-to-suburb ratio
    # while forcing the total to match.
    actual_left = max(wayne_total_projected - wayne_counted, 0.0)
    raw_left = d_left + rest_left
    if raw_left > 0:
        scale = actual_left / raw_left
        d_left *= scale
        rest_left *= scale
    elif actual_left > 0:
        d_left = actual_left * DETROIT_SHARE_OF_WAYNE
        rest_left = actual_left - d_left
    left = d_left + rest_left

    if left <= 0:
        blended = (d_total * d_proj + rest_total * rest_proj) / max(wayne_total_projected, 1.0)
    else:
        blended = (d_left * d_proj + rest_left * rest_proj) / left

    # Detroit's mode split is OBSERVED, not inferred, so it can pin down Wayne's
    # theta directly for the Detroit share of the count. Where the suburbs have
    # also reported, their mode split is still unknown, so the two are blended by
    # volume: the observed part carries its real value and only the unobserved
    # remainder falls back to the prior.
    d_early = d_es["early"] + d_st["early"]
    d_ed = d_es["ed"] + d_st["ed"]
    d_mode_known = (d_early + d_ed) > 0

    theta_observed = None
    theta_coverage = 0.0
    if d_mode_known:
        d_mode_total = d_early + d_ed
        theta_detroit = d_early / d_mode_total
        theta_coverage = min(d_mode_total / max(wayne_counted, 1.0), 1.0)
        if rest_counted <= 0 or theta_coverage >= 0.999:
            theta_observed = theta_detroit
        else:
            # Suburban mode split unknown; fall back to the county's overall early
            # share for that portion only.
            rest_prior = wayne_early_pool / max(wayne_early_pool + wayne_ed_pool, 1.0)
            theta_observed = (d_mode_total * theta_detroit
                              + rest_counted * rest_prior) / max(wayne_counted, 1.0)

    return {
        "available": True,
        "detroit_counted": d_counted,
        "wayne_counted": wayne_counted,
        "detroit_share_of_count": round(d_counted / max(wayne_counted, 1), 3),
        "detroit_margin": round(d_margin, 2),
        "detroit_projected_margin": round(d_proj, 2),
        "detroit_pct_in": round(d_done * 100, 1),
        "detroit_early": d_early,
        "detroit_ed": d_es["ed"] + d_st["ed"],
        "detroit_mode_known": d_mode_known,
        "theta_observed": None if theta_observed is None else round(theta_observed, 4),
        "theta_coverage": round(theta_coverage, 3),
        "rest_counted": int(max(rest_counted, 0)),
        "rest_margin": None if rest_margin is None else round(rest_margin, 2),
        "rest_projected_margin": round(rest_proj, 2),
        "rest_pct_in": round(rest_done * 100, 1),
        "remainder_margin": round(blended, 2),
        "remaining_detroit": int(d_left),
        "remaining_rest": int(rest_left),
    }


if __name__ == "__main__":
    d, rest = wayne_sub_baselines()
    print("Detroit baseline        %+.2f" % d)
    print("Rest of Wayne baseline  %+.2f" % rest)
    print("implied Wayne           %+.2f" % (
        DETROIT_SHARE_OF_WAYNE * d + (1 - DETROIT_SHARE_OF_WAYNE) * rest))
    print()
    for label in ["Early Voting", "Election Day", "Pre-Process Absentee",
                  "Absentee Election Day", "Total Votes"]:
        print("  %-24s -> %s" % (label, classify_method(label)))

    if DETROIT_API_URL:
        print()
        print("fetching %s" % DETROIT_API_URL)
        try:
            import requests
            raw = requests.get(DETROIT_API_URL, timeout=REQUEST_TIMEOUT).json()
            parsed = parse_detroit(raw)
            print(json.dumps(parsed, indent=2))
            if not parsed:
                print("\nNo candidates matched. Top-level shape:")
                print(json.dumps(raw, indent=2)[:2000])
        except Exception as exc:
            print("fetch failed: %s" % exc)
    else:
        print("\nDETROIT_API_URL not set. Set it to the XHR URL from DevTools.")
