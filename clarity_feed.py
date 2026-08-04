# Clarity Election Night Reporting client.
#
# WHAT THIS IS WORTH
#
#     Most Michigan counties publish through Clarity (Scytl), and Clarity's detail
#     report breaks every choice out BY VOTE TYPE. civicAPI does not carry mode at
#     all, which is why vote_mode_inference exists and why theta has been the
#     weakest joint in the model. Every county reachable here stops being inferred
#     and starts being observed.
#
#     Detroit did this for Wayne. Clarity can do it for Genesee, Kent, Macomb,
#     Oakland, Washtenaw and the rest — roughly the counties that decide the race.
#
# ENDPOINT SHAPE
#
#     Clarity sites live at:
#         https://results.enr.clarityelections.com/MI/<County>/<electionId>/
#
#     Results are versioned. The current version has to be read first, because
#     every data path is nested under it:
#         .../<electionId>/current_ver.txt          -> e.g. "314028"
#         .../<electionId>/<ver>/json/en/summary.json
#         .../<electionId>/<ver>/reports/detailxml.zip
#
#     summary.json gives contest totals with no mode breakdown. The vote-type data
#     is in detailxml.zip, which unzips to detail.xml containing, per contest, per
#     choice, a <VoteType name="..."> element with per-precinct counts.
#
#     Election IDs differ per county and change every election, so they have to be
#     supplied. Genesee's for 4 August 2026 is 126773, published by the county
#     clerk. The others are on each county clerk's results page, or visible in the
#     URL when you open their results site.
#
# STATUS
#
#     OPT-IN AND OFF BY DEFAULT. This has not been run against a live Clarity
#     endpoint. Enable one county, confirm the parsed numbers match what the
#     county's own page shows, then add the rest. A silent parse failure here
#     would feed wrong mode data into counties that are currently working fine on
#     inference, which is worse than not having it.

import io
import os
import re
import zipfile
import xml.etree.ElementTree as ET

BASE = "https://results.enr.clarityelections.com/MI"
TIMEOUT = 12

# county -> Clarity election id for 4 August 2026. Add as you confirm them.
# Empty means this module does nothing at all.
CLARITY_ELECTIONS = {
    # 'Genesee': '126773',
}

_env = os.environ.get("CLARITY_ELECTIONS", "").strip()
if _env:
    # CLARITY_ELECTIONS="Genesee:126773,Kent:126801"
    for pair in _env.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            CLARITY_ELECTIONS[k.strip()] = v.strip()

CONTEST_PATTERN = re.compile(r"united\s+states\s+senator", re.I)
PARTY_PATTERN = re.compile(r"\bdem", re.I)

EL_SAYED_KEYS = ("el-sayed", "elsayed", "el sayed")
STEVENS_KEYS = ("stevens",)

# Clarity vote-type labels vary by county. Everything that is not Election Day is
# early, matching how the model buckets Detroit.
ED_LABELS = ("election day", "electionday", "polling", "precinct")
NOT_ED_OVERRIDES = ("absentee election day", "avcb election day")


def classify_vote_type(label: str) -> str:
    text = " ".join(re.sub(r"[^a-z ]+", " ", str(label).lower()).split())
    squashed = text.replace(" ", "")
    if "total" in squashed:
        return "skip"
    for override in NOT_ED_OVERRIDES:
        if override in text or override.replace(" ", "") in squashed:
            return "early"
    for ed in ED_LABELS:
        if ed in text or ed.replace(" ", "") in squashed:
            return "ed"
    return "early"


def _matches(name, keys):
    lowered = str(name).lower()
    return any(k in lowered for k in keys)


def current_version(county: str, election_id: str, session=None) -> str:
    import requests
    getter = session.get if session else requests.get
    url = "{}/{}/{}/current_ver.txt".format(BASE, county, election_id)
    r = getter(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text.strip()


def fetch_detail_xml(county: str, election_id: str, version: str = None,
                     session=None) -> bytes:
    """Download and unzip detailxml.zip, returning the raw detail.xml bytes."""
    import requests
    getter = session.get if session else requests.get
    version = version or current_version(county, election_id, session)
    url = "{}/{}/{}/{}/reports/detailxml.zip".format(BASE, county, election_id, version)
    r = getter(url, timeout=TIMEOUT)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".xml"))
        return z.read(name)


def parse_detail(xml_bytes: bytes) -> dict:
    """
    Pull the Senate Democratic primary out of a Clarity detail.xml, split by mode.

    Returns {'el_sayed': {'early': n, 'ed': n, 'total': n}, 'stevens': {...},
             'contest': <name>} or {} if the contest was not found.

    Structure is roughly:
        <Contest text="United States Senator (DEM)">
          <Choice text="Abdul El-Sayed" totalVotes="...">
            <VoteType name="Election Day" votes="...">
              <Precinct name="..." votes="..."/>
            </VoteType>
    Only the VoteType-level totals are used; precinct detail is ignored.
    """
    root = ET.fromstring(xml_bytes)
    out = {}

    for contest in root.iter():
        if not contest.tag.lower().endswith("contest"):
            continue
        label = contest.get("text") or contest.get("name") or ""
        if not CONTEST_PATTERN.search(label) or not PARTY_PATTERN.search(label):
            continue

        found = {}
        for choice in contest:
            if not choice.tag.lower().endswith("choice"):
                continue
            cname = choice.get("text") or choice.get("name") or ""
            if _matches(cname, EL_SAYED_KEYS):
                who = "el_sayed"
            elif _matches(cname, STEVENS_KEYS):
                who = "stevens"
            else:
                continue

            rec = found.setdefault(who, {"name": cname, "early": 0, "ed": 0, "total": 0})
            try:
                rec["total"] = int(choice.get("totalVotes") or 0)
            except ValueError:
                pass

            for vt in choice:
                if not vt.tag.lower().endswith("votetype"):
                    continue
                bucket = classify_vote_type(vt.get("name") or "")
                if bucket == "skip":
                    continue
                votes = vt.get("votes")
                if votes is None:
                    # Some exports put counts only on child precincts.
                    votes = sum(int(p.get("votes") or 0) for p in vt)
                rec[bucket] += int(votes)

        if "el_sayed" in found and "stevens" in found:
            out = found
            out["contest"] = label
            break

    for who in ("el_sayed", "stevens"):
        if who in out:
            rec = out[who]
            rec["total"] = max(rec["total"], rec["early"] + rec["ed"])
    return out


def fetch_county(county: str, election_id: str, session=None) -> dict:
    return parse_detail(fetch_detail_xml(county, election_id, session=session))


def fetch_all(elections: dict = None) -> dict:
    """
    Fetch every configured county. One county failing never stops the rest.

    Returns {county: {'el_sayed': {...}, 'stevens': {...}, 'theta': float}}
    """
    elections = elections if elections is not None else CLARITY_ELECTIONS
    if not elections:
        return {}

    import requests
    session = requests.Session()
    out = {}

    for county, election_id in elections.items():
        try:
            parsed = fetch_county(county, election_id, session=session)
            if not parsed or "el_sayed" not in parsed:
                continue
            es, st = parsed["el_sayed"], parsed["stevens"]
            early = es["early"] + st["early"]
            ed = es["ed"] + st["ed"]
            if early + ed <= 0:
                continue
            parsed["theta"] = early / (early + ed)
            parsed["counted"] = es["total"] + st["total"]
            parsed["margin"] = ((es["total"] - st["total"])
                                / max(parsed["counted"], 1) * 100.0)
            out[county] = parsed
        except Exception as exc:
            print("   Clarity %s failed: %s" % (county, exc), flush=True)

    return out


def theta_overrides(clarity: dict, reported: dict, tolerance: float = 0.03) -> dict:
    """
    Turn Clarity results into theta overrides, but only where Clarity and civicAPI
    agree on how much has been counted.

    Same hazard as Detroit: the two feeds refresh independently. If Clarity is
    ahead of or behind civicAPI, its mode split describes a different set of
    ballots than the ones the model thinks are counted, and applying it would
    misallocate the remainder. Disagreement means fall back to inference.
    """
    out = {}
    for county, rec in clarity.items():
        civic = reported.get(county)
        if not civic:
            continue
        civic_counted = civic["el_sayed"] + civic["stevens"]
        if civic_counted <= 0 or rec["counted"] <= 0:
            continue
        drift = abs(rec["counted"] - civic_counted) / max(civic_counted, 1)
        if drift <= tolerance:
            out[county] = rec["theta"]
    return out


if __name__ == "__main__":
    for label in ["Election Day", "Absentee", "Early Voting", "AVCB Election Day",
                  "Absentee Election Day", "Total", "Polling Place", "Early Voting Site"]:
        print("  %-24s -> %s" % (label, classify_vote_type(label)))

    if not CLARITY_ELECTIONS:
        print("\nNo counties configured. Set CLARITY_ELECTIONS, e.g.")
        print('  CLARITY_ELECTIONS="Genesee:126773,Kent:126801"')
    else:
        for county, eid in CLARITY_ELECTIONS.items():
            try:
                ver = current_version(county, eid)
                print("\n%s (election %s, version %s)" % (county, eid, ver))
                rec = fetch_county(county, eid)
                if rec:
                    es, st = rec["el_sayed"], rec["stevens"]
                    print("  contest: %s" % rec.get("contest"))
                    print("  El-Sayed  early %s  ed %s  total %s"
                          % (es["early"], es["ed"], es["total"]))
                    print("  Stevens   early %s  ed %s  total %s"
                          % (st["early"], st["ed"], st["total"]))
                else:
                    print("  contest not found")
            except Exception as exc:
                print("  failed: %s" % exc)
