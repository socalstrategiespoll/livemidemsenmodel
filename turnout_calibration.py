"""
Live turnout recalibration from civicAPI percent reporting.

The 1.42M turnout projection and its county split are a prior. Once a county starts
reporting, the feed gives a direct read: if a county has counted N votes at P percent
reporting, its total is heading for roughly N / P. That estimate replaces the prior
for that county.

Counties at exactly 0 percent are left alone. There is nothing to divide by and
nothing to learn, so they keep the Slotkin-Harper scaled projection.

THE THING TO WATCH, AND IT IS NOT SMALL

    civicAPI's percent_reporting counts PRECINCTS, not votes, and in Michigan those
    two numbers pull apart badly. Absentee ballots are tabulated by AV counting
    boards, and whether a county reports those boards as precincts varies by county.
    Where it does not, a county can post its entire absentee batch, most of its
    actual vote, while precinct reporting still reads single digits. Divide by that
    and the implied total comes out enormous.

    So the naive N / P is right in expectation only once precinct reporting and vote
    reporting have converged, which happens late. Early it is not just noisy, it is
    biased, and biased in a direction that depends on county administrative practice
    rather than anything about the election.

    Two guardrails handle this:

      CREDIBILITY RAMP. The implied total is blended with the prior, weighted by how
      far along the county is. At 3 percent reporting the prior dominates. By
      FULL_TRUST_PCT the implied total takes over entirely. Set ramp=False for
      straight replacement at any nonzero reporting.

      CLAMP. The implied total is never allowed further than a fixed multiple away
      from the prior in either direction. A county reading 8x its projection is
      telling you about its precinct accounting, not about turnout.

    Both are tunable and both are visible in the diagnostics, so you can watch the
    implied-to-prior ratios during the night and loosen them once precinct reporting
    looks like it is tracking votes.

PROPAGATION TO COUNTIES THAT HAVE NOT REPORTED

    If every reporting county is running 15 percent above projection, the ones that
    have not opened probably will too. The size-weighted median of the implied-to-
    prior ratio across reporting counties is applied to everything still at zero,
    dampened by how much of the state is reporting. Median rather than mean because
    one county with a broken percent_reporting field should not move the state.
"""

import numpy as np
import pandas as pd

from vote_method_split import build_vote_method_table


FULL_TRUST_PCT = 25.0        # precinct reporting at which the implied total fully
                             # replaces the prior
MIN_PCT = 0.0                # counties at or below this keep the prior entirely
CLAMP = (0.40, 2.50)         # implied total may not stray further than this from prior
PROPAGATION_STRENGTH = 1.0   # how strongly the pooled ratio reaches unreported
                             # counties, scaled by statewide reporting share
MIN_COUNTIES_TO_PROPAGATE = 5


def implied_turnout(counted_votes: float, pct_reporting: float) -> float:
    """Total votes a county is heading for, from its count and precinct share."""
    if pct_reporting is None or pct_reporting <= MIN_PCT:
        return np.nan
    return counted_votes / (pct_reporting / 100.0)


def calibrate_turnout(counties: pd.DataFrame,
                      counted: dict,
                      pct_reporting: dict,
                      full_trust_pct: float = FULL_TRUST_PCT,
                      clamp: tuple = CLAMP,
                      ramp: bool = True,
                      propagate: bool = True) -> dict:
    """
    Replace prior county turnout with feed-implied turnout wherever reporting > 0.

    Args:
        counted: {county: total two-candidate votes counted}
        pct_reporting: {county: precinct percent reporting from civicAPI}

    Returns:
        dict with 'turnout' (pd.Series aligned to counties), 'diagnostics' frame,
        'pooled_ratio', and 'statewide_total'
    """
    prior = counties.set_index('county')['turnout'].astype(float)
    updated = prior.copy()
    rows = []

    for county in counties['county']:
        pct = pct_reporting.get(county)
        votes = counted.get(county, 0)

        if pct is None or pct <= MIN_PCT or votes <= 0:
            continue

        raw = implied_turnout(votes, pct)
        if not np.isfinite(raw) or raw <= 0:
            continue

        raw_ratio = raw / prior[county]
        clamped_ratio = float(np.clip(raw_ratio, clamp[0], clamp[1]))

        if ramp:
            weight = float(np.clip(pct / full_trust_pct, 0.0, 1.0))
        else:
            weight = 1.0

        final_ratio = (1 - weight) * 1.0 + weight * clamped_ratio
        updated[county] = prior[county] * final_ratio

        rows.append({
            'county': county,
            'pct_reporting': pct,
            'votes_counted': int(votes),
            'prior_turnout': int(prior[county]),
            'implied_turnout': int(raw),
            'raw_ratio': round(raw_ratio, 3),
            'clamped': raw_ratio != clamped_ratio,
            'weight': round(weight, 3),
            'final_turnout': int(round(updated[county])),
        })

    diagnostics = pd.DataFrame(rows)

    # Propagate the pooled ratio to counties still at zero
    pooled_ratio = 1.0
    if propagate and len(diagnostics) >= MIN_COUNTIES_TO_PROPAGATE:
        ratios = np.clip(diagnostics['raw_ratio'].values, clamp[0], clamp[1])
        sizes = diagnostics['prior_turnout'].values.astype(float)
        order = np.argsort(ratios)
        cumulative = np.cumsum(sizes[order]) / sizes.sum()
        median_idx = order[int(np.searchsorted(cumulative, 0.5))]
        pooled_ratio = float(ratios[median_idx])

        reporting_share = sizes.sum() / prior.sum()
        strength = PROPAGATION_STRENGTH * float(np.clip(reporting_share, 0.0, 1.0))
        applied = 1.0 + (pooled_ratio - 1.0) * strength

        untouched = [c for c in counties['county'] if c not in set(diagnostics['county'])]
        for county in untouched:
            updated[county] = prior[county] * applied

    return {
        'turnout': updated,
        'diagnostics': diagnostics,
        'pooled_ratio': pooled_ratio,
        'statewide_total': float(updated.sum()),
        'prior_statewide_total': float(prior.sum()),
    }


def apply_turnout(counties: pd.DataFrame,
                  method_table: pd.DataFrame,
                  new_turnout: pd.Series) -> tuple:
    """
    Rewrite county turnout and rescale the early/Election Day pools proportionally.

    Margins are untouched. Only the size of each county and each mode pool changes,
    so recalibrating turnout never moves a single projected margin on its own.
    """
    counties = counties.copy()
    counties['turnout'] = (counties['county'].map(new_turnout)
                           .fillna(counties['turnout']).round().astype(int))

    table = method_table.copy().set_index('county')
    share = (table['early_votes'] / table['total_votes']).copy()

    totals = counties.set_index('county')['turnout']
    table['total_votes'] = totals.reindex(table.index).fillna(table['total_votes']).astype(int)
    table['early_votes'] = (table['total_votes'] * share).round().astype(int)
    table['ed_votes'] = table['total_votes'] - table['early_votes']

    table['early_el_sayed'] = (table['early_votes']
                               * (50 + table['early_margin'] / 2) / 100).round().astype(int)
    table['early_stevens'] = table['early_votes'] - table['early_el_sayed']
    table['ed_el_sayed'] = (table['ed_votes']
                            * (50 + table['ed_margin'] / 2) / 100).round().astype(int)
    table['ed_stevens'] = table['ed_votes'] - table['ed_el_sayed']
    table['total_el_sayed'] = table['early_el_sayed'] + table['ed_el_sayed']
    table['total_stevens'] = table['early_stevens'] + table['ed_stevens']

    return counties, table.reset_index()


if __name__ == '__main__':
    from michigan_primary_model import build_michigan_county_data

    counties = build_michigan_county_data()
    table, _ = build_vote_method_table(counties)
    lookup = table.set_index('county')

    # Ten counties reporting, true turnout running 18% above projection
    TRUE_RATIO = 1.18
    rng = np.random.default_rng(4)
    counted, pct = {}, {}
    for county in ['Wayne', 'Oakland', 'Macomb', 'Washtenaw', 'Kent', 'Genesee',
                   'Ingham', 'Kalamazoo', 'Saginaw', 'Livingston']:
        true_total = lookup.loc[county, 'total_votes'] * TRUE_RATIO
        p = float(rng.uniform(8, 60))
        counted[county] = int(true_total * p / 100)
        pct[county] = round(p, 1)

    for ramp in (True, False):
        result = calibrate_turnout(counties, counted, pct, ramp=ramp)
        print("ramp={}  statewide {:,.0f} -> {:,.0f}  pooled ratio {:.3f}".format(
            ramp, result['prior_statewide_total'], result['statewide_total'],
            result['pooled_ratio']))
    print()
    result = calibrate_turnout(counties, counted, pct, ramp=True)
    print(result['diagnostics'].to_string(index=False))
