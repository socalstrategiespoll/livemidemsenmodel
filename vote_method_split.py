"""
County x vote-method estimates for the Michigan Democratic Primary.

Splits each county's blended baseline margin into an early/absentee margin and an
Election Day margin, subject to two statewide constraints:

    1. Turnout-weighted blended margin  = TARGET_OVERALL_MARGIN  (14.53)
    2. Early-vote-weighted early margin = TARGET_EARLY_MARGIN    (8.00)

Constraint 1 holds automatically by construction, since each county's two mode
margins are defined to reconcile to that county's blended baseline. Constraint 2
is what pins down the size of the mode gap.

For a county c with blended margin m_c and early share s_c, define the mode gap
g_c = (Election Day margin) - (early margin). Then:

    early_margin_c = m_c - (1 - s_c) * g_c
    ed_margin_c    = m_c + s_c * g_c

which satisfies s_c * early_margin_c + (1 - s_c) * ed_margin_c = m_c exactly.

The gap is parameterized as g_c = G * scale_c, where G is solved analytically from
constraint 2 and scale_c is a per-county multiplier (default 1.0) you can override
for counties where you expect the mode split to behave differently.

IMPORTANT: the per-county early shares produced by default_early_shares() are
MODELED, not observed. They come from a size-based heuristic calibrated to hit a
target statewide share. Replace them with real county absentee + early in-person
counts from the Michigan Voting Dashboard (mi.gov/votingdashboard) as soon as you
have them. See load_early_shares_from_csv().
"""

import numpy as np
import pandas as pd

from michigan_primary_model import build_michigan_county_data


TARGET_OVERALL_MARGIN = 14.53   # El-Sayed statewide margin, points
TARGET_EARLY_MARGIN = 8.00      # El-Sayed margin among early/absentee votes, points
TARGET_EARLY_SHARE = 0.70       # share of all votes cast before Election Day

EARLY_SHARE_FLOOR = 0.55
EARLY_SHARE_CEILING = 0.78
EARLY_SHARE_SIZE_SLOPE = 0.025  # points of early share per log unit of county turnout


def default_early_shares(counties: pd.DataFrame,
                         target_share: float = TARGET_EARLY_SHARE,
                         slope: float = EARLY_SHARE_SIZE_SLOPE,
                         floor: float = EARLY_SHARE_FLOOR,
                         ceiling: float = EARLY_SHARE_CEILING) -> pd.Series:
    """
    Heuristic per-county early-vote share, driven by county size.

    Larger counties get higher early shares. The intercept is solved so the
    turnout-weighted mean equals target_share after clipping.

    This is a placeholder. Swap in real dashboard numbers when available.
    """
    log_size = np.log(counties['turnout'].values)
    turnout = counties['turnout'].values.astype(float)

    # Solve the intercept iteratively because clipping breaks the closed form.
    intercept = target_share - slope * np.average(log_size, weights=turnout)
    for _ in range(200):
        shares = np.clip(intercept + slope * log_size, floor, ceiling)
        achieved = np.average(shares, weights=turnout)
        if abs(achieved - target_share) < 1e-9:
            break
        intercept += (target_share - achieved)

    return pd.Series(shares, index=counties.index, name='early_share')


def load_early_shares_from_csv(counties: pd.DataFrame, path: str) -> pd.Series:
    """
    Load observed early shares from a CSV with columns: county, early_share.

    Counties missing from the file fall back to the size heuristic.
    """
    observed = pd.read_csv(path)
    lookup = dict(zip(observed['county'].str.strip(), observed['early_share']))

    fallback = default_early_shares(counties)
    shares = counties['county'].map(lookup)
    return shares.fillna(fallback).rename('early_share')


def solve_mode_gap(counties: pd.DataFrame,
                   early_shares: pd.Series,
                   gap_scale: pd.Series = None,
                   target_early_margin: float = TARGET_EARLY_MARGIN) -> float:
    """
    Solve for the base mode gap G such that the early-vote-weighted statewide
    early margin equals target_early_margin.

    Early votes in county c: E_c = T_c * s_c
    early_margin_c = m_c - (1 - s_c) * G * scale_c

    sum(E_c * early_margin_c) / sum(E_c) = target
    => G = [ sum(E_c * m_c) / sum(E_c) - target ] / [ sum(E_c * (1-s_c) * scale_c) / sum(E_c) ]
    """
    turnout = counties['turnout'].values.astype(float)
    margins = counties['margin'].values.astype(float)
    shares = early_shares.values.astype(float)

    if gap_scale is None:
        scale = np.ones(len(counties))
    else:
        scale = gap_scale.values.astype(float)

    early_votes = turnout * shares
    total_early = early_votes.sum()

    blended_early_margin = np.sum(early_votes * margins) / total_early
    denominator = np.sum(early_votes * (1 - shares) * scale) / total_early

    if abs(denominator) < 1e-12:
        raise ValueError("Mode gap is unidentified: denominator collapsed to zero.")

    return (blended_early_margin - target_early_margin) / denominator


def build_vote_method_table(counties: pd.DataFrame = None,
                            early_shares: pd.Series = None,
                            gap_scale: pd.Series = None,
                            target_early_margin: float = TARGET_EARLY_MARGIN,
                            clip_margins: bool = True) -> tuple:
    """
    Build the full county x vote-method estimate table.

    Returns:
        (table, diagnostics)

        table columns:
            county, total_votes, early_share
            early_votes, ed_votes
            blended_margin, early_margin, ed_margin
            early_el_sayed, early_stevens
            ed_el_sayed, ed_stevens
            total_el_sayed, total_stevens

        diagnostics: dict with the solved gap and constraint checks
    """
    if counties is None:
        counties = build_michigan_county_data()
    counties = counties.reset_index(drop=True)

    if early_shares is None:
        early_shares = default_early_shares(counties)
    early_shares = pd.Series(np.asarray(early_shares, dtype=float),
                             index=counties.index)

    if gap_scale is None:
        gap_scale = pd.Series(np.ones(len(counties)), index=counties.index)
    gap_scale = pd.Series(np.asarray(gap_scale, dtype=float), index=counties.index)

    base_gap = solve_mode_gap(counties, early_shares, gap_scale, target_early_margin)
    gaps = base_gap * gap_scale.values

    turnout = counties['turnout'].values.astype(float)
    margins = counties['margin'].values.astype(float)
    shares = early_shares.values

    early_margin = margins - (1 - shares) * gaps
    ed_margin = margins + shares * gaps

    if clip_margins:
        early_margin = np.clip(early_margin, -100.0, 100.0)
        ed_margin = np.clip(ed_margin, -100.0, 100.0)

    early_votes = np.round(turnout * shares).astype(int)
    ed_votes = (np.round(turnout).astype(int) - early_votes)

    early_el_sayed = np.round(early_votes * (50 + early_margin / 2) / 100).astype(int)
    early_stevens = early_votes - early_el_sayed
    ed_el_sayed = np.round(ed_votes * (50 + ed_margin / 2) / 100).astype(int)
    ed_stevens = ed_votes - ed_el_sayed

    table = pd.DataFrame({
        'county': counties['county'].values,
        'total_votes': np.round(turnout).astype(int),
        'early_share': np.round(shares, 4),
        'early_votes': early_votes,
        'ed_votes': ed_votes,
        'blended_margin': np.round(margins, 2),
        'early_margin': np.round(early_margin, 2),
        'ed_margin': np.round(ed_margin, 2),
        'early_el_sayed': early_el_sayed,
        'early_stevens': early_stevens,
        'ed_el_sayed': ed_el_sayed,
        'ed_stevens': ed_stevens,
    })
    table['total_el_sayed'] = table['early_el_sayed'] + table['ed_el_sayed']
    table['total_stevens'] = table['early_stevens'] + table['ed_stevens']

    # Constraint checks against the realized integer vote counts
    tot_early_es = table['early_el_sayed'].sum()
    tot_early_st = table['early_stevens'].sum()
    tot_ed_es = table['ed_el_sayed'].sum()
    tot_ed_st = table['ed_stevens'].sum()
    tot_es = tot_early_es + tot_ed_es
    tot_st = tot_early_st + tot_ed_st

    diagnostics = {
        'base_mode_gap': base_gap,
        'statewide_early_share': table['early_votes'].sum() / table['total_votes'].sum(),
        'statewide_early_margin': (tot_early_es - tot_early_st) / (tot_early_es + tot_early_st) * 100,
        'statewide_ed_margin': (tot_ed_es - tot_ed_st) / (tot_ed_es + tot_ed_st) * 100,
        'statewide_overall_margin': (tot_es - tot_st) / (tot_es + tot_st) * 100,
        'total_early_votes': int(table['early_votes'].sum()),
        'total_ed_votes': int(table['ed_votes'].sum()),
        'el_sayed_total': int(tot_es),
        'stevens_total': int(tot_st),
        'counties_flipping_mode': int(
            ((table['early_margin'] < 0) & (table['ed_margin'] > 0)).sum()
            + ((table['early_margin'] > 0) & (table['ed_margin'] < 0)).sum()
        ),
    }

    return table, diagnostics


if __name__ == '__main__':
    counties = build_michigan_county_data()
    table, diag = build_vote_method_table(counties)

    print("Solved base mode gap: {:.2f} points (ED margin minus early margin)".format(
        diag['base_mode_gap']))
    print("Statewide early share: {:.1%}".format(diag['statewide_early_share']))
    print("Statewide early margin: El-Sayed {:+.2f}".format(diag['statewide_early_margin']))
    print("Statewide ED margin:    El-Sayed {:+.2f}".format(diag['statewide_ed_margin']))
    print("Statewide overall:      El-Sayed {:+.2f}".format(diag['statewide_overall_margin']))
    print("Totals: El-Sayed {:,}  Stevens {:,}".format(
        diag['el_sayed_total'], diag['stevens_total']))
    print("Counties where the two modes disagree on the winner: {}".format(
        diag['counties_flipping_mode']))
    print()

    show = ['county', 'total_votes', 'early_share', 'early_votes', 'ed_votes',
            'blended_margin', 'early_margin', 'ed_margin']
    print(table.sort_values('total_votes', ascending=False)[show].head(20).to_string(index=False))

    table.to_csv('mi_county_vote_method_estimates.csv', index=False)
    print("\nWrote mi_county_vote_method_estimates.csv")
