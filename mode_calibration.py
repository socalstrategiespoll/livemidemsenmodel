"""
Live vote-mode calibration.

The mode gap (21.04 points, solved in vote_method_split.py) is an assumption derived
from your +8 early-vote expectation. Everything downstream inherits it. If the real
gap is 12 or 30, every county's mode inference is wrong in the same direction all
night, and no amount of Bayesian machinery around a wrong constant will save it.

This module stops treating the gap as fixed. It learns it from counties that have
finished counting and feeds the answer back.

WHY COMPLETED COUNTIES IDENTIFY THE GAP AND PARTIAL ONES DO NOT

    A single snapshot of a partial county gives one number, its margin so far, and
    two unknowns, the mode composition and the county's true shift. Not identified.
    This is the wall the margin-based inference kept hitting.

    A completed county gives something qualitatively different: a SEQUENCE of batches
    whose sizes must sum to a known total, of which a known number were early votes.
    That constraint is what breaks the degeneracy. If Macomb posted 70,000 votes at
    +7 and then 38,000 more at +26, and Macomb only has 76,742 early votes, then the
    second batch is overwhelmingly Election Day and its margin is close to a direct
    read of the Election Day number. No inference required, just arithmetic against a
    pool that cannot be exceeded.

    So the model leans on counties in proportion to how much their batch structure
    actually pins the gap down, which is a different thing from how much of their
    vote is in. A county that reported all at once in one dump tells you almost
    nothing about the gap no matter how complete it is.

WHAT GETS ESTIMATED

    Per county, two numbers relative to the model's current expectation:

        delta  - the county's shift, common to both modes
        kappa  - a multiplier on the mode gap. 1.0 means the assumed gap was right,
                 0.5 means the modes are half as far apart as assumed, 1.5 means
                 they are further apart.

    Kappa is then pooled across counties, weighted by identification strength, into a
    global posterior. The pooled kappa rebuilds the vote-method table, which updates
    early and Election Day margins everywhere, which updates every downstream
    inference. Delta stays local and is left to the hierarchical model, which already
    handles shifts properly.

THE ORDERING ASSUMPTION

    Allocating batches to modes uses a sequential assumption: within a county, early
    votes clear before Election Day votes, softened by a mixing parameter. This is
    the weak link, and it is weak in a specific way. It holds well in counties that
    post a large absentee batch first, which is the Michigan norm, and fails in
    counties that report precinct-by-precinct with absentee folded in throughout.
    SEQUENTIAL_MIXING controls how hard the assumption bites. Set it to 0 for pure
    sequential, higher for counties that blend.
"""

import numpy as np
import pandas as pd

from vote_method_split import build_vote_method_table


SEQUENTIAL_MIXING = 0.15     # 0 = early strictly clears first; higher blends modes
                             # across the counting sequence
MIN_COMPLETENESS = 0.85      # only counties at least this counted inform the gap
MIN_IDENTIFICATION = 0.01    # minimum theta spread across batches to bother using
KAPPA_PRIOR_MEAN = 1.0
KAPPA_PRIOR_SD = 0.45        # how wrong the assumed gap could plausibly be
KAPPA_BOUNDS = (0.0, 2.5)
ORDERING_ASSUMPTION_SD = 0.10  # irreducible uncertainty from the sequential ordering
                               # assumption; does not shrink with more counties


class VoteFeed:
    """
    Tracks cumulative county results over time and derives the batches between them.

    Feed it whatever the wire gives you. It stores snapshots and differences them.
    """

    def __init__(self):
        self.snapshots = {}       # county -> list of (el_sayed, stevens)
        self.pct_reporting = {}   # county -> latest precinct percent from the feed

    def update(self, county: str, el_sayed: int, stevens: int,
               pct_reporting: float = None) -> None:
        """Record a cumulative result. Non-monotonic updates are ignored."""
        if pct_reporting is not None:
            self.pct_reporting[county] = float(pct_reporting)
        history = self.snapshots.setdefault(county, [])
        if history:
            last_es, last_st = history[-1]
            if el_sayed < last_es or stevens < last_st:
                return
            if el_sayed == last_es and stevens == last_st:
                return
        history.append((int(el_sayed), int(stevens)))

    def batches(self, county: str) -> list:
        """
        Return [(votes, margin, cumulative_before), ...] for each incremental batch.
        """
        history = self.snapshots.get(county, [])
        out = []
        prev_es = prev_st = 0
        for es, st in history:
            n = (es - prev_es) + (st - prev_st)
            if n > 0:
                margin = ((es - prev_es) - (st - prev_st)) / n * 100.0
                out.append((n, margin, prev_es + prev_st))
            prev_es, prev_st = es, st
        return out

    def counted(self) -> dict:
        """{county: total two-candidate votes counted so far}"""
        return {c: sum(self.latest(c)) for c in self.snapshots}

    def latest(self, county: str) -> tuple:
        history = self.snapshots.get(county, [])
        return history[-1] if history else (0, 0)

    def as_reported_dict(self, method_table: pd.DataFrame) -> dict:
        """Build the `reported` dict the hierarchical model consumes."""
        table = method_table.set_index('county')
        out = {}
        for county in self.snapshots:
            if county not in table.index:
                continue
            es, st = self.latest(county)
            total = float(table.loc[county, 'total_votes'])
            out[county] = {
                'el_sayed': es,
                'stevens': st,
                'pct_in': min((es + st) / max(total, 1.0), 1.0),
            }
        return out


def allocate_batches(batches: list, early_pool: float,
                     mixing: float = SEQUENTIAL_MIXING) -> np.ndarray:
    """
    Assign each batch an early-vote share (theta) using the sequential assumption.

    Votes counted before the early pool is exhausted are early; votes after are
    Election Day. `mixing` bleeds the boundary so the transition is not a hard step.

    Returns theta per batch.
    """
    thetas = []
    for n, _margin, cumulative_before in batches:
        start, end = cumulative_before, cumulative_before + n
        # Fraction of this batch falling below the early-pool cutoff
        overlap = max(0.0, min(end, early_pool) - start)
        pure = overlap / n if n > 0 else 0.0
        # Blend toward the county's overall early share
        thetas.append((1 - mixing) * pure + mixing * min(1.0, early_pool / max(end, 1.0)))
    return np.clip(np.array(thetas), 0.0, 1.0)


def estimate_county_kappa(batches: list,
                          early_pool: float,
                          ed_pool: float,
                          expected_early_margin: float,
                          expected_ed_margin: float,
                          mixing: float = SEQUENTIAL_MIXING) -> dict:
    """
    Estimate the gap multiplier kappa and the county shift delta from batch structure.

    Model for batch i:
        margin_i = [theta_i * ME + (1 - theta_i) * MD] + delta + noise

    with ME and MD parameterized around the model's expectation via kappa:
        ME(kappa) = blended - (1 - s) * kappa * G
        MD(kappa) = blended + s * kappa * G

    where G is the expected gap and s the county's overall early share. This makes
    kappa and delta separately identified: delta shifts all batches together, kappa
    spreads them apart.

    Returns dict with kappa, delta, identification (theta spread), and se.
    """
    if len(batches) < 2:
        return {'kappa': np.nan, 'delta': np.nan, 'identification': 0.0,
                'se': np.inf, 'n_batches': len(batches)}

    total_pool = early_pool + ed_pool
    s = early_pool / max(total_pool, 1e-9)
    gap = expected_ed_margin - expected_early_margin
    blended = s * expected_early_margin + (1 - s) * expected_ed_margin

    if abs(gap) < 1e-6:
        return {'kappa': np.nan, 'delta': np.nan, 'identification': 0.0,
                'se': np.inf, 'n_batches': len(batches)}

    theta = allocate_batches(batches, early_pool, mixing)
    sizes = np.array([b[0] for b in batches], dtype=float)
    margins = np.array([b[1] for b in batches], dtype=float)

    # Identification strength: size-weighted spread of theta across batches. If every
    # batch has the same composition, kappa is unidentified no matter how many votes
    # came in.
    theta_bar = np.average(theta, weights=sizes)
    identification = float(np.sqrt(np.average((theta - theta_bar) ** 2, weights=sizes)))

    if identification < MIN_IDENTIFICATION:
        return {'kappa': np.nan, 'delta': np.nan, 'identification': identification,
                'se': np.inf, 'n_batches': len(batches)}

    # Weighted least squares: margin_i - blended = delta + kappa * gap * (s - theta_i)
    y = margins - blended
    x = gap * (s - theta)
    w = sizes / sizes.sum()

    # Ridge toward the kappa prior keeps weakly identified counties from exploding
    x_bar = np.sum(w * x)
    y_bar = np.sum(w * y)
    sxx = np.sum(w * (x - x_bar) ** 2)
    sxy = np.sum(w * (x - x_bar) * (y - y_bar))

    ridge = 1.0 / (KAPPA_PRIOR_SD ** 2)
    kappa = (sxy + ridge * KAPPA_PRIOR_MEAN * 0.0 + ridge * KAPPA_PRIOR_MEAN) / (sxx + ridge)
    kappa = float(np.clip(kappa, *KAPPA_BOUNDS))
    delta = float(y_bar - kappa * x_bar)

    residuals = y - (delta + kappa * x)
    resid_var = float(np.sum(w * residuals ** 2))
    se = float(np.sqrt(max(resid_var, 1e-6) / max(sxx + ridge, 1e-9)))

    return {'kappa': kappa, 'delta': delta, 'identification': identification,
            'se': se, 'n_batches': len(batches)}


def calibrate_mode_gap(feed: VoteFeed,
                       method_table: pd.DataFrame,
                       min_completeness: float = MIN_COMPLETENESS,
                       mixing: float = SEQUENTIAL_MIXING) -> dict:
    """
    Pool kappa across sufficiently complete counties into a global posterior.

    Returns dict with kappa_mean, kappa_sd, and a per-county diagnostics frame.
    """
    table = method_table.set_index('county')
    rows = []

    for county in feed.snapshots:
        if county not in table.index:
            continue
        row = table.loc[county]
        es, st = feed.latest(county)
        counted = es + st
        completeness = counted / max(float(row['total_votes']), 1.0)
        if completeness < min_completeness:
            continue

        est = estimate_county_kappa(
            feed.batches(county),
            float(row['early_votes']), float(row['ed_votes']),
            float(row['early_margin']), float(row['ed_margin']),
            mixing=mixing)

        if not np.isfinite(est['kappa']) or not np.isfinite(est['se']):
            continue

        rows.append({
            'county': county,
            'completeness': round(completeness, 3),
            'n_batches': est['n_batches'],
            'identification': round(est['identification'], 3),
            'kappa': round(est['kappa'], 3),
            'delta': round(est['delta'], 2),
            'se': round(est['se'], 3),
        })

    diagnostics = pd.DataFrame(rows)

    # Random-effects pooling. Pure inverse-variance weighting understates the
    # uncertainty badly here, because the within-county standard errors only capture
    # batch-level noise and miss the thing that actually varies: whether the
    # sequential ordering assumption holds in that county. DerSimonian-Laird
    # heterogeneity picks that up from the spread between counties, which is the only
    # place it is visible.
    prior_precision = 1.0 / KAPPA_PRIOR_SD ** 2
    tau = 0.0

    if len(diagnostics) > 1:
        k_hat = diagnostics['kappa'].values
        se_hat = np.maximum(diagnostics['se'].values, 1e-3)
        w_fe = 1.0 / se_hat ** 2
        pooled_fe = float(np.sum(w_fe * k_hat) / np.sum(w_fe))
        q = float(np.sum(w_fe * (k_hat - pooled_fe) ** 2))
        k_n = len(k_hat)
        c = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
        if q > (k_n - 1) and c > 0:
            tau = float(np.sqrt((q - (k_n - 1)) / c))

    num = prior_precision * KAPPA_PRIOR_MEAN
    den = prior_precision

    if len(diagnostics) > 0:
        precision = 1.0 / (np.maximum(diagnostics['se'].values ** 2, 1e-6) + tau ** 2)
        num += float(np.sum(precision * diagnostics['kappa'].values))
        den += float(np.sum(precision))

    kappa_mean = num / den
    # Floor the reported sd. The ordering assumption is a modelling choice, not a
    # measured quantity, and it does not get more certain with more counties.
    kappa_sd = float(np.sqrt(1.0 / den + ORDERING_ASSUMPTION_SD ** 2))

    return {
        'kappa_mean': float(np.clip(kappa_mean, *KAPPA_BOUNDS)),
        'kappa_sd': kappa_sd,
        'tau': tau,
        'n_counties_used': len(diagnostics),
        'diagnostics': diagnostics,
    }


def rebuild_method_table(counties: pd.DataFrame,
                         kappa: float,
                         base_table: pd.DataFrame = None,
                         early_shares: pd.Series = None) -> pd.DataFrame:
    """
    Rebuild the vote-method table with the calibrated gap multiplier applied.

    Each county's blended margin is preserved exactly. Only the spread between the
    two modes changes, so calibrating the gap never silently moves your topline.
    """
    if base_table is None:
        base_table, _ = build_vote_method_table(counties, early_shares=early_shares)

    table = base_table.copy()
    s = table['early_votes'] / table['total_votes']
    blended = table['blended_margin']
    base_gap = table['ed_margin'] - table['early_margin']
    new_gap = base_gap * kappa

    table['early_margin'] = (blended - (1 - s) * new_gap).round(2)
    table['ed_margin'] = (blended + s * new_gap).round(2)

    table['early_el_sayed'] = (table['early_votes']
                               * (50 + table['early_margin'] / 2) / 100).round().astype(int)
    table['early_stevens'] = table['early_votes'] - table['early_el_sayed']
    table['ed_el_sayed'] = (table['ed_votes']
                            * (50 + table['ed_margin'] / 2) / 100).round().astype(int)
    table['ed_stevens'] = table['ed_votes'] - table['ed_el_sayed']
    table['total_el_sayed'] = table['early_el_sayed'] + table['ed_el_sayed']
    table['total_stevens'] = table['early_stevens'] + table['ed_stevens']

    return table


if __name__ == '__main__':
    from michigan_primary_model import build_michigan_county_data

    counties = build_michigan_county_data()
    base_table, _ = build_vote_method_table(counties)

    print("Assumed gap: {:.2f} points".format(
        float(base_table.loc[base_table.county == 'Macomb', 'ed_margin'].iloc[0]
              - base_table.loc[base_table.county == 'Macomb', 'early_margin'].iloc[0])))
    print()

    # Simulate a night where the TRUE gap is only 12 points, not 21.
    TRUE_KAPPA = 12.0 / 21.04
    TRUE_SHIFT = 2.0
    rng = np.random.default_rng(11)

    feed = VoteFeed()
    big = ['Wayne', 'Oakland', 'Macomb', 'Washtenaw', 'Kent', 'Genesee',
           'Ingham', 'Kalamazoo', 'Saginaw', 'Livingston']

    for county in big:
        row = base_table[base_table.county == county].iloc[0]
        s = row.early_votes / row.total_votes
        true_early = row.blended_margin - (1 - s) * (row.ed_margin - row.early_margin) * TRUE_KAPPA
        true_ed = row.blended_margin + s * (row.ed_margin - row.early_margin) * TRUE_KAPPA

        # Three batches: most of the early pool, the rest of early plus some ED, then
        # the remaining Election Day vote.
        cuts = [int(row.early_votes * 0.85), int(row.early_votes * 1.05), int(row.total_votes)]
        es = st = 0
        prev = 0
        for cut in cuts:
            n = cut - prev
            theta = max(0.0, min(row.early_votes, cut) - min(row.early_votes, prev)) / n
            m = theta * true_early + (1 - theta) * true_ed + TRUE_SHIFT + rng.normal(0, 1.0)
            add_es = int(n * (50 + m / 2) / 100)
            es += add_es
            st += n - add_es
            feed.update(county, es, st)
            prev = cut

    result = calibrate_mode_gap(feed, base_table)
    print("True kappa: {:.3f}  (true gap {:.1f} points)".format(TRUE_KAPPA, 12.0))
    print("Estimated:  {:.3f} +/- {:.3f}  from {} counties".format(
        result['kappa_mean'], result['kappa_sd'], result['n_counties_used']))
    print("Implied gap: {:.1f} points".format(result['kappa_mean'] * 21.04))
    print()
    print(result['diagnostics'].to_string(index=False))
