"""
Hierarchical correlated-shift engine for the Michigan Democratic Primary model.

This replaces the flat DerSimonian-Laird pooling in michigan_primary_model.py with a
Gaussian-process style structure that does two things the flat version could not:

  1. DAMPENS SINGLE-COUNTY LEVERAGE. A fully-counted Wayne can no longer drag the
     statewide projection on its own. Every county carries a baseline-error nugget:
     variance that represents "our prior for this county was simply wrong" and that
     does NOT shrink as more of its vote comes in. A large residual in one county is
     therefore split between a real statewide signal and a county-specific miss,
     rather than being read entirely as signal. A hard leverage cap sits on top of
     that as a guard.

  2. PROPAGATES INFORMATION BY REGION AND DEMOGRAPHIC SIMILARITY, not just a single
     universal number. The shift vector across counties is drawn from a multivariate
     normal whose covariance decomposes into four independent components:

         Sigma = s_state^2 * J
               + s_region^2 * SameRegion
               + s_demo^2  * K(demographic distance)
               + s_local^2 * I

     A surprise in Kent moves West Michigan more than it moves the Thumb, and moves
     demographically similar counties anywhere in the state more than dissimilar
     neighbors. The universal shift is still in there as the first term.

Conditioning reported counties on this prior gives the posterior for the unreported
ones in closed form. Monte Carlo then draws correlated shift vectors from that
posterior, so simulation-to-simulation the whole state moves together the way real
polling error does. The old model drew 83 independent errors that cancelled, which
is why its pre-election interval collapsed to a tenth of a point.

NOTE ON DEMOGRAPHICS: the default covariate matrix uses county size (urbanicity
proxy) and baseline margin (coalition proxy). Baseline margin as a covariate is
mildly circular, since it makes counties you expect to behave alike err alike. That
is defensible and standard, but real covariates are better. Pass your own via the
`covariates` argument: Black share, college share, median age, whatever you have in
the voter file.
"""

import numpy as np
import pandas as pd

from michigan_primary_model import build_michigan_county_data
from vote_method_split import build_vote_method_table
from vote_mode_inference import observed_shifts_with_mode_inference
from mode_calibration import VoteFeed, calibrate_mode_gap, rebuild_method_table
from turnout_calibration import calibrate_turnout, apply_turnout


# ---------------------------------------------------------------------------
# Covariance component scales, in margin points of standard deviation.
# These are the knobs that set how much the model thinks can go wrong, and where.
# ---------------------------------------------------------------------------
SIGMA_STATE = 6.0      # universal shift: every county moves together
SIGMA_REGION = 4.0     # regional shift: counties in the same region move together
SIGMA_DEMO = 3.5       # demographic shift: similar counties move together
SIGMA_LOCAL = 3.0      # idiosyncratic county noise
DEMO_LENGTHSCALE = 1.0 # correlation decay in standardized covariate space

# Leverage dampening
BASELINE_NUGGET = 4.0      # points of irreducible per-county baseline error
MAX_LEVERAGE = 0.15        # a single county coming in X points off baseline may move
                           # the statewide projection by at most 0.15 * X points

# Joint mode/shift fit
MODE_SHIFT_PRIOR_SD = 8.0  # starting belief about county shift sd, before any
                           # counties report
MAX_JOINT_PASSES = 15
MODE_ERROR_CORRELATION = 0.75  # how much vote-mode misreads are shared across
                               # counties rather than independent
JOINT_TOLERANCE = 0.01     # points; stop when county shift estimates settle


REGION_MAP = {
    'Wayne': 'Wayne',

    'Oakland': 'Oakland_Macomb', 'Macomb': 'Oakland_Macomb',

    'Kent': 'West', 'Ottawa': 'West', 'Allegan': 'West', 'Kalamazoo': 'West',
    'Muskegon': 'West', 'Newaygo': 'West', 'Oceana': 'West', 'Mecosta': 'West',
    'Barry': 'West', 'Van Buren': 'West', 'Branch': 'West', 'St. Joseph': 'West',
    'Berrien': 'West', 'Manistee': 'West', 'Benzie': 'West', 'Mason': 'West',
    'Lake': 'West', 'Calhoun': 'West',

    'Washtenaw': 'Southeast', 'Livingston': 'Southeast', 'Jackson': 'Southeast',
    'Lenawee': 'Southeast', 'Monroe': 'Southeast', 'Eaton': 'Southeast',
    'Cass': 'Southeast', 'Hillsdale': 'Southeast',

    'Ingham': 'Central', 'Midland': 'Central', 'Isabella': 'Central',
    'Gratiot': 'Central', 'Montcalm': 'Central', 'Ionia': 'Central',
    'Clinton': 'Central', 'Shiawassee': 'Central',

    'Genesee': 'Saginaw_Thumb', 'Saginaw': 'Saginaw_Thumb', 'Lapeer': 'Saginaw_Thumb',
    'Tuscola': 'Saginaw_Thumb', 'Huron': 'Saginaw_Thumb', 'Sanilac': 'Saginaw_Thumb',
    'St. Clair': 'Saginaw_Thumb', 'Bay': 'Saginaw_Thumb',

    'Grand Traverse': 'North_UP', 'Wexford': 'North_UP', 'Charlevoix': 'North_UP',
    'Emmet': 'North_UP', 'Antrim': 'North_UP', 'Cheboygan': 'North_UP',
    'Kalkaska': 'North_UP', 'Missaukee': 'North_UP', 'Otsego': 'North_UP',
    'Leelanau': 'North_UP',
    'Alcona': 'North_UP', 'Arenac': 'North_UP', 'Iosco': 'North_UP',
    'Ogemaw': 'North_UP', 'Oscoda': 'North_UP', 'Alpena': 'North_UP',
    'Presque Isle': 'North_UP', 'Montmorency': 'North_UP', 'Clare': 'North_UP',
    'Gladwin': 'North_UP', 'Roscommon': 'North_UP', 'Osceola': 'North_UP',
    'Crawford': 'North_UP',
    'Alger': 'North_UP', 'Baraga': 'North_UP', 'Chippewa': 'North_UP',
    'Delta': 'North_UP', 'Dickinson': 'North_UP', 'Gogebic': 'North_UP',
    'Houghton': 'North_UP', 'Iron': 'North_UP', 'Keweenaw': 'North_UP',
    'Luce': 'North_UP', 'Mackinac': 'North_UP', 'Marquette': 'North_UP',
    'Menominee': 'North_UP', 'Ontonagon': 'North_UP', 'Schoolcraft': 'North_UP',
}


def assign_regions(counties: pd.DataFrame) -> pd.Series:
    """Map each county to its region. Unmapped counties fall into 'Other'."""
    regions = counties['county'].map(REGION_MAP)
    if regions.isna().any():
        missing = counties.loc[regions.isna(), 'county'].tolist()
        raise ValueError("Counties missing a region assignment: {}".format(missing))
    return regions.rename('region')


def default_covariates(counties: pd.DataFrame) -> np.ndarray:
    """
    Standardized covariate matrix used for demographic-similarity correlation.

    Columns:
        0. log county turnout  (urbanicity proxy)
        1. baseline margin     (coalition proxy)

    Replace with real demographics when you have them.
    """
    size = np.log(counties['turnout'].values.astype(float))
    coalition = counties['margin'].values.astype(float)

    X = np.column_stack([size, coalition])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X


def build_covariance(counties: pd.DataFrame,
                     regions: pd.Series = None,
                     covariates: np.ndarray = None,
                     sigma_state: float = SIGMA_STATE,
                     sigma_region: float = SIGMA_REGION,
                     sigma_demo: float = SIGMA_DEMO,
                     sigma_local: float = SIGMA_LOCAL,
                     lengthscale: float = DEMO_LENGTHSCALE) -> np.ndarray:
    """
    Build the prior covariance of the county shift vector, in squared margin points.

    Sigma = s_state^2 * J + s_region^2 * SameRegion + s_demo^2 * K + s_local^2 * I
    """
    n = len(counties)

    if regions is None:
        regions = assign_regions(counties)
    if covariates is None:
        covariates = default_covariates(counties)

    # Universal component: all-ones matrix
    universal = np.ones((n, n))

    # Regional component: 1 where two counties share a region
    codes = pd.Categorical(regions).codes
    same_region = (codes[:, None] == codes[None, :]).astype(float)

    # Demographic component: squared-exponential kernel on standardized covariates
    diff = covariates[:, None, :] - covariates[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    demo = np.exp(-dist_sq / (2.0 * lengthscale ** 2))

    sigma = (sigma_state ** 2 * universal
             + sigma_region ** 2 * same_region
             + sigma_demo ** 2 * demo
             + sigma_local ** 2 * np.eye(n))

    return sigma


def condition_on_reported(sigma: np.ndarray,
                          reported_idx: np.ndarray,
                          reported_shifts: np.ndarray,
                          sampling_se: np.ndarray,
                          turnout_weights: np.ndarray,
                          mode_se: np.ndarray = None,
                          mode_correlation: float = None,
                          baseline_nugget: float = BASELINE_NUGGET,
                          max_leverage: float = MAX_LEVERAGE) -> tuple:
    """
    Condition the prior on reported counties, with leverage dampening.

    Observation noise for county j is sampling_se_j^2 + baseline_nugget^2. The nugget
    is the piece that dampens leverage: it does not shrink as county j finishes
    counting, so a fully-reported county can never be treated as a noiseless read on
    the statewide shift.

    Leverage for county j is the sensitivity of the turnout-weighted statewide
    posterior mean shift to one point of movement in county j's residual. A leverage
    of 0.15 means a county that comes in 10 points off baseline moves the statewide
    projection by 1.5 points. Any county exceeding max_leverage has its observation
    variance inflated and the system is re-solved until the cap holds.

    Returns:
        (posterior_mean, posterior_cov, leverage)
    """
    n = sigma.shape[0]

    if len(reported_idx) == 0:
        return np.zeros(n), sigma.copy(), np.zeros(0)

    w = turnout_weights / turnout_weights.sum()

    noise = sampling_se ** 2 + baseline_nugget ** 2

    # Correlated mode-error block. Vote-mode misreads are shared across counties,
    # because reporting conventions are shared. Modelling them as independent noise
    # lets six counties' worth of the same mistake average down to nothing, which is
    # precisely the failure this block prevents.
    if mode_correlation is None:
        mode_correlation = MODE_ERROR_CORRELATION
    if mode_se is None:
        mode_se = np.zeros(len(reported_idx))
    mode_block = (mode_correlation * np.outer(mode_se, mode_se)
                  + (1 - mode_correlation) * np.diag(mode_se ** 2))

    weights = np.ones(len(reported_idx))

    sigma_rr = sigma[np.ix_(reported_idx, reported_idx)]
    sigma_ar = sigma[:, reported_idx]

    influence = None
    leverage = None

    for _ in range(200):
        inflated = noise / weights
        solve_matrix = sigma_rr + np.diag(inflated) + mode_block
        influence = np.linalg.solve(solve_matrix, sigma_ar.T).T  # n x k

        # Sensitivity of the statewide (turnout-weighted) mean shift to each county
        leverage = np.abs(w @ influence)

        worst = leverage.max()
        if worst <= max_leverage * 1.001:
            break

        over = leverage > max_leverage
        # Damped multiplicative update on the precision weights
        ratio = max_leverage / np.maximum(leverage[over], 1e-9)
        weights[over] *= ratio ** 0.5

    posterior_mean = influence @ reported_shifts
    posterior_cov = sigma - influence @ sigma_ar.T
    posterior_cov = (posterior_cov + posterior_cov.T) / 2.0

    return posterior_mean, posterior_cov, leverage


def simulate(counties: pd.DataFrame = None,
             method_table: pd.DataFrame = None,
             reported: dict = None,
             feed: 'VoteFeed' = None,
             n_sims: int = 20000,
             seed: int = None,
             **cov_kwargs) -> dict:
    """
    Run the full hierarchical simulation.

    Args:
        reported: {county: {'el_sayed': int, 'stevens': int, 'mode': str,
                            'pct_in': float}}
                  pct_in is the share of that county's total vote already counted.
                  Counted votes are held fixed; only the remainder is projected.

    Returns:
        dict of projection results
    """
    rng = np.random.default_rng(seed)

    if counties is None:
        counties = build_michigan_county_data()
    counties = counties.reset_index(drop=True)

    if method_table is None:
        method_table, _ = build_vote_method_table(counties)

    # Recalibrate the mode gap from completed counties before anything else. Counties
    # that have finished counting give a direct read on how far apart the two modes
    # actually are, and that read replaces the assumed gap for every county still
    # outstanding. Blended margins are preserved, so this never moves the topline on
    # its own.
    # Turnout first. Every pool size downstream depends on it, including the early
    # vote pools that the mode inference uses as hard volume bounds, so a stale
    # turnout number quietly corrupts the mode work rather than just the totals.
    turnout_calibration = None
    if feed is not None and getattr(feed, 'pct_reporting', None):
        turnout_calibration = calibrate_turnout(
            counties, feed.counted(), feed.pct_reporting)
        if len(turnout_calibration['diagnostics']) > 0:
            counties, method_table = apply_turnout(
                counties, method_table, turnout_calibration['turnout'])
            counties = counties.reset_index(drop=True)

    calibration = None
    if feed is not None:
        calibration = calibrate_mode_gap(feed, method_table)
        if calibration['n_counties_used'] > 0:
            method_table = rebuild_method_table(
                counties, calibration['kappa_mean'], base_table=method_table)
        if reported is None:
            reported = feed.as_reported_dict(method_table)

    reported = reported or {}

    sigma = build_covariance(counties, **cov_kwargs)
    turnout_arr = counties['turnout'].values.astype(float)
    n_counties = len(counties)

    # Joint fit of vote mode and county shifts.
    #
    # Run separately, these two estimates fight each other: a county running hot
    # gets read as Election Day vote arriving, so theta absorbs a shift that is
    # actually real, and the projection comes in biased against whoever is beating
    # baseline. Alternating between them fixes that. The shift is identified across
    # many counties at once, so once the state and regional terms have absorbed it,
    # theta is left to be pinned by volume bounds where it belongs.
    shift_mean = np.zeros(n_counties)
    shift_sd = np.full(n_counties, MODE_SHIFT_PRIOR_SD)
    post_mean = np.zeros(n_counties)
    post_cov = sigma.copy()
    leverage = np.zeros(0)
    mode_diag = pd.DataFrame()
    n_passes = 0

    for _ in range(MAX_JOINT_PASSES):
        n_passes += 1
        idx, shifts, ses, mode_ses, mode_diag = observed_shifts_with_mode_inference(
            counties, method_table, reported,
            shift_mean=shift_mean, shift_sd=shift_sd)
        post_mean, post_cov, leverage = condition_on_reported(
            sigma, idx, shifts, ses, turnout_arr, mode_se=mode_ses)

        if len(idx) == 0:
            break
        delta = np.max(np.abs(post_mean - shift_mean))
        shift_mean = post_mean.copy()
        shift_sd = np.sqrt(np.maximum(np.diag(post_cov), 1e-6))
        if delta < JOINT_TOLERANCE:
            break

    # Correlated draws of the shift vector
    jitter = 1e-8 * np.trace(post_cov) / post_cov.shape[0]
    chol = np.linalg.cholesky(post_cov + jitter * np.eye(post_cov.shape[0]))
    draws = post_mean[None, :] + rng.standard_normal((n_sims, len(counties))) @ chol.T

    turnout = counties['turnout'].values.astype(float)
    baseline = counties['margin'].values.astype(float)

    # Split each county into counted and uncounted portions
    pct_in = np.zeros(len(counties))
    counted_el_sayed = np.zeros(len(counties))
    counted_stevens = np.zeros(len(counties))
    idx_of = {c: i for i, c in enumerate(counties['county'])}

    for county, rec in reported.items():
        if county not in idx_of:
            continue
        i = idx_of[county]
        counted_el_sayed[i] = rec['el_sayed']
        counted_stevens[i] = rec['stevens']
        counted = rec['el_sayed'] + rec['stevens']
        pct_in[i] = rec.get('pct_in', min(counted / turnout[i], 1.0))

    remaining = np.maximum(turnout * (1 - pct_in), 0.0)

    projected_margin = np.clip(baseline[None, :] + draws, -100.0, 100.0)
    share_el_sayed = (50.0 + projected_margin / 2.0) / 100.0

    rem_el_sayed = remaining[None, :] * share_el_sayed
    rem_stevens = remaining[None, :] * (1.0 - share_el_sayed)

    total_el_sayed = counted_el_sayed.sum() + rem_el_sayed.sum(axis=1)
    total_stevens = counted_stevens.sum() + rem_stevens.sum(axis=1)

    margins = (total_el_sayed - total_stevens) / (total_el_sayed + total_stevens) * 100.0
    win_prob = float(np.mean(total_el_sayed > total_stevens))

    leverage_report = None
    if len(idx) > 0:
        leverage_report = (pd.DataFrame({
            'county': counties['county'].values[idx],
            'observed_shift': np.round(shifts, 2),
            'leverage': np.round(leverage, 4),
        }).sort_values('leverage', ascending=False).reset_index(drop=True))

    return {
        'el_sayed_win_probability': win_prob,
        'median_margin': float(np.median(margins)),
        'margin_ci_lower': float(np.percentile(margins, 5)),
        'margin_ci_upper': float(np.percentile(margins, 95)),
        'margin_ci_50_lower': float(np.percentile(margins, 25)),
        'margin_ci_50_upper': float(np.percentile(margins, 75)),
        'el_sayed_median_votes': float(np.median(total_el_sayed)),
        'stevens_median_votes': float(np.median(total_stevens)),
        'margins': margins,
        'implied_state_shift': float(np.average(post_mean, weights=turnout)),
        'county_posterior_mean_shift': pd.Series(post_mean, index=counties['county']),
        'region_posterior_shift': (pd.Series(post_mean, index=counties['county'])
                                   .groupby(assign_regions(counties).values).mean()),
        'leverage': leverage_report,
        'mode_inference': mode_diag,
        'n_reported': len(idx),
        'joint_passes': n_passes,
        'calibration': calibration,
        'turnout_calibration': turnout_calibration,
        'projected_turnout': int(counties['turnout'].sum()),
        'method_table': method_table,
        'pct_counted': float((counted_el_sayed.sum() + counted_stevens.sum())
                             / turnout.sum()),
    }


if __name__ == '__main__':
    counties = build_michigan_county_data()
    method_table, _ = build_vote_method_table(counties)

    print("=" * 68)
    print("PRE-ELECTION (no counties reported)")
    print("=" * 68)
    pre = simulate(counties, method_table, reported={}, seed=1)
    print("El-Sayed win probability: {:.1%}".format(pre['el_sayed_win_probability']))
    print("Median margin: El-Sayed {:+.1f}".format(pre['median_margin']))
    print("50% interval: {:+.1f} to {:+.1f}".format(
        pre['margin_ci_50_lower'], pre['margin_ci_50_upper']))
    print("90% interval: {:+.1f} to {:+.1f}".format(
        pre['margin_ci_lower'], pre['margin_ci_upper']))

    print()
    print("=" * 68)
    print("LEVERAGE TEST: Wayne fully counted, 10 points off baseline")
    print("=" * 68)
    wayne_total = int(counties.loc[counties.county == 'Wayne', 'turnout'].iloc[0])
    # Wayne blended baseline +8.0; run it in at +18.0
    wayne_es = int(wayne_total * 0.59)
    reported = {'Wayne': {'el_sayed': wayne_es,
                          'stevens': wayne_total - wayne_es,
                          'mode': 'mixed',
                          'pct_in': 1.0}}
    post = simulate(counties, method_table, reported=reported, seed=1)
    print("Wayne observed shift: {:+.1f} points".format(
        post['leverage'].iloc[0]['observed_shift']))
    print("Wayne leverage on statewide mean: {:.1%}".format(
        post['leverage'].iloc[0]['leverage']))
    print("Implied statewide shift: {:+.2f} points".format(post['implied_state_shift']))
    print("Median margin: El-Sayed {:+.1f}  (was {:+.1f})".format(
        post['median_margin'], pre['median_margin']))
    print("90% interval: {:+.1f} to {:+.1f}".format(
        post['margin_ci_lower'], post['margin_ci_upper']))
    print()
    print("Regional posterior shifts:")
    print(post['region_posterior_shift'].round(2).to_string())
