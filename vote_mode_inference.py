"""
Vote-mode inference for the Michigan live model.

Michigan county feeds do not tell you what you actually need to know: whether the
47,000 votes that just posted in Macomb are the absentee batch, the first Election
Day precincts, or a mix. Since the two modes are projected to differ by roughly 21
margin points, guessing wrong is the single fastest way to blow a call.

This module infers the mode from what IS observable: how many votes posted, and what
the margin looks like.

THE LATENT PARAMETER
    theta = share of a county's reported votes that were cast before Election Day

Expected margin at theta is the interpolation between the two mode baselines:

    mu(theta) = theta * early_margin + (1 - theta) * ed_margin

THE IDENTIFICATION PROBLEM
    A county reporting well above its blended baseline could mean the Election Day
    vote is landing (high 1-theta), or it could mean a genuine pro-El-Sayed shift.
    Margin alone cannot separate those. Read this honestly: the margin signal is
    weakly informative about theta whenever the shift prior is wide relative to the
    mode gap. Volume does most of the identifying work here, and the margin only
    breaks ties.

WHAT ACTUALLY PINS IT DOWN
    1. HARD VOLUME BOUNDS. A county cannot report more early votes than it has, or
       more Election Day votes than it has. With N reported against projected early
       pool E and Election Day pool D:

           theta in [ max(0, 1 - D/N),  min(1, E/N) ]

       These are constraints, not preferences, and they tighten fast as N grows.

    2. REPORTING-ORDER PRIOR. Absentee boards in Michigan tend to post in large
       batches while Election Day precincts trickle, so early dumps skew early-heavy.
       This is a tendency, not a rule, and it varies by county, so the prior is
       deliberately weak.

    3. MARGIN LIKELIHOOD. Given a prior on the county's true shift, the observed
       margin gives some evidence about theta. Weak, but not nothing.

OUTPUT
    Rather than committing to a single mode, the module returns the posterior mean
    expected baseline and the posterior VARIANCE of that baseline across theta. That
    variance is mode uncertainty, and it is added to the observation noise in the
    hierarchical model. So an ambiguous dump does not get treated as a confident
    read. It gets downweighted automatically, which is exactly what you want when you
    cannot tell what you are looking at.
"""

import numpy as np
import pandas as pd


ORDER_BIAS = 0.70        # weight on the "early reports first" tendency vs the
                         # county's overall early share
PRIOR_CONCENTRATION = 6.0  # Beta concentration on theta. Higher = trust the
                           # reporting-order prior more. 6 is deliberately weak.
SHIFT_PRIOR_SD = 8.0     # assumed sd of a county's true shift, used only to weigh
                         # how much the margin should inform theta
THETA_GRID = 81


def infer_county_mode(reported_votes: int,
                      observed_margin: float,
                      early_pool: float,
                      ed_pool: float,
                      early_margin: float,
                      ed_margin: float,
                      order_bias: float = None,
                      concentration: float = None,
                      shift_prior_sd: float = None,
                      sampling_var: float = 0.0,
                      shift_mean: float = 0.0,
                      grid_size: int = THETA_GRID) -> dict:
    """
    Posterior over theta for a single county's reported vote.

    shift_mean and shift_prior_sd carry the model's CURRENT belief about this
    county's true shift. On the first pass they are 0 and the wide prior. On later
    passes the hierarchical model feeds back its posterior, which is what removes the
    bias described in the module docstring: once the shift is known from elsewhere,
    theta no longer has to absorb it.

    Returns:
        dict with theta_mean, theta_sd, expected_baseline, baseline_var,
        theta_lower, theta_upper (the hard feasible bounds)
    """
    # Resolved at call time, not def time, so the module constants stay tunable.
    if order_bias is None:
        order_bias = ORDER_BIAS
    if concentration is None:
        concentration = PRIOR_CONCENTRATION
    if shift_prior_sd is None:
        shift_prior_sd = SHIFT_PRIOR_SD

    total_pool = early_pool + ed_pool
    n = float(reported_votes)

    if n <= 0 or total_pool <= 0:
        blended = ((early_pool * early_margin + ed_pool * ed_margin) / max(total_pool, 1e-9))
        return {
            'theta_mean': early_pool / max(total_pool, 1e-9),
            'theta_sd': 0.0,
            'expected_baseline': blended,
            'baseline_var': 0.0,
            'theta_lower': 0.0,
            'theta_upper': 1.0,
        }

    # 1. Hard feasible bounds from volume
    lower = max(0.0, 1.0 - ed_pool / n)
    upper = min(1.0, early_pool / n)
    if upper < lower:
        # Reported vote exceeds the projected total for this county. The turnout
        # projection is the thing that is wrong, not the mode. Fall back to the
        # pool ratio and flag it with wide uncertainty.
        ratio = early_pool / max(total_pool, 1e-9)
        return {
            'theta_mean': ratio,
            'theta_sd': 0.25,
            'expected_baseline': ratio * early_margin + (1 - ratio) * ed_margin,
            'baseline_var': (0.25 * abs(ed_margin - early_margin)) ** 2,
            'theta_lower': 0.0,
            'theta_upper': 1.0,
        }

    theta = np.linspace(lower, upper, grid_size)

    # 2. Reporting-order prior
    order_expectation = min(1.0, early_pool / n)      # if early reports first
    overall_share = early_pool / total_pool           # if reporting is proportional
    prior_mean = order_bias * order_expectation + (1 - order_bias) * overall_share
    prior_mean = float(np.clip(prior_mean, 1e-3, 1 - 1e-3))

    # Shape parameters are floored at 1 so the prior stays unimodal with its mode at
    # prior_mean. Without the floor, a prior_mean near 1 produces a Beta with b < 1,
    # which puts an infinite density spike on theta = 1 and makes the model certain
    # every dump is pure early vote regardless of what the margin says.
    a = 1.0 + prior_mean * concentration
    b = 1.0 + (1 - prior_mean) * concentration
    with np.errstate(divide='ignore', invalid='ignore'):
        log_prior = (a - 1) * np.log(np.clip(theta, 1e-9, 1)) \
                    + (b - 1) * np.log(np.clip(1 - theta, 1e-9, 1))
    log_prior = np.nan_to_num(log_prior, neginf=-1e9)

    # 3. Margin likelihood, marginalizing over the county's unknown true shift
    mu = theta * early_margin + (1 - theta) * ed_margin + shift_mean
    var = shift_prior_sd ** 2 + sampling_var
    log_lik = -0.5 * (observed_margin - mu) ** 2 / var

    log_post = log_prior + log_lik
    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= post.sum()

    theta_mean = float(np.sum(post * theta))
    theta_var = float(np.sum(post * (theta - theta_mean) ** 2))
    base = mu - shift_mean
    baseline_mean = float(np.sum(post * base))
    baseline_var = float(np.sum(post * (base - baseline_mean) ** 2))

    return {
        'theta_mean': theta_mean,
        'theta_sd': float(np.sqrt(theta_var)),
        'expected_baseline': baseline_mean,
        'baseline_var': baseline_var,
        'theta_lower': float(lower),
        'theta_upper': float(upper),
    }


def observed_shifts_with_mode_inference(counties: pd.DataFrame,
                                        method_table: pd.DataFrame,
                                        reported: dict,
                                        shift_mean: np.ndarray = None,
                                        shift_sd: np.ndarray = None,
                                        **infer_kwargs) -> tuple:
    """
    Compute reported-county shifts, inferring vote mode where it is not supplied.

    Each record in `reported` may optionally carry 'mode' ('early', 'ed', 'mixed') to
    override inference, or 'theta' to fix the early share directly.

    Returns:
        (indices, shifts, sampling_sd, mode_sd, diagnostics_frame)

        sampling_sd and mode_sd are returned SEPARATELY on purpose. Sampling error is
        independent across counties and averages away. Mode error does not: if the
        clerks are all posting absentee batches first, every county's theta is wrong
        in the same direction at the same time. The hierarchical model gives mode
        error a correlated block in the observation covariance so it cannot wash out
        across counties the way independent noise would.
    """
    idx_of = {c: i for i, c in enumerate(counties['county'])}
    table = method_table.set_index('county')

    n_counties = len(counties)
    if shift_mean is None:
        shift_mean = np.zeros(n_counties)
    if shift_sd is None:
        shift_sd = np.full(n_counties, SHIFT_PRIOR_SD)

    indices, shifts, sds, mode_sds, rows = [], [], [], [], []

    for county, rec in reported.items():
        if county not in idx_of:
            continue
        votes = rec['el_sayed'] + rec['stevens']
        if votes <= 0:
            continue

        row = table.loc[county]
        p = rec['el_sayed'] / votes
        observed = (2 * p - 1) * 100.0
        sampling_var = (100.0 * 2.0) ** 2 * max(p * (1 - p), 1e-6) / votes

        mode = rec.get('mode')
        theta_fixed = rec.get('theta')

        if theta_fixed is not None:
            th = float(theta_fixed)
            baseline = th * row['early_margin'] + (1 - th) * row['ed_margin']
            baseline_var = 0.0
            theta_mean, theta_sd = th, 0.0
            source = 'fixed_theta'
        elif mode == 'early':
            baseline, baseline_var = row['early_margin'], 0.0
            theta_mean, theta_sd, source = 1.0, 0.0, 'declared'
        elif mode in ('ed', 'election_day'):
            baseline, baseline_var = row['ed_margin'], 0.0
            theta_mean, theta_sd, source = 0.0, 0.0, 'declared'
        elif mode == 'mixed':
            baseline, baseline_var = row['blended_margin'], 0.0
            theta_mean = row['early_votes'] / row['total_votes']
            theta_sd, source = 0.0, 'declared'
        else:
            i = idx_of[county]
            post = infer_county_mode(
                reported_votes=votes,
                observed_margin=observed,
                early_pool=float(row['early_votes']),
                ed_pool=float(row['ed_votes']),
                early_margin=float(row['early_margin']),
                ed_margin=float(row['ed_margin']),
                sampling_var=sampling_var,
                shift_mean=float(shift_mean[i]),
                shift_prior_sd=float(shift_sd[i]),
                **infer_kwargs)
            baseline = post['expected_baseline']
            baseline_var = post['baseline_var']
            theta_mean, theta_sd = post['theta_mean'], post['theta_sd']
            source = 'inferred'

        indices.append(idx_of[county])
        shifts.append(observed - baseline)
        sds.append(np.sqrt(sampling_var))
        mode_sds.append(np.sqrt(baseline_var))
        rows.append({
            'county': county,
            'votes_reported': votes,
            'observed_margin': round(observed, 2),
            'theta_mean': round(theta_mean, 3),
            'theta_sd': round(theta_sd, 3),
            'implied_baseline': round(baseline, 2),
            'shift': round(observed - baseline, 2),
            'mode_uncertainty_sd': round(np.sqrt(baseline_var), 2),
            'source': source,
        })

    diagnostics = pd.DataFrame(rows)
    return (np.array(indices, dtype=int), np.array(shifts),
            np.array(sds), np.array(mode_sds), diagnostics)


if __name__ == '__main__':
    from michigan_primary_model import build_michigan_county_data
    from vote_method_split import build_vote_method_table

    counties = build_michigan_county_data()
    method_table, _ = build_vote_method_table(counties)
    macomb = method_table[method_table.county == 'Macomb'].iloc[0]

    print("Macomb: {:,} total, {:,} early ({:+.1f}), {:,} ED ({:+.1f})".format(
        macomb.total_votes, macomb.early_votes, macomb.early_margin,
        macomb.ed_votes, macomb.ed_margin))
    print()
    print("{:>9} {:>9} {:>9} {:>9} {:>18} {:>10}".format(
        "reported", "margin", "theta", "theta_sd", "feasible range", "mode_sd"))

    scenarios = [
        (20000, 6.0), (20000, 20.0),
        (70000, 7.0), (70000, 15.0),
        (95000, 10.0), (105000, 12.5),
    ]
    for n, m in scenarios:
        post = infer_county_mode(n, m, float(macomb.early_votes), float(macomb.ed_votes),
                                 float(macomb.early_margin), float(macomb.ed_margin))
        print("{:>9,} {:>+9.1f} {:>9.3f} {:>9.3f} {:>8.2f} - {:<7.2f} {:>10.2f}".format(
            n, m, post['theta_mean'], post['theta_sd'],
            post['theta_lower'], post['theta_upper'],
            np.sqrt(post['baseline_var'])))
