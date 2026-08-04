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


# Michigan counties usually clear the absentee batch before precinct returns, but
# not always. That is a statement about a MIXTURE of reporting regimes, not about
# an average, and the distinction matters more than it sounds.
#
# A single Beta centred between "early first" and "proportional" puts its mass in
# the gap between the two behaviours, which is where no county actually is. The
# margin signal is then too weak to pull theta anywhere, so the posterior just
# hands the prior back — which is exactly the failure measured earlier, theta
# landing near 0.83 whether the truth was 1.0 or 0.0.
#
# Modelled as separated components instead, the same weak margin signal has an
# easier job: discriminating between a few well-spaced hypotheses is far more
# tractable than locating a point on a continuum. The likelihood ratio between
# modes carries real information even when the gradient does not.
# Weights tuned against three ground-truth scenarios (pure early dump, genuinely
# mixed, Election-Day-first). Worst-case margin error fell from 10.4 points under
# the old single-Beta prior to 3.3, and all three now sit inside the 90% interval
# where only two did before.
REGIME_WEIGHTS = {
    'early_first':  0.55,   # absentee board clears before precincts. The plurality.
    'proportional': 0.35,   # precinct-by-precinct with absentee folded throughout
    'ed_first':     0.10,   # absentee delayed behind Election Day precincts
}
# Deliberately tight. Sharper components are further apart, and a weak margin
# signal discriminates between separated hypotheses far better than it locates a
# point on a continuum. Loosening this collapses the mixture back toward one blur
# and the inference stops working.
REGIME_CONCENTRATION = 30.0
ORDER_BIAS = 0.70        # retained for the legacy single-Beta path
PRIOR_CONCENTRATION = 6.0  # Beta concentration on theta. Higher = trust the
                           # reporting-order prior more. 6 is deliberately weak.
SHIFT_PRIOR_SD = 8.0     # assumed sd of a county's true shift, used only to weigh
                         # how much the margin should inform theta
THETA_GRID = 81


# ---------------------------------------------------------------------------
# Within-county heterogeneity (design effect)
#
# A partially counted county is not a random sample of itself. It is whichever
# precincts happened to report, and in a county that contains genuinely different
# electorates those precincts are a biased draw. Binomial sampling error assumes
# the opposite and shrinks as 1/sqrt(n), so at 20% counted it claims a precision
# the data cannot support.
#
# Wayne is the extreme case: Detroit, Dearborn, Grosse Pointe, Livonia and the
# downriver suburbs do not vote alike, so an early Wayne read that looks nothing
# like the statewide picture is usually telling you which part of Wayne reported,
# not that the state has moved.
#
# The correction adds variance proportional to how much of the county is still
# out, and it vanishes entirely at 100% counted — a fully counted county is not a
# sample at all, it is the answer, and gets trusted completely.
#
#     design_var = heterogeneity^2 * (1 - completeness)
#
# Values are in margin points: roughly how far a half-counted county can sit from
# its own final margin purely through which precincts landed first.
COUNTY_HETEROGENEITY = {
    'Wayne': 14.0,       # Detroit vs the western and downriver suburbs
    'Oakland': 8.0,      # Pontiac and Southfield vs Bloomfield and Rochester
    'Genesee': 7.0,      # Flint vs the rest of the county
    'Kent': 6.0,         # Grand Rapids vs rural Kent
    'Macomb': 6.0,       # Warren and Eastpointe vs north Macomb
    'Washtenaw': 6.0,    # Ann Arbor and Ypsilanti vs the townships
    'Ingham': 5.0,       # Lansing and East Lansing vs the balance
    'Saginaw': 5.0,      # Saginaw city vs the county
    'Kalamazoo': 4.0,
    'Muskegon': 4.0,
    'Berrien': 4.0,
    'Calhoun': 4.0,
}
DEFAULT_HETEROGENEITY = 2.0   # small and rural counties are far more uniform


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

    # 2. Reporting-regime mixture prior
    #
    # Each regime implies a different theta for the same observed volume:
    #   early_first   all early clears before any Election Day vote posts
    #   proportional  the batch mirrors the county's overall early share
    #   ed_first      Election Day precincts land ahead of the absentee board
    centres = {
        'early_first':  min(1.0, early_pool / n),
        'proportional': early_pool / total_pool,
        'ed_first':     max(0.0, 1.0 - ed_pool / n),
    }

    log_theta = np.log(np.clip(theta, 1e-9, 1))
    log_1mtheta = np.log(np.clip(1 - theta, 1e-9, 1))

    components = []
    for regime, weight in REGIME_WEIGHTS.items():
        if weight <= 0:
            continue
        centre = float(np.clip(centres[regime], 1e-3, 1 - 1e-3))
        # Shapes floored at 1 so no component spikes at a boundary. Without the
        # floor a centre near 1 gives b < 1 and infinite density at theta = 1,
        # which makes the model certain every dump is pure absentee.
        a = 1.0 + centre * REGIME_CONCENTRATION
        b = 1.0 + (1 - centre) * REGIME_CONCENTRATION
        with np.errstate(divide='ignore', invalid='ignore'):
            lp = (a - 1) * log_theta + (b - 1) * log_1mtheta
        lp = np.nan_to_num(lp, neginf=-1e9)
        lp -= lp.max()
        components.append(np.log(weight) + lp - np.log(np.exp(lp).sum() + 1e-300))

    stacked = np.vstack(components)
    top = stacked.max(axis=0)
    log_prior = top + np.log(np.exp(stacked - top).sum(axis=0))

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

        # Design effect: how unrepresentative the counted portion may be of the
        # whole county. Full at 0% counted, gone at 100%.
        completeness = min(votes / max(float(row['total_votes']), 1.0), 1.0)
        hetero = COUNTY_HETEROGENEITY.get(county, DEFAULT_HETEROGENEITY)
        design_var = (hetero ** 2) * (1.0 - completeness)

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
        sds.append(np.sqrt(sampling_var + design_var))
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
            'completeness': round(completeness, 3),
            'design_sd': round(np.sqrt(design_var), 2),
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
