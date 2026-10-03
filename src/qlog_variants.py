"""Experimental centered q-log variant for comparison with empirical q-log."""

import numpy as np

from .distributions import q_gaussian_norm, q_log
from .empirical import EmpiricalFit, Q_GRID, histogram


def fit_centered_qlog(x, bins=50, q_grid=Q_GRID, return_scan=False):
    """Fit ln_q(density)=intercept+slope*x² with mu fixed at zero.

    Data are NOT centered. For every q, weighted least squares use
    sigma_lnq = density**(-q)*sigma_density. Select the smallest raw weighted
    sum of squared residuals across valid grid candidates. This is a different
    objective from the current Pearson selection in empirical.fit_histogram
    (..., 'qlog', qlog_selection='pearson_counts'), which fixes mu=0 by default.

    The freely fitted intercept is not constrained by PDF normalization.
    This historical experiment infers b from the slope and normalization;
    it does not use the corrected intercept-and-slope recovery in empirical.py.
    The scan reports the discrepancy between the fitted
    intercept and ln_q(a(b,q)); selection scores describe the regression line,
    not necessarily the reconstructed normalized PDF.

    If return_scan=True, return (EmpiricalFit, list_of_candidate_dicts).
    Otherwise return EmpiricalFit. Scores across q depend on q-dependent
    transformations and propagated uncertainties; this variant is exploratory.
    """
    x = np.asarray(x, dtype=float)
    grid = np.asarray(q_grid, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        raise ValueError("x must be a finite nonconstant one-dimensional sample")
    if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or np.any((grid <= 1) | (grid >= 3)):
        raise ValueError("q_grid must contain finite values strictly between 1 and 3")
    _, _, centers, heights, sigma = histogram(x, bins)
    x2 = centers**2
    scale2 = max(float(np.max(x2)), np.finfo(float).tiny)
    design = np.column_stack((np.ones(len(centers)), x2 / scale2))
    log_y = np.log(heights)
    rows = []
    for q in grid:
        row = {"q": float(q), "b": np.nan, "mu": 0.0, "valid": False,
               "weighted_sse": np.nan, "intercept": np.nan, "slope": np.nan,
               "normalization_intercept_gap": np.nan}
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            response = np.expm1((1-q)*log_y) / (1-q)
            log_weight = q*log_y - np.log(sigma)
            weights = np.exp(log_weight - log_weight.max())
        if not np.all(np.isfinite(response)):
            rows.append(row)
            continue
        coef, _, rank, _ = np.linalg.lstsq(design*weights[:, None], response*weights, rcond=None)
        intercept, slope = float(coef[0]), float(coef[1] / scale2)
        row.update(intercept=intercept, slope=slope)
        if rank != 2 or slope >= 0:
            rows.append(row)
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            b = np.exp(2*(np.log(-slope) - (1-q)*np.log(q_gaussian_norm(1, q))) / (3-q))
            # Use unscaled uncertainties in the reported score. The scaling
            # above conditions the linear solve only and must not rank q.
            residual = (response - design @ coef) * np.exp(log_weight)
            score = float(np.sum(residual**2))
        if np.isfinite(b) and b > 0 and np.isfinite(score):
            row.update(b=float(b), valid=True, weighted_sse=score,
                       normalization_intercept_gap=float(intercept-q_log(q_gaussian_norm(b, q), q)))
        rows.append(row)
    valid = [row for row in rows if row["valid"]]
    if valid:
        best = min(valid, key=lambda row: row["weighted_sse"])
        result = EmpiricalFit("qlog_centered_weighted", best["b"], best["q"], 0.0, True,
                              best["weighted_sse"], "mu=0; q selected by weighted q-log regression SSE")
    else:
        result = EmpiricalFit("qlog_centered_weighted", mu=0.0, message="no valid centered q-log candidate")
    return (result, rows) if return_scan else result
