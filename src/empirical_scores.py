"""Common, descriptive scores for fitted q-Gaussians on real increments.

All metrics use the same observations for a given dataset. They describe
in-sample fit, not out-of-sample predictive performance or uncertainty.
"""

import numpy as np
from scipy.stats import t

from .distributions import q_gaussian, q_gaussian_cdf

TAIL_PERCENTILES = (1, 5, 10, 90, 95, 99)


def _sample(x):
    sample = np.asarray(x, dtype=float)
    if sample.ndim != 1 or len(sample) < 3 or not np.all(np.isfinite(sample)):
        raise ValueError("x must be a finite one-dimensional sample of at least three values")
    return sample


def _parameters(fit):
    if not fit.success or not (fit.b > 0 and 1 < fit.q < 3):
        raise ValueError("a successful fit with b>0 and 1<q<3 is required")
    nu = (3 - fit.q) / (fit.q - 1)
    scale = 1 / np.sqrt(fit.b * (3 - fit.q))
    return nu, scale


def body_scores(x, fit, bins=100):
    """Scores on a common sample and a common full-range histogram.

    dic = E_{F_n}[(F_n(X)-F_model(X))²], using the right-continuous
    empirical CDF at each observation (ties share their right rank). This is
    an integral against dF_n, not against dx. KS tests both sides of every
    empirical jump. The histogram R²/chi² use all/occupied bins respectively.
    """
    sample = _sample(x)
    nu, scale = _parameters(fit)
    sorted_x = np.sort(sample)
    n = len(sample)
    cdf = q_gaussian_cdf(sorted_x, fit.b, fit.q, fit.mu)
    right = np.searchsorted(sorted_x, sorted_x, side="right") / n
    left = np.searchsorted(sorted_x, sorted_x, side="left") / n
    dic = np.mean((cdf - right) ** 2)
    ks = max(np.max(np.abs(cdf - left)), np.max(np.abs(cdf - right)))

    counts, edges = np.histogram(sample, bins=bins)
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    density = counts / (n * widths)
    predicted = q_gaussian(centers, fit.b, fit.q, fit.mu)
    ss_tot = np.sum((density - density.mean()) ** 2)
    r2 = 1 - np.sum((density - predicted) ** 2) / ss_tot if ss_tot > 0 else np.nan
    occupied = counts > 0
    sigma = np.sqrt(counts[occupied]) / (n * widths[occupied])
    chi2 = np.sum(((density[occupied] - predicted[occupied]) / sigma) ** 2)
    mean_loglik = np.mean(t.logpdf((sample - fit.mu) / scale, nu) - np.log(scale))
    return {"dic_dfn": float(dic), "ks": float(ks), "r2_hist_100": float(r2),
            "chi2_hist_100": float(chi2), "mean_loglik": float(mean_loglik)}


def tail_thresholds(x, percentiles=TAIL_PERCENTILES):
    """Select thresholds once per dataset, independently of fitted methods."""
    sample = _sample(x)
    levels = np.asarray(percentiles, dtype=float)
    if np.any((levels <= 0) | (levels >= 100)):
        raise ValueError("percentiles must lie strictly between 0 and 100")
    return tuple((float(p), float(v)) for p, v in zip(levels, np.percentile(sample, levels)))


def tail_scores(x, fit, thresholds):
    """Left P(X<=u) and right P(X>u), model minus observed in percentage points.

    The return value is (six detailed threshold rows, mean absolute error).
    Thresholds should be produced once with tail_thresholds(x) and reused.
    """
    sample = _sample(x)
    nu, scale = _parameters(fit)
    rows = []
    for percentile, threshold in thresholds:
        side = "left" if percentile < 50 else "right"
        z = (threshold - fit.mu) / scale
        observed = (np.mean(sample <= threshold) if side == "left"
                    else np.mean(sample > threshold))
        predicted = (t.cdf(z, nu) if side == "left" else t.sf(z, nu))
        error_pp = 100 * (predicted - observed)
        rows.append({"percentile": percentile, "side": side, "threshold": threshold,
                     "observed_probability": float(observed),
                     "predicted_probability": float(predicted),
                     "error_pp": float(error_pp), "abs_error_pp": float(abs(error_pp))})
    return rows, float(np.mean([row["abs_error_pp"] for row in rows]))


def exceedance_curve(x, fit, percentiles=None):
    """Observed and model P(X>u) on a shared set of empirical thresholds."""
    sample = _sample(x)
    nu, scale = _parameters(fit)
    if percentiles is None:
        percentiles = np.linspace(0.1, 99.9, 400)
    thresholds = np.unique(np.percentile(sample, percentiles))
    observed = 1 - np.searchsorted(np.sort(sample), thresholds, side="right") / len(sample)
    predicted = t.sf((thresholds - fit.mu) / scale, nu)
    return thresholds, observed, predicted
