"""Pilot estimators for the normalized, centered q-Gaussian.

All methods fit (q, b), with mu fixed at zero. Results are deliberately
unconstrained by the generating parameters and use the same bounds/start.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

from .distributions import q_gaussian, q_gaussian_cdf, q_gaussian_norm, q_log


BOUNDS = ((1.01, 2.95), (0.01, 50.0))
START = (1.6, 2.0)


@dataclass
class Fit:
    q: float
    b: float
    success: bool
    message: str
    objective: float
    nfev: int


def _histogram(data, bins=50):
    # Equal-width bins over the observed range; zero-count bins are omitted
    # only for the q-log objective, which requires positive densities.
    density, edges = np.histogram(data, bins=bins, density=True)
    counts, _ = np.histogram(data, bins=edges)
    return (edges[1:] + edges[:-1]) / 2, density, counts


def fit(data, method, bins=50):
    """Fit one centered sample via pdf, mle, qlog, or cdf."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("data must be a finite one-dimensional sample")
    if method not in {"pdf", "mle", "qlog", "cdf"}:
        raise ValueError("unknown method")
    if method in {"pdf", "qlog"}:
        centers, heights, counts = _histogram(x, bins)
        used = counts > 0
        centers, heights, counts = centers[used], heights[used], counts[used]
        # Poisson count variance propagated into density and q-log density.
        widths = (x.max() - x.min()) / bins
        sigma = np.sqrt(counts) / (len(x) * widths)
    if method == "cdf":
        sorted_x = np.sort(x)
        empirical = (np.arange(len(x)) + 0.5) / len(x)

    def objective(p):
        q, b = p
        if method == "mle":
            a = q_gaussian_norm(b, q)
            return -len(x) * np.log(a) + np.log1p((q - 1) * b * x*x).sum() / (q - 1)
        if method == "cdf":
            return np.mean((q_gaussian_cdf(sorted_x, b, q) - empirical)**2)
        predicted = q_gaussian(centers, b, q)
        if method == "pdf":
            return np.mean(((heights - predicted) / sigma)**2)
        # Propagated histogram uncertainty: d ln_q(y)/dy = y**(-q).
        residual = (q_log(heights, q) - q_log(predicted, q)) * heights**q / sigma
        return np.mean(residual**2)

    try:
        result = minimize(objective, START, bounds=BOUNDS, method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-11})
        q, b = result.x
        valid = bool(result.success and np.isfinite(result.fun))
        return Fit(float(q), float(b), valid, str(result.message),
                   float(result.fun), int(result.nfev))
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return Fit(np.nan, np.nan, False, repr(exc), np.nan, 0)
