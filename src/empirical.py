"""Exploratory q-Gaussian fits for the original, uncentered real-data increments.

This module is separate from ``estimators.py``: the synthetic pilot there
fixes mu=0 and uses a different, tail-aware histogram construction.
Here the direct PDF reproduces the paper's Poisson-weighted density fit on
equal-width bins. Q-log uses a quadratic regression to allow a free mu.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import gammaln

from .distributions import q_gaussian, q_gaussian_cdf, q_gaussian_norm
from .real_data import load_bitcoin, load_discharge, load_solar_wind

Q_GRID = np.linspace(1.02, 2.98, 197)  # step 0.01
METHODS = ("pdf", "mle", "cdf", "qlog")


@dataclass
class EmpiricalFit:
    method: str
    b: float = np.nan
    q: float = np.nan
    mu: float = np.nan
    success: bool = False
    objective: float = np.nan
    message: str = ""


def load_increments(solar, bitcoin, discharge):
    """Return chronological increments, omitting pairs across missing times.

    Solar and river: 2*(next-current)/(next+current). Bitcoin: log(next/current).
    No centering or rescaling is applied. Solar density and river flow must be
    positive; otherwise the normalized difference is undefined here.
    """
    specs = (
        ("solar", load_solar_wind(solar), "np", "h", False),
        ("bitcoin", load_bitcoin(bitcoin), "close", "D", True),
        ("discharge", load_discharge(discharge), "value", "D", False),
    )
    datasets = {}
    for name, frame, column, cadence, use_log in specs:
        values = frame[column].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"{name}: {column} must contain positive finite values")
        valid = np.diff(frame.index.values) == np.timedelta64(1, cadence)
        increments = (np.log(values[1:] / values[:-1]) if use_log else
                      2 * (values[1:] - values[:-1]) / (values[1:] + values[:-1]))
        x = increments[valid]
        if len(x) < 3:
            raise ValueError(f"{name}: too few consecutive observations")
        datasets[name] = {"x": x, "raw_n": len(frame), "n_gaps": int((~valid).sum()),
                          "first": str(frame.index.min()), "last": str(frame.index.max())}
    return datasets


def histogram(x, bins):
    """Equal-width, full-range histogram and Poisson density errors."""
    if bins < 4:
        raise ValueError("At least four bins are needed for the free-mu q-log fit")
    counts, edges = np.histogram(x, bins=bins)
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    positive = counts > 0
    heights = counts[positive] / (len(x) * widths[positive])
    sigma = np.sqrt(counts[positive]) / (len(x) * widths[positive])
    return counts, edges, centers[positive], heights, sigma


def _negative_log_likelihood(x, b, q, mu):
    log_a = (.5 * np.log((q - 1) * b / np.pi) + gammaln(1 / (q - 1))
             - gammaln((3 - q) / (2 * (q - 1))))
    return -len(x) * log_a + np.log1p((q - 1) * b * (x - mu) ** 2).sum() / (q - 1)


def _qlog_fit(counts, edges, centers, heights, sigma, q_grid):
    """Weighted ln_q density versus (1,x,x²), then choose q by Pearson score.

    The intercept is fitted freely; the slope is mapped to b using normalized
    q-Gaussian amplitude. Bin-center approximation can be poor in sparse tails.
    """
    scale = max(float(np.std(centers)), 1e-10)
    xx = centers / scale
    design = np.column_stack((np.ones(len(xx)), xx, xx * xx))
    log_y = np.log(heights)
    candidates = []
    for q in q_grid:
        with np.errstate(over="ignore", invalid="ignore"):
            response = np.expm1((1 - q) * log_y) / (1 - q)
            log_weight = q * log_y - np.log(sigma)
        if not np.all(np.isfinite(response)):
            continue
        weight = np.exp(log_weight - log_weight.max())
        coef, _, rank, _ = np.linalg.lstsq(design * weight[:, None], response * weight, rcond=None)
        if rank != 3 or coef[2] >= 0:
            continue
        mu = -coef[1] * scale / (2 * coef[2])
        slope = coef[2] / scale**2
        log_b = 2 * (np.log(-slope) - (1 - q) * np.log(q_gaussian_norm(1, q))) / (3 - q)
        if not np.isfinite(log_b) or not -20 < log_b < 20 or not edges[0] < mu < edges[-1]:
            continue
        b = np.exp(log_b)
        expected = counts.sum() * np.diff(q_gaussian_cdf(edges, b, q, mu))
        score = np.sum((counts - expected)**2 / np.maximum(expected, 1e-10))
        if np.isfinite(score):
            candidates.append((float(score), float(b), float(q), float(mu)))
    if not candidates:
        return EmpiricalFit("qlog", message="no valid q-log candidate")
    score, b, q, mu = min(candidates)
    return EmpiricalFit("qlog", b, q, mu, True, score,
                        "q selected by Pearson histogram score")


def fit_histogram(x, bins=50, method="pdf", q_grid=Q_GRID):
    """Fit PDF or q-log with b>0, 1<q<3 and free location mu."""
    x = np.asarray(x, dtype=float)
    counts, edges, centers, heights, sigma = histogram(x, bins)
    if method == "qlog":
        return _qlog_fit(counts, edges, centers, heights, sigma, q_grid)
    if method != "pdf":
        raise ValueError("Histogram method must be pdf or qlog")
    sd = np.std(x, ddof=1)
    try:
        parameters, _ = curve_fit(
            q_gaussian, centers, heights, p0=(1 / (2 * sd**2), 1.3, np.median(x)),
            sigma=sigma, absolute_sigma=True,
            bounds=([1e-10, 1.0001, -np.inf], [np.inf, 2.9999, np.inf]), maxfev=30000)
        b, q, mu = map(float, parameters)
        score = float(np.sum(((heights - q_gaussian(centers, b, q, mu)) / sigma)**2))
        return EmpiricalFit("pdf", b, q, mu, True, score, "weighted PDF least squares")
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return EmpiricalFit("pdf", message=str(error))


def fit_unbinned(x, method):
    """Fit MLE or empirical CDF on observations without using bins."""
    x = np.asarray(x, dtype=float)
    if method not in ("mle", "cdf"):
        raise ValueError("Unbinned method must be mle or cdf")
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return EmpiricalFit(method, message="zero or nonfinite sample spread")
    med = float(np.median(x))
    sorted_x = np.sort(x) if method == "cdf" else None
    target = (np.arange(len(x)) + .5) / len(x) if method == "cdf" else None

    def objective(theta):
        b, q, mu = np.exp(theta[0]), theta[1], theta[2]
        if method == "mle":
            return _negative_log_likelihood(x, b, q, mu)
        return np.mean((q_gaussian_cdf(sorted_x, b, q, mu) - target)**2)

    log_b = np.log(1 / (2 * sd**2))
    starts = ((log_b, 1.5, med), (log_b, 2.4, med), (log_b + np.log(4), 1.8, med))
    bounds = ((np.log(1e-8), np.log(1e9)), (1.001, 2.99), (float(x.min()), float(x.max())))
    trials = [minimize(objective, start, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 350, "ftol": 1e-12}) for start in starts]
    valid = [trial for trial in trials if trial.success and np.isfinite(trial.fun)]
    if not valid:
        return EmpiricalFit(method, message="optimization failed")
    best = min(valid, key=lambda trial: trial.fun)
    return EmpiricalFit(method, float(np.exp(best.x[0])), float(best.x[1]),
                        float(best.x[2]), True, float(best.fun), str(best.message))


def diagnostics(x, result, bins=50):
    """Descriptive metrics on the common 50-bin histogram and empirical CDF."""
    if not result.success:
        return {"r2_hist": np.nan, "chi2_hist": np.nan, "cdf_max_abs": np.nan}
    _, _, centers, heights, sigma = histogram(x, bins)
    predicted = q_gaussian(centers, result.b, result.q, result.mu)
    sorted_x = np.sort(x)
    empirical = (np.arange(len(x)) + .5) / len(x)
    return {"r2_hist": float(1 - np.sum((heights - predicted)**2) /
                             np.sum((heights - heights.mean())**2)),
            "chi2_hist": float(np.sum(((heights - predicted) / sigma)**2)),
            "cdf_max_abs": float(np.max(np.abs(q_gaussian_cdf(
                sorted_x, result.b, result.q, result.mu) - empirical)))}
