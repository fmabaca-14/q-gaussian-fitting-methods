"""Paper estimators: mu=0, uniform PDF errors, original Pearson-selected q-log.

No sample mean or median is subtracted. Historical free-location/Poisson fits
remain available through explicit keyword arguments.
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


def load_increments(solar, bitcoin, discharge, discharge_transform="mean-scaled"):
    """Return chronological increments, omitting pairs across missing times.

    Solar: symmetric difference; Bitcoin: log-return; river: diff(Q)/mean(Q).
    The river mean uses all cleaned observations, including those adjacent to
    gaps. Gaps are excluded from pairs, not from that mean. No centering or
    smoothing is applied. River zero values are allowed for mean-scaled/diff.
    """
    if discharge_transform not in {"mean-scaled", "difference", "symmetric"}:
        raise ValueError("unknown discharge transform")
    specs = (
        ("solar", load_solar_wind(solar), "np", "h", False),
        ("bitcoin", load_bitcoin(bitcoin), "close", "D", True),
        ("discharge", load_discharge(discharge), "value", "D", False),
    )
    datasets = {}
    for name, frame, column, cadence, use_log in specs:
        values = frame[column].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0) or (
                name != "discharge" and np.any(values == 0)) or (
                name == "discharge" and discharge_transform == "symmetric" and np.any(values == 0)):
            raise ValueError(f"{name}: invalid {column} values for increment definition")
        valid = np.diff(frame.index.values) == np.timedelta64(1, cadence)
        mean_q = float(values.mean()) if name == "discharge" else None
        if name == "discharge" and discharge_transform != "symmetric":
            if mean_q <= 0:
                raise ValueError("discharge: mean flow must be positive")
            increments = np.diff(values)
            if discharge_transform == "mean-scaled":
                increments = increments / mean_q
        else:
            increments = (np.log(values[1:] / values[:-1]) if use_log else
                          2 * (values[1:] - values[:-1]) / (values[1:] + values[:-1]))
        x = increments[valid]
        if len(x) < 3:
            raise ValueError(f"{name}: too few consecutive observations")
        datasets[name] = {"x": x, "raw_n": len(frame), "n_gaps": int((~valid).sum()),
                          "first": str(frame.index.min()), "last": str(frame.index.max()),
                          "transform": (discharge_transform if name == "discharge" else
                                        "log-return" if use_log else "symmetric"),
                          "mean_q": mean_q}
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


def _qlog_fit(counts, edges, centers, heights, sigma, q_grid, centered=True):
    """Regress ln_q density on (1,x²), or (1,x,x²) for historical free mu.

    The intercept is fitted freely; the slope is mapped to b using normalized
    q-Gaussian amplitude. Bin-center approximation can be poor in sparse tails.
    """
    grid = np.asarray(q_grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or np.any((grid <= 1) | (grid >= 3)):
        raise ValueError("q_grid must contain finite values strictly between 1 and 3")
    scale = max(float(np.std(centers)), 1e-10)
    xx = centers / scale
    design = (np.column_stack((np.ones(len(xx)), xx * xx)) if centered else
              np.column_stack((np.ones(len(xx)), xx, xx * xx)))
    log_y = np.log(heights)
    candidates = []
    for q in grid:
        with np.errstate(over="ignore", invalid="ignore"):
            response = np.expm1((1 - q) * log_y) / (1 - q)
            log_weight = q * log_y - np.log(sigma)
        if not np.all(np.isfinite(response)):
            continue
        weight = np.exp(log_weight - log_weight.max())
        coef, _, rank, _ = np.linalg.lstsq(design * weight[:, None], response * weight, rcond=None)
        if rank != design.shape[1] or coef[-1] >= 0:
            continue
        mu = 0.0 if centered else -coef[1] * scale / (2 * coef[2])
        slope = coef[-1] / scale**2
        log_b = 2 * (np.log(-slope) - (1 - q) * np.log(q_gaussian_norm(1, q))) / (3 - q)
        if not np.isfinite(log_b) or not -20 < log_b < 20 or (
                not centered and not edges[0] < mu < edges[-1]):
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


def fit_histogram(x, bins=50, method="pdf", q_grid=Q_GRID, *,
                  centered=True, weighting="uniform"):
    """Fit PDF/q-log with paper defaults; weighting refers to PDF errors.

    Q-log propagates uniform PDF errors: its squared regression weights are
    density**(2*q). It still selects q by Pearson expected-count discrepancies.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)) or np.ptp(x) == 0:
        raise ValueError("x must be a finite nonconstant one-dimensional sample")
    if weighting not in {"uniform", "poisson"}:
        raise ValueError("weighting must be uniform or poisson")
    counts, edges, centers, heights, sigma = histogram(x, bins)
    errors = np.ones_like(sigma) if weighting == "uniform" else sigma
    if method == "qlog":
        return _qlog_fit(counts, edges, centers, heights, errors, q_grid, centered)
    if method != "pdf":
        raise ValueError("Histogram method must be pdf or qlog")
    sd = np.std(x, ddof=1)
    try:
        model = (lambda v, b, q: q_gaussian(v, b, q, 0.0)) if centered else q_gaussian
        # Extreme heavy-tailed samples can make the variance-based starting
        # b smaller than the optimizer's lower bound. Keep the start feasible.
        b0 = max(1 / (2 * sd**2), 1.01e-10)
        p0 = (b0, 1.3) + (() if centered else (np.median(x),))
        bounds = (([1e-10, 1.0001], [np.inf, 2.9999]) if centered else
                  ([1e-10, 1.0001, -np.inf], [np.inf, 2.9999, np.inf]))
        parameters, _ = curve_fit(model, centers, heights, p0=p0,
            sigma=None if weighting == "uniform" else sigma,
            absolute_sigma=True, bounds=bounds, maxfev=30000)
        b, q = map(float, parameters[:2])
        mu = 0.0 if centered else float(parameters[2])
        score = float(np.sum(((heights - q_gaussian(centers, b, q, mu)) / errors)**2))
        return EmpiricalFit("pdf", b, q, mu, True, score, f"{weighting} PDF least squares")
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return EmpiricalFit("pdf", message=str(error))


def fit_unbinned(x, method, *, centered=True):
    """Fit MLE or empirical CDF on observations without using bins."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional sample")
    if method not in ("mle", "cdf"):
        raise ValueError("Unbinned method must be mle or cdf")
    sd = np.std(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return EmpiricalFit(method, message="zero or nonfinite sample spread")
    med = float(np.median(x))
    sorted_x = np.sort(x) if method == "cdf" else None
    target = (np.arange(len(x)) + .5) / len(x) if method == "cdf" else None

    def objective(theta):
        b, q = np.exp(theta[0]), theta[1]
        mu = 0.0 if centered else theta[2]
        if method == "mle":
            return _negative_log_likelihood(x, b, q, mu)
        return np.mean((q_gaussian_cdf(sorted_x, b, q, mu) - target)**2)

    log_b = np.log(1 / (2 * sd**2))
    starts = ((log_b, 1.5, med), (log_b, 2.4, med), (log_b + np.log(4), 1.8, med))
    bounds = ((np.log(1e-8), np.log(1e9)), (1.001, 2.99), (float(x.min()), float(x.max())))
    if centered:
        starts = tuple(start[:2] for start in starts)
        bounds = bounds[:2]
    trials = [minimize(objective, start, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 350, "ftol": 1e-12}) for start in starts]
    valid = [trial for trial in trials if trial.success and np.isfinite(trial.fun)]
    if not valid:
        return EmpiricalFit(method, message="optimization failed")
    best = min(valid, key=lambda trial: trial.fun)
    return EmpiricalFit(method, float(np.exp(best.x[0])), float(best.x[1]),
                        0.0 if centered else float(best.x[2]), True, float(best.fun), str(best.message))


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
