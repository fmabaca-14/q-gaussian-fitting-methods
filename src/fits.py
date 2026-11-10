"""The four centered, normalized q-Gaussian estimators used in the paper."""

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, minimize
from scipy.special import gammaln
from scipy.stats import t


METHODS = ("pdf", "qlog", "mle", "cdf")
Q_GRID = np.linspace(1.02, 2.98, 197)
SWEEP_BINS = tuple(range(10, 111, 2))
TABLE_BINS = (20, 50, 100)


@dataclass(frozen=True)
class Fit:
    b: float
    q: float
    objective: float


@dataclass(frozen=True)
class Histogram:
    counts: np.ndarray
    edges: np.ndarray
    centers: np.ndarray
    density: np.ndarray

    @classmethod
    def from_sample(cls, x, bins):
        counts, edges = np.histogram(x, bins=bins)
        return cls(counts, edges, (edges[:-1] + edges[1:]) / 2, counts / (len(x) * np.diff(edges)))


def log_amplitude(b, q):
    """Log of a(b,q) for b>0 and 1<q<3."""
    return (0.5 * np.log((q - 1) * b / np.pi) + gammaln(1 / (q - 1))
            - gammaln((3 - q) / (2 * (q - 1))))


def density(x, b, q):
    """Normalized density with symmetry center zero."""
    return np.exp(log_amplitude(b, q)) * (1 + (q - 1) * b * np.asarray(x)**2)**(-1 / (q - 1))


def cdf(x, b, q):
    """Analytic CDF via the equivalent Student-t distribution."""
    scale = 1 / np.sqrt(b * (3 - q))
    return t.cdf(np.asarray(x) / scale, df=(3 - q) / (q - 1))


def fit_direct(hist, spread):
    """Minimize unweighted density SSE over occupied bins (Eq. 8)."""
    used = hist.counts > 0
    centers, heights = hist.centers[used], hist.density[used]
    # Scaling both sides by the sample SD conditions the optimization.
    # It does not subtract a location or change the minimizer.
    u, target = centers / spread, heights * spread

    def residual(theta):
        return density(u, np.exp(theta[0]), theta[1]) - target

    trials = [least_squares(residual, [np.log(b0), q0], bounds=([-30., 1.0001], [30., 2.9999]),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
              for q0 in (1.1, 1.5, 2.0, 2.5, 2.9) for b0 in (.5, 5., 50.)]
    valid = [r for r in trials if r.success and np.all(np.isfinite(r.fun))]
    if not valid:
        raise RuntimeError("Direct PDF optimization failed from all starting points")
    best = min(valid, key=lambda r: np.sum(r.fun**2))
    b, q = float(np.exp(best.x[0]) / spread**2), float(best.x[1])
    return Fit(b, q, float(np.sum((heights - density(centers, b, q))**2)))


def bin_probabilities(edges, b, q):
    """Integrate the normalized PDF in each bin, in the coordinate x*sqrt(b)."""
    u = np.asarray(edges) * np.sqrt(b)
    amp, power, shape = float(np.exp(log_amplitude(1., q))), -1 / (q - 1), q - 1

    def scaled_density(v):
        return amp * (1 + shape * v*v)**power

    probs = []
    for lo, hi in zip(u[:-1], u[1:]):
        mass, error = quad(scaled_density, lo, hi, points=[0.] if lo < 0 < hi else None,
                           epsabs=1e-13, epsrel=1e-9, limit=200)
        if not np.isfinite(mass) or mass <= 0 or error > max(1e-12, 1e-7 * mass):
            raise FloatingPointError("PDF bin quadrature failed")
        probs.append(mass)
    return np.asarray(probs)


def fit_qlog(hist):
    """Fit A+B*x² for each q, then choose by Pearson expected-count discrepancy."""
    used = hist.counts > 0
    centers, heights = hist.centers[used], hist.density[used]
    scale = max(float(np.std(centers)), 1e-10)
    x2 = (centers / scale)**2
    design = np.column_stack((np.ones_like(x2), x2))
    log_y = np.log(heights)
    candidates = []
    for q in Q_GRID:
        response = np.expm1((1 - q) * log_y) / (1 - q)
        # Uniform reference scale in density -> sigma_z proportional to y**(-q).
        log_weight = q * log_y
        weight = np.exp(log_weight - log_weight.max())
        (intercept, slope), _, rank, _ = np.linalg.lstsq(
            design * weight[:, None], response * weight, rcond=None)
        slope = slope / scale**2
        denominator = 1 + (1 - q) * intercept
        if rank < 2 or slope >= 0 or not np.isfinite(denominator) or denominator <= 0:
            continue
        b = float(-slope / denominator)
        if not np.isfinite(b) or b <= 0:
            continue
        try:
            expected = hist.counts.sum() * bin_probabilities(hist.edges, b, q)
        except FloatingPointError:
            continue
        # Score all bins, including zeros. Do not renormalize within the observed range.
        score = float(np.sum((hist.counts - expected)**2 / np.maximum(expected, 1e-10)))
        if np.isfinite(score):
            candidates.append((score, b, float(q)))
    if not candidates:
        raise RuntimeError("No valid q-log candidate in the manuscript grid")
    score, b, q = min(candidates)
    return Fit(b, q, score)


def fit_unbinned(x, method):
    """Fit once per dataset by marginal likelihood or midpoint ECDF SSE."""
    if method not in ("mle", "cdf"):
        raise ValueError(f"Unknown unbinned method: {method}")
    spread = np.std(x, ddof=1)
    sorted_x = np.sort(x) if method == "cdf" else None
    ranks = midpoint_ranks(len(x)) if method == "cdf" else None

    def objective(theta):
        b, q = np.exp(theta[0]), theta[1]
        if method == "cdf":
            return np.mean((cdf(sorted_x, b, q) - ranks)**2)
        return -len(x) * log_amplitude(b, q) + np.log1p((q - 1) * b * x**2).sum() / (q - 1)

    log_b = np.log(1 / (2 * spread**2))
    starts = ((log_b, 1.5), (log_b, 2.4), (log_b + np.log(4), 1.8))
    bounds = ((np.log(1e-8), np.log(1e9)), (1.001, 2.99))
    trials = [minimize(objective, start, bounds=bounds, method="L-BFGS-B",
                       options={"maxiter": 350, "ftol": 1e-12}) for start in starts]
    valid = [r for r in trials if r.success and np.isfinite(r.fun)]
    if not valid:
        raise RuntimeError(f"{method.upper()} optimization failed from all starting points")
    best = min(valid, key=lambda r: r.fun)
    return Fit(float(np.exp(best.x[0])), float(best.x[1]), float(best.fun))


def midpoint_ranks(n):
    return (np.arange(n) + .5) / n


def cumulative_mean(errors):
    return np.cumsum(errors) / np.arange(1, len(errors) + 1)


def diagnostics(sorted_x, hist, fit):
    """Return manuscript statistics and the errors used by the plots."""
    n = len(sorted_x)
    fitted = density(hist.centers, fit.b, fit.q)
    error = np.abs(hist.density - fitted)
    k = ceil(.10 * len(hist.counts))
    cvm = np.mean((cdf(sorted_x, fit.b, fit.q) - midpoint_ranks(n))**2) + 1 / (12 * n*n)
    variance = np.sum((hist.density - hist.density.mean())**2)
    r2 = 1 - np.sum(error**2) / variance if variance > 0 else np.nan
    return {"q": fit.q, "b": fit.b, "cvm": float(cvm), "r2": float(r2),
            "mae_left": float(error[:k].mean()), "mae_right": float(error[-k:].mean()),
            "mae": float(error.mean())}, fitted, error
