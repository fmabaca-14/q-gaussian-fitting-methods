"""Normalized Gaussian fitted to the empirical CDF."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class GaussianFit:
    mu: float = np.nan
    sigma: float = np.nan
    success: bool = False
    objective: float = np.nan
    message: str = ""


def fit_gaussian_cdf(x):
    """Fit location and positive standard deviation using midpoint ECDF SSE.

    Matches empirical.fit_unbinned(..., 'cdf')'s CDF target. Optimization is
    performed in standardized coordinates; no centering is imposed on the fit.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("x must be a finite one-dimensional sample")
    spread = float(np.std(x))
    if spread <= 0:
        return GaussianFit(message="zero sample spread")
    origin = float(np.median(x))
    z = np.sort((x-origin)/spread)
    target = (np.arange(len(x))+.5)/len(x)

    def objective(p):
        return np.mean((norm.cdf(z, loc=p[0], scale=np.exp(p[1]))-target)**2)

    robust = max(float(np.quantile(z,.75)-np.quantile(z,.25))/1.349, 1e-8)
    trials = [minimize(objective, start, method="L-BFGS-B",
                       bounds=((float(z.min()), float(z.max())), (-20., 20.)),
                       options={"ftol":1e-13,"maxiter":500})
              for start in ((0., np.log(robust)), (float(z.mean()), 0.))]
    valid = [r for r in trials if r.success and np.isfinite(r.fun)]
    if not valid:
        return GaussianFit(message="CDF optimization failed")
    best = min(valid,key=lambda r:r.fun)
    return GaussianFit(origin+spread*float(best.x[0]), spread*float(np.exp(best.x[1])),
                       True,float(best.fun),str(best.message))


def histogram_r2(x, pdf, bins=100):
    """Descriptive R² on all centers of one full-range density histogram."""
    density, edges = np.histogram(x,bins=bins,density=True)
    centers = (edges[:-1]+edges[1:])/2
    total = np.sum((density-density.mean())**2)
    return float(1-np.sum((density-pdf(centers))**2)/total) if total > 0 else np.nan
