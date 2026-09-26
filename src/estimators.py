"""Estimadores exploratorios para la q-Gaussiana normalizada con mu=0.

PDF: verosimilitud multinomial de conteos en bins; usa todos los datos.
MLE: verosimilitud de cada observacion sin agrupar.
CDF: distancia cuadratica con la CDF empirica.
QLOG: para cada q de una grilla, regresion ponderada de ln_q(densidad)
      contra x**2; convierte la pendiente a b mediante la normalizacion y selecciona q
      evaluando la PDF normalizada en la escala original de conteos.

El metodo QLOG aproxima la densidad de cada bin en su centro. En bins de cola
muy anchos, esta aproximacion puede ser mala: informar siempre el diagnostico.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import t

from .distributions import q_gaussian_cdf, q_gaussian_norm

Q_BOUNDS = (1.01, 2.99)
B_BOUNDS = (0.01, 50.0)
STARTS = ((1.4, 2.0), (2.4, 2.0))
DEFAULT_Q_GRID = np.linspace(1.02, 2.98, 99)


@dataclass
class Fit:
    q: float = np.nan
    b: float = np.nan
    success: bool = False
    message: str = ""
    objective: float = np.nan
    nfev: int = 0
    occupied_bins: int = 0
    at_bound: bool = False


def histogram(data, bins=50):
    """Bins simetricos; todos los valores pertenecen a algun bin.

    El 80 % de los bins cubre [-r,r], r=percentil 90 de |x|. El resto
    crece geometricamente hasta el maximo observado. La division central
    no depende de la posicion de la observacion mas extrema.
    """
    if bins < 10 or bins % 10:
        raise ValueError("bins debe ser multiplo de 10 y >= 10")
    absx = np.abs(data)
    r = max(float(np.quantile(absx, .9)), 1e-10)
    outer = max(float(absx.max()) * (1 + 1e-9), r * (1 + 1e-9))
    n_tail = bins // 10
    middle = np.linspace(-r, r, bins - 2 * n_tail + 1)
    positive_tail = np.geomspace(r, outer, n_tail + 1)[1:]
    edges = np.r_[-positive_tail[::-1], middle, positive_tail]
    counts, _ = np.histogram(data, bins=edges)
    assert int(counts.sum()) == len(data)
    return edges, counts


def _bin_probabilities(edges, q, b):
    # Evita cancelacion 1-F en los bins positivos de la cola.
    nu = (3 - q) / (q - 1)
    scale = 1 / np.sqrt(b * (3 - q))
    z = edges / scale
    cdf = t.cdf(z, df=nu)
    survival = t.sf(z, df=nu)
    left = z[:-1]
    probs = np.where(left >= 0, survival[:-1] - survival[1:], cdf[1:] - cdf[:-1])
    cross = (z[:-1] < 0) & (z[1:] > 0)
    probs[cross] = cdf[1:][cross] - cdf[:-1][cross]
    return np.maximum(probs, np.finfo(float).tiny)


def _deviance(counts, probs):
    # Dos veces la distancia de log-verosimilitud a un histograma saturado.
    expected = counts.sum() * probs
    nonzero = counts > 0
    return 2 * (np.sum(counts[nonzero] * np.log(counts[nonzero] / expected[nonzero]))
                - np.sum(counts - expected))


def _qlog_fit(edges, counts, q_grid):
    widths = np.diff(edges)
    occupied = counts > 0
    centers = (edges[:-1] + edges[1:]) / 2
    y = counts[occupied] / (counts.sum() * widths[occupied])
    x2 = centers[occupied]**2
    # Reescalamiento para acondicionar la regresion sin alterar x**2.
    x_scale2 = max(float(np.quantile(x2, .75)), 1e-10)
    X = np.column_stack([np.ones(len(x2)), x2 / x_scale2])
    logy = np.log(y)
    log_sigma = .5 * np.log(counts[occupied]) - np.log(counts.sum() * widths[occupied])
    best = Fit(message="ningun candidato q-log valido", occupied_bins=int(occupied.sum()))
    for q in q_grid:
        # ln_q(y) = (y**(1-q)-1)/(1-q), calculado de forma estable.
        with np.errstate(over="ignore", invalid="ignore"):
            response = np.expm1((1-q) * logy) / (1-q)
            log_w_sqrt = q * logy - log_sigma  # sigma_lnq = y**(-q)*sigma_y
        if not np.all(np.isfinite(response)):
            continue
        weights_sqrt = np.exp(log_w_sqrt - np.max(log_w_sqrt))
        coef, _, rank, _ = np.linalg.lstsq(X * weights_sqrt[:, None],
                                           response * weights_sqrt, rcond=None)
        if rank < 2:
            continue
        intercept, slope_scaled = coef
        slope = slope_scaled / x_scale2
        if slope >= 0 or not np.isfinite(intercept):
            continue
        # A(b,q)=A(1,q)*sqrt(b). Por normalizacion:
        # -pendiente=A(1,q)**(1-q) * b**((3-q)/2).
        # El intercepto se estima libre durante la regresion, pero no se
        # interpreta como amplitud independiente del parametro b.
        log_c = np.log(q_gaussian_norm(1., q))
        with np.errstate(over="ignore", invalid="ignore"):
            b = np.exp(2 * (np.log(-slope) - (1-q)*log_c) / (3-q))
        if not np.isfinite(b) or not (B_BOUNDS[0] < b < B_BOUNDS[1]):
            continue
        score = _deviance(counts, _bin_probabilities(edges, q, b))
        if np.isfinite(score) and score < best.objective or (not best.success and np.isfinite(score)):
            best = Fit(float(q), float(b), True, "barrido q-log; q por deviance de bins",
                       float(score), 0, int(occupied.sum()))
    return best


def fit(data, method, bins=50, q_grid=DEFAULT_Q_GRID, hist=None):
    """Ajusta q,b con mu=0. `hist` permite compartir bins entre PDF y QLOG."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("datos deben ser finitos y unidimensionales")
    if method not in {"pdf", "mle", "qlog", "cdf"}:
        raise ValueError("metodo desconocido")
    edges, counts = hist if hist is not None else histogram(x, bins)
    occupied = int(np.count_nonzero(counts))
    if method == "qlog":
        return _qlog_fit(edges, counts, q_grid)
    sorted_x = np.sort(x) if method == "cdf" else None
    empirical = (np.arange(len(x)) + .5) / len(x) if method == "cdf" else None

    def objective(params):
        q, b = params
        if method == "pdf":
            return _deviance(counts, _bin_probabilities(edges, q, b))
        if method == "cdf":
            return np.mean((q_gaussian_cdf(sorted_x, b, q) - empirical)**2)
        # log de la constante de normalizacion y log1p para estabilidad.
        log_a = (.5 * np.log((q-1)*b/np.pi) + gammaln(1/(q-1))
                 - gammaln((3-q)/(2*(q-1))))
        return -len(x)*log_a + np.log1p((q-1)*b*x*x).sum()/(q-1)

    candidates = []
    for start in STARTS:
        try:
            result = minimize(objective, start, bounds=(Q_BOUNDS, B_BOUNDS),
                              method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-11})
            if result.success and np.isfinite(result.fun):
                candidates.append(result)
        except (ValueError, FloatingPointError, OverflowError):
            continue
    if method == "pdf":
        # L-BFGS-B puede declarar convergencia en una cuenca local de la
        # deviance agrupada. Comparar con anclas fijadas antes del estudio;
        # si alguna es mejor, reiniciar desde la mejor ancla.
        anchors = ((1.4, .5), (1.4, 5.), (1.9, .5), (1.9, 5.),
                   (2.4, .5), (2.4, 5.))
        ranked = sorted((objective(anchor), anchor) for anchor in anchors)
        if not candidates or ranked[0][0] + 1e-7 < min(c.fun for c in candidates):
            for _, anchor in ranked[:2]:
                try:
                    result = minimize(objective, anchor, bounds=(Q_BOUNDS, B_BOUNDS),
                                      method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-11})
                    if result.success and np.isfinite(result.fun):
                        candidates.append(result)
                except (ValueError, FloatingPointError, OverflowError):
                    continue
    if not candidates:
        return Fit(message="optimizacion fallida", occupied_bins=occupied)
    result = min(candidates, key=lambda res: res.fun)
    q, b = (float(v) for v in result.x)
    at_bound = (min(q-Q_BOUNDS[0], Q_BOUNDS[1]-q) < 1e-4
                or min(b-B_BOUNDS[0], B_BOUNDS[1]-b) < 1e-4)
    return Fit(q, b, True, str(result.message), float(result.fun),
               int(sum(c.nfev for c in candidates)), occupied, at_bound)
