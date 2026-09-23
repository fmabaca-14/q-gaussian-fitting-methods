"""
Probability distributions used in the q-Gaussian fitting project.

This module contains the normalized q-Gaussian probability density
function, its cumulative distribution function, and related utilities.
"""
import numpy as np
from scipy.special import gammaln
from scipy.stats import t


def q_gaussian_norm(b, q):
    """
    Calculate the normalization constant of a q-Gaussian.

    Parameters
    ----------
    b : float
        Scale parameter. Must satisfy b > 0.

    q : float
        Entropic index. Must satisfy 1 < q < 3.

    Returns
    -------
    float
        Normalization constant a(b, q).
    """

    if b <= 0:
        raise ValueError("b must be positive.")

    if not (1 < q < 3):
        raise ValueError("q must satisfy 1 < q < 3.")

    log_a = (0.5 * np.log((q - 1) * b / np.pi) + gammaln(1 / (q - 1)) - gammaln((3 - q)/ (2 * (q - 1))))

    return np.exp(log_a)


def q_gaussian(x, b, q, mu=0.0):
    """
    Normalized q-Gaussian probability density function.

    Parameters
    ----------
    x : array_like
        Points where the PDF is evaluated.

    b : float
        Scale parameter.

    q : float
        Entropic index.

    mu : float, optional
        Location parameter. Default is 0.

    Returns
    -------
    ndarray
        q-Gaussian PDF evaluated at x.
    """

    x = np.asarray(x, dtype=float)

    a = q_gaussian_norm(b, q)

    return (a* (1+ (q - 1)* b* (x - mu) ** 2)** (-1 / (q - 1)))


def q_gaussian_cdf(x, b, q, mu=0.0):
    """
    Cumulative distribution function of the normalized q-Gaussian.

    The q-Gaussian is mapped exactly onto a Student-t distribution.

    Parameters
    ----------
    x : array_like
        Points where the CDF is evaluated.

    b : float
        Scale parameter.

    q : float
        Entropic index.

    mu : float, optional
        Location parameter.

    Returns
    -------
    ndarray
        q-Gaussian CDF evaluated at x.
    """

    if b <= 0:
        raise ValueError("b must be positive.")

    if not (1 < q < 3):
        raise ValueError("q must satisfy 1 < q < 3.")

    x = np.asarray(x, dtype=float)

    nu = ((3 - q)/ (q - 1))

    scale = (1/ np.sqrt( b * (3 - q)))

    return t.cdf((x - mu) / scale,df=nu)


def q_log(y, q):
    """
    q-logarithm.

    Parameters
    ----------
    y : array_like
        Positive input values.

    q : float
        Entropic index.

    Returns
    -------
    ndarray
        q-logarithm evaluated at y.
    """

    y = np.asarray(y, dtype=float)

    if np.any(y <= 0):
        raise ValueError(
            "q-logarithm requires y > 0."
        )

    return (y ** (1 - q) - 1) / (1 - q)