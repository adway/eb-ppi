"""Confidence interval utilities for the two-estimator setting."""

import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import norm


def _mean_variances(
    n1: int, n2: int, sigma1: float, sigma2: float, rho: float
) -> tuple[float, float, float]:
    """Return var(mean 1), var(mean 2), and their covariance.

    The samples share ``min(n1, n2)`` paired observations; any excess
    observations are independent, matching ``generate_two_estimator_data``.
    """
    if not isinstance(n1, (int, np.integer)) or not isinstance(n2, (int, np.integer)):
        raise ValueError("n1 and n2 must be integers")
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must both be at least 2")
    if not np.isfinite(sigma1) or not np.isfinite(sigma2) or min(sigma1, sigma2) <= 0:
        raise ValueError("standard deviations must be finite and positive")
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("rho must be finite and satisfy |rho| < 1")

    var1 = sigma1**2 / n1
    var2 = sigma2**2 / n2
    cov12 = rho * sigma1 * sigma2 * min(n1, n2) / (n1 * n2)
    return var1, var2, cov12


def psi_two_estimator(
    y: np.ndarray,
    n1: int,
    n2: int,
    sigma1: float,
    sigma2: float,
    rho: float,
    eps: float = 1e-8,
) -> float:
    """Compute the plug-in ``psi`` weight for two estimators."""
    y = np.asarray(y, dtype=float)
    if y.shape != (2,) or not np.all(np.isfinite(y)):
        raise ValueError("y must contain two finite sample means")
    var1, var2, cov12 = _mean_variances(n1, n2, sigma1, sigma2, rho)
    diff = y[1] - y[0]
    sigma_delta_sq_hat = max(diff**2 - var1 - var2 + 2 * cov12, 0.0)
    return (var1 - cov12) / (var1 + var2 + sigma_delta_sq_hat - 2 * cov12 + eps)


def ci_delta_two_estimator(
    y: np.ndarray,
    n1: int,
    n2: int,
    sigma1: float,
    sigma2: float,
    rho: float,
    gamma: float = 0.05,
) -> tuple[float, float]:
    """Compute a normal confidence interval for ``delta_f = E[Y2] - E[Y1]``."""
    y = np.asarray(y, dtype=float)
    if y.shape != (2,) or not np.all(np.isfinite(y)):
        raise ValueError("y must contain two finite sample means")
    if not 0 < gamma < 1:
        raise ValueError("gamma must be between 0 and 1")
    var1, var2, cov12 = _mean_variances(n1, n2, sigma1, sigma2, rho)
    se = np.sqrt(var1 + var2 - 2 * cov12)
    radius = norm.ppf(1 - gamma / 2) * se
    diff = y[1] - y[0]
    return float(diff - radius), float(diff + radius)


def ci_tau_two_estimator(
    y: np.ndarray,
    n1: int,
    n2: int,
    sigma1: float,
    sigma2: float,
    rho: float,
    gamma: float = 0.05,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute the two-estimator confidence interval for ``tau``.

    ``gamma`` controls the normal quantile used in the plug-in center, while
    ``alpha`` is the tail probability used to solve for the interval radius.
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (2,) or not np.all(np.isfinite(y)):
        raise ValueError("y must contain two finite sample means")
    if not 0 < gamma < 1 or not 0 < alpha < 1:
        raise ValueError("alpha and gamma must be between 0 and 1")
    var1, var2, cov12 = _mean_variances(n1, n2, sigma1, sigma2, rho)
    var_diff = var1 + var2 - 2 * cov12
    if var_diff <= 0:
        raise ValueError("variance of the difference must be positive")

    psi = psi_two_estimator(y, n1, n2, sigma1, sigma2, rho)
    diff = y[1] - y[0]
    mu_bar = abs(psi * diff) + var1 / np.sqrt(var_diff) * norm.ppf(1 - gamma / 2)
    sigma = np.sqrt(var1 * var2 / var_diff)

    def equation(eta: float) -> float:
        return (
            norm.cdf((eta - mu_bar) / sigma)
            - norm.cdf((-eta - mu_bar) / sigma)
            - (1 - alpha)
        )

    upper = max(abs(mu_bar) + 8 * sigma, sigma)
    while equation(upper) < 0:
        upper *= 2
    eta = root_scalar(equation, bracket=(0.0, upper), method="brentq").root
    center = y[0] + psi * diff
    return float(center - eta), float(center + eta)
