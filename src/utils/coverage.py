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
    var_diff = var1 + var2 - 2 * cov12

    sigma_delta_sq_hat = max(
        diff**2 - var_diff,
        0.0,
    )

    return (var1 - cov12) / (
        var_diff + sigma_delta_sq_hat + eps
    )


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

    var_diff = var1 + var2 - 2 * cov12
    se_diff = np.sqrt(var_diff)

    radius = norm.ppf(1 - gamma / 2) * se_diff
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

    ``gamma`` determines the confidence set for the unknown bias ``delta``,
    while ``alpha`` controls the conditional coverage of the confidence
    interval for ``tau``.
    """
    y = np.asarray(y, dtype=float)
    if y.shape != (2,) or not np.all(np.isfinite(y)):
        raise ValueError("y must contain two finite sample means")
    if not 0 < gamma < 1 or not 0 < alpha < 1:
        raise ValueError("alpha and gamma must be between 0 and 1")

    var1, var2, cov12 = _mean_variances(
        n1, n2, sigma1, sigma2, rho
    )

    # Variance of D = tau_hat_1 - tau_hat_0.
    var_diff = var1 + var2 - 2 * cov12
    if var_diff <= 0:
        raise ValueError("variance of the difference must be positive")

    diff = y[1] - y[0]

    # Shrinkage weight.
    psi = psi_two_estimator(
        y, n1, n2, sigma1, sigma2, rho
    )

    # Conditional regression coefficient:
    #
    # E[tau_hat_0 - tau | D = d]
    #     = -(var1 - cov12) / var_diff * (d - delta).
    regression_coef = (var1 - cov12) / var_diff

    # Worst-case absolute conditional mean over the
    # (1 - gamma) confidence interval for delta.
    z_gamma = norm.ppf(1 - gamma / 2)

    mu_bar = (
        abs(psi * diff)
        + abs(var1 - cov12) / np.sqrt(var_diff) * z_gamma
    )

    # Conditional variance:
    #
    # Var(tau_hat_0 | D)
    #     = var1 - (var1 - cov12)^2 / var_diff.
    conditional_var = (
        var1
        - (var1 - cov12) ** 2 / var_diff
    )

    if conditional_var <= 0:
        raise ValueError("conditional variance must be positive")

    conditional_sd = np.sqrt(conditional_var)

    # Find eta such that the worst-case conditional coverage is 1 - alpha:
    #
    # P(|N(mu_bar, conditional_var)| <= eta) = 1 - alpha.
    def equation(eta: float) -> float:
        return (
            norm.cdf((eta - mu_bar) / conditional_sd)
            - norm.cdf((-eta - mu_bar) / conditional_sd)
            - (1 - alpha)
        )

    upper = max(
        abs(mu_bar) + 8 * conditional_sd,
        conditional_sd,
    )

    while equation(upper) < 0:
        upper *= 2

    eta = root_scalar(
        equation,
        bracket=(0.0, upper),
        method="brentq",
    ).root

    # Shrinkage estimator.
    center = y[0] + psi * diff

    return float(center - eta), float(center + eta)