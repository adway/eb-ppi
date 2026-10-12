"""Estimators used by the EB/PPI simulation study.

The data reduction convention is ``y = (mean(Y1), mean(Y2), mean(Y3))`` and
``cov_y`` is the estimated covariance matrix of these three means.
"""

from __future__ import annotations

import numpy as np


M = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])


def _inputs(y: np.ndarray, cov_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    cov_y = np.asarray(cov_y, dtype=float)
    if y.shape != (3,):
        raise ValueError(f"y must have shape (3,), got {y.shape}")
    if cov_y.shape != (3, 3) or not np.all(np.isfinite(cov_y)):
        raise ValueError("cov_y must be a finite 3 by 3 matrix")
    return y, (cov_y + cov_y.T) / 2


def eb(
    y: np.ndarray,
    n1: int,
    n2: int,
    sigma1: float,
    sigma2: float,
    sigma3: float,
    rho: float,
    eps: float = 1e-10,
) -> tuple[float, float]:
    """Return the empirical-Bayes point estimate.

    The nuisance standard deviations and correlation are treated as known,
    matching the notebook. The two contrast variances are estimated from the
    observed contrasts and clipped at ``eps`` for numerical stability.
    """
    y, _ = _inputs(y, np.eye(3))
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must both be at least 2")
    if min(sigma1, sigma2, sigma3) <= 0 or abs(rho) >= 1:
        raise ValueError("standard deviations must be positive and |rho| < 1")
    sigma = np.array([
        [sigma1**2 / n1, rho * sigma1 * sigma2 / n1, 0.0],
        [rho * sigma1 * sigma2 / n1, sigma2**2 / n1, 0.0],
        [0.0, 0.0, sigma3**2 / n2],
    ])
    contrast_f = max((y[1] - y[0]) ** 2 - sigma[0, 0] - sigma[1, 1] + 2 * sigma[0, 1], eps)
    contrast_c = max((y[2] - y[0]) ** 2 - sigma[0, 0] - sigma[2, 2], eps)
    precision = np.diag([0.0, 1.0 / contrast_f, 1.0 / contrast_c])
    sigma_inv = np.linalg.inv(sigma)
    v_post = np.linalg.inv(M.T @ sigma_inv @ M + precision)
    theta_post = v_post @ M.T @ sigma_inv @ y
    return float(theta_post[0])


def ppi(y: np.ndarray, lam: float = 1.0) -> float:
    """The plug-in PPI estimator ``Y1 - Y2 + Y3`` (or arbitrary lambda)."""
    y, _ = _inputs(y, np.eye(3))
    return float(y[0] + lam * (y[2] - y[1]))


def ppi_plus_plus(
    y: np.ndarray,
    n1: int,
    n2: int,
    cov_y: np.ndarray,
    lam: float | None = None,
) -> float:

    y, cov_y = _inputs(y, cov_y)

    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must both be at least 2")

    f_bar = (n1 * y[1] + n2 * y[2]) / (n1 + n2)

    # Recover observation-level variances
    var_f1 = n1 * cov_y[1, 1]
    var_f2 = n2 * cov_y[2, 2]

    # Pooled variance of all n1 + n2 predictions
    var_f = (
        (n1 - 1) * var_f1
        + (n2 - 1) * var_f2
        + n1 * (y[1] - f_bar) ** 2
        + n2 * (y[2] - f_bar) ** 2
    ) / (n1 + n2 - 1)

    var_f = max(float(var_f), np.finfo(float).eps)

    if lam is None:
        # Recover observation-level Cov(Y, f(X))
        cov_yf = n1 * cov_y[0, 1]

        lam = float(
            cov_yf / ((1 + n1 / n2) * var_f)
        )

    return ppi(y, lam)


def ppi_plus_plus_lrt(
    y: np.ndarray, n1: int, n2: int, cov_y: np.ndarray,
    y2: np.ndarray, y3: np.ndarray, mu: float, delta_f: float, delta_c: float,
    sigma2: float, sigma3: float,
) -> float:
    """LRT-weighted PPI++ for prediction samples with different means.

    Each third-component observation is weighted by the normal likelihood
    ratio ``p_2(x) / p_3(x)`` using the true population means ``mu + delta_f``
    and ``mu + delta_c`` and true population variances ``sigma2**2`` and
    ``sigma3**2``. The weighted third mean and variance are then used in the
    usual PPI++ pooled-variance lambda.
    """
    y, cov_y = _inputs(y, cov_y)
    y2, y3 = np.asarray(y2, dtype=float), np.asarray(y3, dtype=float)
    if y2.ndim != 1 or len(y2) != n1 or y3.ndim != 1 or len(y3) != n2:
        raise ValueError("y2 and y3 must be one-dimensional samples of lengths n1 and n2")
    if not np.all(np.isfinite(y2)) or not np.all(np.isfinite(y3)):
        raise ValueError("y2 and y3 must contain only finite values")
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must both be at least 2")
    if sigma2 <= 0 or sigma3 <= 0:
        raise ValueError("the second and third prediction standard deviations must be positive")
    var2, var3 = float(sigma2**2), float(sigma3**2)

    log_lrt = (
        -0.5 * (
            (y3 - (mu + delta_f)) ** 2 / var2
            - (y3 - (mu + delta_c)) ** 2 / var3
        )
        - 0.5 * np.log(var2 / var3)
    )
    weights = np.exp(log_lrt - np.max(log_lrt))
    weight_sum = float(weights.sum())
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("could not form finite LRT weights")
    weighted_mean3 = float(np.dot(weights, y3) / weight_sum)
    weights *= n2 / weight_sum
    weighted_sum = float(weights.sum())
    effective_denom = weighted_sum - float(np.dot(weights, weights) / weighted_sum)
    if effective_denom <= 0:
        raise ValueError("LRT weights have insufficient effective sample size")
    weighted_var3 = float(np.dot(weights, (y3 - weighted_mean3) ** 2) / effective_denom)

    f_bar = (n1 * y[1] + n2 * weighted_mean3) / (n1 + n2)
    var_f = (
        (n1 - 1) * var2 + (n2 - 1) * weighted_var3
        + n1 * (y[1] - f_bar) ** 2 + n2 * (weighted_mean3 - f_bar) ** 2
    ) / (n1 + n2 - 1)
    var_f = max(float(var_f), np.finfo(float).eps)
    lam = (n1 * cov_y[0, 1]) / ((1 + n1 / n2) * var_f)
    return float(y[0] + lam * (weighted_mean3 - y[1]))


def ppi_plus_plus_oracle(
    y: np.ndarray,
    n1: int,
    n2: int,
    sigma1: float,
    sigma2: float,
    sigma3: float,
    rho: float,
    delta_f: float = 0.0,
    delta_c: float = 0.0,
) -> float:
    """PPI++ using the population covariance and pooled prediction variance.

    This is the oracle counterpart of :func:`ppi_plus_plus`: it replaces the
    sample covariance entries and pooled variance with their data-generating
    values. The pooled variance includes the difference between the two
    prediction means, matching the finite-sample pooled-variance target used
    by ``ppi_plus_plus``.
    """
    y, _ = _inputs(y, np.eye(3))
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must both be at least 2")
    if min(sigma1, sigma2, sigma3) <= 0 or abs(rho) >= 1:
        raise ValueError("standard deviations must be positive and |rho| < 1")

    f_bar = (n1 * delta_f + n2 * delta_c) / (n1 + n2)
    var_f = (
        (n1 - 1) * sigma2**2
        + (n2 - 1) * sigma3**2
        + n1 * (delta_f - f_bar) ** 2
        + n2 * (delta_c - f_bar) ** 2
    ) / (n1 + n2 - 1)
    cov_yf = rho * sigma1 * sigma2
    lam = cov_yf / ((1 + n1 / n2) * var_f)
    return ppi(y, float(lam))

