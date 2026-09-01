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


def ppi_plus_plus(y: np.ndarray, n1: int, n2: int, cov_y: np.ndarray, lam: float | None = None) -> float:
    """Return the PPI++ point estimate.

    ``lam=None`` selects the estimated variance-minimizing lambda. The
    The covariance estimate is used only to select the variance-minimizing
    lambda; uncertainty estimates are intentionally outside this API.
    """
    y, cov_y = _inputs(y, cov_y)
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must both be at least 2")
    f_bar = (n1 * y[1] + n2 * y[2]) / (n1 + n2)
    var_f = ((n1 - 1) * cov_y[1, 1] + (n2 - 1) * cov_y[2, 2]
             + n1 * (y[1] - f_bar) ** 2 + n2 * (y[2] - f_bar) ** 2) / (n1 + n2 - 1)
    var_f = max(float(var_f), np.finfo(float).eps)
    if lam is None:
        lam = float((n1 * cov_y[0, 1]) / ((1 + n1 / n2) * var_f))
    estimate = ppi(y, lam)
    # Conditional plug-in variance of Y1 + lam(Y3 - Y2).
    return estimate
