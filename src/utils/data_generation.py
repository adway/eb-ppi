"""Data-generating process for the EB/PPI simulation study."""

from __future__ import annotations

import numpy as np


def generate_data(n1: int, n2: int, mu: float, delta_f: float, delta_c: float,
                  sigma1: float, sigma2: float, sigma3: float, rho: float,
                  rng: np.random.Generator,
                  return_samples: bool = False) -> tuple:
    """Generate sample means and covariance, optionally retaining raw samples."""
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must be at least 2")
    if min(sigma1, sigma2, sigma3) <= 0 or abs(rho) >= 1:
        raise ValueError("standard deviations must be positive and |rho| < 1")
    cov12 = np.array([[sigma1**2, rho * sigma1 * sigma2],
                      [rho * sigma1 * sigma2, sigma2**2]])
    y12 = rng.multivariate_normal([mu, mu + delta_f], cov12, size=n1)
    y3 = rng.normal(mu + delta_c, sigma3, size=n2)
    y = np.array([y12[:, 0].mean(), y12[:, 1].mean(), y3.mean()])
    s12 = np.atleast_2d(np.cov(y12, rowvar=False, ddof=1))
    s3 = float(np.var(y3, ddof=1))
    cov_y = np.array([[s12[0, 0] / n1, s12[0, 1] / n1, 0.0],
                      [s12[1, 0] / n1, s12[1, 1] / n1, 0.0],
                      [0.0, 0.0, s3 / n2]])
    if return_samples:
        return y, cov_y, y12, y3
    return y, cov_y
