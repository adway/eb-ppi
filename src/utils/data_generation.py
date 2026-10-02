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


def generate_two_estimator_data(
    n1: int,
    n2: int,
    mu: float,
    delta_f: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    rng: np.random.Generator,
    return_samples: bool = False,
) -> tuple:
    """Generate paired samples for a two-estimator setting.

    The first estimator's observations have mean ``mu`` and sample size
    ``n1``; the second's have mean ``mu + delta_f`` and sample size ``n2``.
    The first ``min(n1, n2)`` observations are paired with correlation
    ``rho``. Any excess observations are generated independently. Returns the
    sample means and estimated covariance matrix. With ``return_samples=True``,
    also returns the two raw sample arrays.
    """
    if not isinstance(n1, (int, np.integer)) or not isinstance(n2, (int, np.integer)):
        raise ValueError("n1 and n2 must be integers")
    if n1 < 2 or n2 < 2:
        raise ValueError("n1 and n2 must both be at least 2")
    if (not np.isfinite(sigma1) or not np.isfinite(sigma2)
            or min(sigma1, sigma2) <= 0 or not np.isfinite(rho) or abs(rho) >= 1):
        raise ValueError("standard deviations must be positive and |rho| < 1")

    cov12 = np.array([
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2],
    ])
    n_paired = min(n1, n2)
    paired = rng.multivariate_normal([mu, mu + delta_f], cov12, size=n_paired)
    samples1 = paired[:, 0]
    samples2 = paired[:, 1]
    if n1 > n_paired:
        samples1 = np.concatenate((samples1, rng.normal(mu, sigma1, n1 - n_paired)))
    if n2 > n_paired:
        samples2 = np.concatenate((samples2, rng.normal(mu + delta_f, sigma2, n2 - n_paired)))

    y = np.array([samples1.mean(), samples2.mean()])
    covariance_of_paired_observations = np.cov(paired, rowvar=False, ddof=1)[0, 1]
    covariance_of_means = covariance_of_paired_observations * n_paired / (n1 * n2)
    cov_y = np.array([
        [np.var(samples1, ddof=1) / n1,
         covariance_of_means],
        [covariance_of_means,
         np.var(samples2, ddof=1) / n2],
    ])

    if return_samples:
        return y, cov_y, samples1, samples2
    return y, cov_y
