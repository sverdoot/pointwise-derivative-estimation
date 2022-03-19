import numpy as np


def unbiased_risk_estimate(
    kernel_matrix: np.ndarray, ys: np.ndarray, sigma: float
) -> float:
    """Compute unbiased risk estimate for linear model with homogeneous noise."""
    bias = np.linalg.norm(kernel_matrix @ ys - ys, axis=-1, ord=2)
    variance = 2 * sigma ** 2 * np.trace(kernel_matrix)

    return bias ** 2 + variance


def compute_det(point, xs: np.ndarray, weights: np.ndarray):
    prod = weights * (xs - point)
    return weights.sum() * (prod * (xs - point)).sum() - (prod).sum() ** 2


def compute_transform(point, xs: np.ndarray, weights: np.ndarray):
    det = compute_det(point, xs, weights)
    prod = weights * (xs - point)
    return 1.0 / det * (-prod.sum() * weights + prod * weights.sum())


def true_risk(
    point,
    xs: np.ndarray,
    weights: np.ndarray,
    f_x: np.ndarray,
    true_grad: np.ndarray,
    sigma: float,
    order: int = 1,
) -> float:
    """Computes true risk given function obseretions and true gradient value.

    $\mathcal{R}(\hat{f}) = \mathbb{E}((f^*)'(x_0) - \hat{f}'(x_0))^2$

    """
    if order == 1:  # compute explicitly
        transform = compute_transform(point, xs, weights)
    else:
        psi = (xs[None, :] - point) ** np.arange(order + 1)[:, None]
        transform = (
            np.linalg.inv(psi @ (weights[None, :] * psi).T) @ (weights[None, :] * psi)
        )[
            1
        ]  # TODO: change explicit index to variable
    return (true_grad - transform.dot(f_x)) ** 2 + sigma ** 2 * np.linalg.norm(
        transform
    ) ** 2
