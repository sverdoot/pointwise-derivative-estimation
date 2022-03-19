import numpy as np


class LocalSmoothing:
    """Class for performing local smoothing estimation."""

    def __init__(self, order: int = 1):
        self._estimate = None
        self.order = order

    def _construct_feature(self, point, xs: np.ndarray):
        return (xs[None, :] - point) ** np.arange(self.order + 1)[:, None]

    def fit(self, point, xs: np.ndarray, ys: np.ndarray, weights: np.ndarray):
        assert xs.shape[0] == ys.shape[0]
        assert xs.shape[0] == weights.shape[0]
        psi = self._construct_feature(point, xs)  # (m + 1) x n

        lse_poly_coef = (
            np.linalg.inv(psi @ (weights[None, :] * psi).T)
            @ (weights[None, :] * psi)
            @ ys
        )
        self._estimate = lse_poly_coef

    @property
    def estimate(self) -> np.ndarray:
        return self._estimate

    def construct_kernel_matrix(self, xs, weights):
        K = np.empty((xs.shape[0], xs.shape[0]))
        for row_idx, point in enumerate(xs):
            psi = self._construct_feature(point, xs)  # (m + 1) x n
            K[row_idx] = (
                np.linalg.inv(psi @ (weights[None, row_idx, :] * psi).T)
                @ (weights[None, row_idx, :] * psi)
            )[0]
        return K
