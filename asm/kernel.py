import numpy as np


def gaussian_kernel(x: np.ndarray, h) -> np.ndarray:
    return np.exp(-(x ** 2) / (2 * h))
