from typing import Callable

import numpy as np


class KernelFactory:
    @classmethod
    def get_kernel(cls, name: str) -> Callable:
        if name == "gaussian":
            return gaussian_kernel
        elif name == "rectangular":
            return rectangular_kernel
        elif name == "epanechnikov":
            return epanichnikov_kernel
        else:
            raise KeyError


def gaussian_kernel(x: np.ndarray, h) -> np.ndarray:
    return np.exp(-(x ** 2) / (2 * h))


def rectangular_kernel(x: np.ndarray, h) -> np.ndarray:
    return (1.0 / h * np.abs(x) < 1).astype(float)


def epanichnikov_kernel(x: np.ndarray, h) -> np.ndarray:
    return np.clip(1 - 1.0 / h * x ** 2, 0)
