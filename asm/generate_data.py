from typing import Optional, Tuple

import numpy as np

from .utils import random_seed


def generate_xs(sample_size):
    return np.linspace(0, 1, sample_size + 1)[1:]


class DataModel:
    def __init__(self, n_coef, seed: Optional[int] = None, sigma: float = 1.0):
        self.n_coef = n_coef
        self.seed = seed
        if self.seed:
            random_seed(self.seed)
        self.coefs = self.sample_coefs(self.n_coef)
        self.sigma = sigma

    @staticmethod
    def sample_coefs(n_coef) -> np.ndarray:
        gammas = np.random.randn(n_coef)
        cs = np.empty(n_coef)
        cs[:10] = gammas[:10]
        js = np.arange(11, n_coef + 1)
        cs[10:] = gammas[10:] / (js - 10) ** 2

        return cs

    @staticmethod
    def fourier_basis(xs: np.ndarray, n_coef) -> np.ndarray:
        evens = np.arange(2, n_coef + 1, 2)
        odds = np.arange(1, n_coef + 1, 2)
        psi = np.empty(list(xs.shape) + [n_coef])
        psi[..., evens - 1] = np.cos(np.pi * xs[..., None] * evens[None, :])
        psi[..., odds - 1] = np.sin(np.pi * xs[..., None] * (odds[None, :] + 1))

        return psi

    @staticmethod
    def grad_fourier_basis(xs: np.ndarray, n_coef) -> np.ndarray:
        evens = np.arange(2, n_coef + 1, 2)
        odds = np.arange(1, n_coef + 1, 2)
        psi = np.empty(list(xs.shape) + [n_coef])
        psi[..., evens - 1] = (
            -np.sin(np.pi * xs[..., None] * evens[None, :]) * np.pi * evens[..., :]
        )
        psi[..., odds - 1] = (
            np.cos(np.pi * xs[..., None] * (odds[None, :] + 1)) * np.pi * odds[..., :]
        )

        return psi

    def func(self, xs: np.ndarray):
        psi = self.fourier_basis(xs, self.n_coef)
        return psi @ self.coefs

    def grad_func(self, xs: np.ndarray):
        psi = self.grad_fourier_basis(xs, self.n_coef)
        return psi @ self.coefs

    def sample_ys(self, xs: np.ndarray, seed: Optional[int] = None):
        if seed:
            random_seed(seed)
        return self.func(xs) + np.random.randn(len(xs)) * self.sigma
