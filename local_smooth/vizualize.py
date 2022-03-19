from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import seaborn
from matplotlib import pyplot as plt


seaborn.set_theme("poster")
seaborn.set_style("dark")

SMALL_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_data(
    xs: np.ndarray,
    f_x: np.ndarray,
    ys: np.ndarray,
    x0: Optional[np.ndarray] = None,
    savepath: Optional[Union[str, Path]] = None,
):
    fig = plt.figure(figsize=(7, 5))

    plt.plot(xs, f_x, label="true function")
    plt.scatter(xs, ys, marker=".", color="r", label="data")
    if x0:
        plt.axvline(x0, linestyle='--', label=r'$x_0$', color='black')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.grid()

    plt.legend()
    fig.tight_layout()

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()


def plot_true_risk(
    bandwidths: np.ndarray,
    risks: np.ndarray,
    estimate: Optional[Tuple[int, float]] = None,
    plot_optimal=True,
    savepath: Optional[Union[str, Path]] = None,
):
    fig = plt.figure(figsize=(7, 5))

    plt.plot(bandwidths, risks, label=r"true risk $\mathcal{R}(\hat{f})$", zorder=1)
    if plot_optimal:
        optimal_idx = np.argmin(risks)
        plt.scatter(
            bandwidths[optimal_idx],
            risks[optimal_idx],
            marker="*",
            color='r',
            label=r"optimal $\mathcal{R}_{m^*}$",
            s=100,
            zorder=2
        )
    if estimate:
        # plt.scatter(bandwidths[estimate[0]], estimate[1], marker='x', label=r'estimate $\hat{\mathcal{R}_{m^*}$')
        plt.scatter(
            bandwidths[estimate[0]],
            risks[estimate[0]],
            marker="x",
            color='g',
            label=r"estimate $\mathcal{R}_{\hat{m}}$",
            s=100,
            zorder=2
        )

    plt.xscale("log")
    plt.xlabel(r"bandwidth")
    plt.ylabel(r"Risk")
    plt.grid()
    plt.legend()
    fig.tight_layout()

    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
