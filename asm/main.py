import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import ruamel.yaml as yaml

from asm.generate_data import DataModel, generate_xs
from asm.kernel import KernelFactory
from asm.local_smoothing import LocalSmoothing
from asm.risk import true_risk, unbiased_risk_estimate
from asm.utils import random_seed
from asm.vizualize import plot_data, plot_true_risk


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args


def main(config: Dict[str, Any], args: argparse.Namespace):
    if config["seed"]:
        random_seed(config["seed"])

    xs = generate_xs()
    data_model = DataModel(seed=config["seed"], sigma=config["sigma"])
    f_x = data_model.func(xs)
    ys = data_model.sample_ys(xs, config["seed"])

    Path(config["figpath"]).mkdir(exist_ok=True)
    plot_data(
        xs,
        f_x,
        ys,
        savepath=Path(config["figpath"], f"function_data_{config['name']}.png"),
    )

    x0 = np.array(np.random.rand())
    print(f"x_0: {x0:.3f}")
    grad_x0 = data_model.grad_func(x0)
    bandwidths = np.logspace(
        config["bandwidth_max"], config["bandwidth_min"], config["num_bandwidths"]
    )

    local_smother = LocalSmoothing(config["order"])
    kernel = KernelFactory.get_kernel(config["kernel"])

    risk_estimates = np.empty(bandwidths.shape[0])
    true_risks = np.empty(bandwidths.shape[0])
    for bandwidth_idx, bandwidth in enumerate(bandwidths):
        print(f"\nBandwidth: {bandwidth}")
        weights = kernel(xs - x0, bandwidth)
        local_smother.fit(x0, xs, ys, weights)
        derivative_estimate = local_smother.estimate[
            1
        ]  # TODO: change explicit index to variable

        print(f"Estimate: {derivative_estimate:.3f}, true value: {grad_x0:.3f}")

        weights = kernel(xs[:, None] - xs[None, :], bandwidth)
        kernel_matrix = local_smother.construct_kernel_matrix(xs, weights)
        risk_estimate = unbiased_risk_estimate(kernel_matrix, ys, config["sigma"])
        risk_estimates[bandwidth_idx] = risk_estimate

        print(f"Unbiased risk estimate: {risk_estimate:.3f}")

        weights = kernel(xs - x0, bandwidth)
        true_risk_ = true_risk(
            x0, xs, weights, f_x, grad_x0, config["sigma"], config["order"]
        )
        true_risks[bandwidth_idx] = true_risk_

        print(f"True risk: {true_risk_:.3f}")
    print("\n")

    best_idx = np.argmin(risk_estimates)
    best_bandwidth_emp = bandwidths[best_idx]
    best_risk_emp = risk_estimates[best_idx]

    optimal_idx = np.argmin(true_risks)
    optimal_bandwidth = bandwidths[optimal_idx]
    optimal_risk = true_risks[optimal_idx]

    print(f"Oracle bandwidth: {optimal_bandwidth:.5f}, oracle risk: {optimal_risk:.3f}")
    print(
        f"Chosen bandwidth: {best_bandwidth_emp:.5f}, risk: {true_risks[best_idx]:.3f}"
    )

    plot_true_risk(
        bandwidths,
        true_risks,
        estimate=(best_idx, best_risk_emp),
        savepath=Path(config["figpath"], f"true_risk_{config['name']}.png"),
    )


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(Path(args.config).open("r"))
    main(config, args)
