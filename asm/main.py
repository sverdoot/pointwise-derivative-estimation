import argparse
from pathlib import Path

import numpy as np
import ruamel.yaml as yaml
from asm.generate_data import DataModel, generate_xs
from asm.kernel import gaussian_kernel
from asm.local_smoothing import LocalSmoothing
from asm.risk import true_risk, unbiased_risk_estimate
from asm.utils import random_seed
from asm.vizualize import plot_data, plot_true_risk


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args


def main(config, args: argparse.Namespace):
    if config["seed"]:
        random_seed(config["seed"])

    xs = generate_xs()
    data_model = DataModel(seed=config["seed"], sigma=config["sigma"])
    f_x = data_model.func(xs)
    ys = data_model.sample_ys(xs, config["seed"])

    Path(config["figpath"]).mkdir(exist_ok=True)
    plot_data(xs, f_x, ys, savepath=Path(config["figpath"], "function_data.png"))

    x0 = np.array(np.random.rand())
    print(f"x_0: {x0:.3f}")
    grad_x0 = data_model.grad_func(x0)
    bandwidths = np.logspace(
        config["bandwidth_max"], config["bandwidth_min"], config["num_bandwidths"]
    )

    local_smother = LocalSmoothing(config["order"])

    risk_estimates = np.empty(bandwidths.shape[0])
    true_risks = np.empty(bandwidths.shape[0])
    for bandwidth_idx, bandwidth in enumerate(bandwidths):
        print(f"\nBandwidth: {bandwidth}")
        weights = gaussian_kernel(xs - x0, bandwidth)
        local_smother.fit(x0, xs, ys, weights)
        derivative_estimate = local_smother.estimate[1]

        print(f"Estimate: {derivative_estimate:.3f}, true value: {grad_x0}")

        weights = gaussian_kernel(xs[:, None] - xs[None, :], bandwidth)
        kernel_matrix = local_smother.construct_kernel_matrix(xs, weights)
        risk_estimate = unbiased_risk_estimate(kernel_matrix, ys, sigma=config["sigma"])
        risk_estimates[bandwidth_idx] = risk_estimate

        print(f"Risk estimate: {risk_estimate:.3f}")

        weights = gaussian_kernel(xs - x0, bandwidth)
        true_risk_ = true_risk(x0, xs, weights, f_x, grad_x0, config["sigma"])
        true_risks[bandwidth_idx] = true_risk_

        print(f"True risk: {true_risk_:.3f}")

    best_idx = np.argmin(risk_estimates)
    best_bandwidth_emp = bandwidths[best_idx]
    best_risk_emp = risk_estimates[best_idx]

    plot_true_risk(
        bandwidths,
        true_risks,
        estimate=(best_idx, best_risk_emp),
        savepath=Path(config["figpath"], "true_risk.png"),
    )


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(Path(args.config).open("r"))
    main(config, args)
