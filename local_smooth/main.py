import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import ruamel.yaml as yaml

from local_smooth.generate_data import DataModel, generate_xs
from local_smooth.kernel import KernelFactory
from local_smooth.local_smoothing import LocalSmoothing
from local_smooth.risk import true_risk, unbiased_risk_estimate
from local_smooth.utils import random_seed
from local_smooth.vizualize import plot_data, plot_true_risk


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    return args


def main(config: Dict[str, Any], args: argparse.Namespace):
    print('Config: ')
    print(yaml.safe_dump(config, default_flow_style=False))
    
    if config["seed"]:
        random_seed(config["seed"])

    xs = generate_xs(config["sample_size"])
    data_model = DataModel(
        n_coef=config["n_coef"], seed=config["seed"], sigma=config["sigma"]
    )
    f_x = data_model.func(xs)
    ys = data_model.sample_ys(xs, config["seed"])

    x0 = np.array(np.random.rand())
    print(f"x_0: {x0:.3f}")

    Path(config["figpath"]).mkdir(exist_ok=True)
    plot_data(
        xs,
        f_x,
        ys,
        x0=x0,
        savepath=Path(config["figpath"], f"function_data_{config['name']}"),
    )

    grad_x0 = data_model.grad_func(x0)
    f_x0 = data_model.func(x0)
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

        print(f"Function estimate: {local_smother.estimate[0]:.3f}, true value: {f_x0:.3f}")
        print(f"Derivative estimate: {local_smother.estimate[1]:.3f}, true value: {grad_x0:.3f}")

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

    # plot_true_risk(
    #     bandwidths,
    #     risk_estimates,
    #     #estimate=(best_idx, best_risk_emp),
    #     savepath=Path(config["figpath"], f"unbiased_risk_{config['name']}.png"),
    # )
    
    plot_true_risk(
        bandwidths,
        true_risks,
        estimate=(best_idx, best_risk_emp),
        savepath=Path(config["figpath"], f"true_risk_{config['name']}"),
    )

    print('='*20)


if __name__ == "__main__":
    args = parse_arguments()
    config = yaml.safe_load(Path(args.config).open("r"))
    main(config, args)
