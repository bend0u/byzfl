"""Run and verify a concrete CNN clipping/measurement experiment on one GPU."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    # The repository has no package metadata, so server-side scripts executed
    # from a subdirectory must make the checkout importable explicitly.
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from byzfl.benchmark.managers import ParamsManager, get_model_result_name
from byzfl.benchmark.measurements import MeasurementRecorder
from byzfl.benchmark.train import start_training
from byzfl.fed_framework.clipping import create_clipping_method


DEFAULT_CONFIG = Path(__file__).with_name("config_gpu_smoke.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a CNN/MNIST client-clipping smoke test and validate its measurements."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        help="Override benchmark_config.device, for example cuda:0 or cuda:1.",
    )
    parser.add_argument("--steps", type=int, help="Override benchmark_config.nb_steps.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the configuration and exit without downloading data or training.",
    )
    return parser.parse_args()


def load_config(path, device=None, steps=None):
    with path.resolve().open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if device is not None:
        config["benchmark_config"]["device"] = device
    if steps is not None:
        if steps <= 0:
            raise ValueError("--steps must be positive.")
        config["benchmark_config"]["nb_steps"] = steps

    # Resolve data and results relative to the checkout, independently of the
    # directory from which this script is launched on the server.
    result_config = config["evaluation_and_results"]
    for field in ("data_folder", "results_directory"):
        path_value = Path(result_config[field])
        if not path_value.is_absolute():
            result_config[field] = str((REPOSITORY_ROOT / path_value).resolve())
    return config


def validate_config(config):
    manager = ParamsManager(config)
    if manager.get_training_algorithm_name() != "DSGD":
        raise ValueError("This smoke test requires the DSGD training algorithm.")
    if manager.is_snn():
        raise ValueError("This smoke test intentionally requires a normal ANN model.")
    create_clipping_method(manager.get_honest_clients_clipping())
    recorder = MeasurementRecorder(manager.get_measurements_config())
    if not recorder.enabled:
        raise ValueError("Measurements must be enabled for this smoke test.")
    return manager, recorder


def read_rows(path):
    if not path.is_file():
        raise RuntimeError(f"Missing expected measurement file: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise RuntimeError(f"Measurement file is empty: {path}")
    return rows


def find_experiment_directory(config, manager):
    results_directory = Path(manager.get_results_directory())
    result_name = get_model_result_name(config)
    prefix = f"{manager.get_dataset_name()}_{result_name}_"
    candidates = [
        path for path in results_directory.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    expected_config = manager.get_data()
    exact_candidates = []
    for path in candidates:
        saved_config_path = path / "config.json"
        if not saved_config_path.is_file():
            continue
        with saved_config_path.open(encoding="utf-8") as config_file:
            if json.load(config_file) == expected_config:
                exact_candidates.append(path)
    candidates = exact_candidates
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one experiment directory with the resolved smoke configuration, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def verify_measurements(config, manager, recorder):
    experiment_directory = find_experiment_directory(config, manager)
    training_seed = manager.get_training_seed()
    distribution_seed = manager.get_data_distribution_seed()
    measurement_directory = experiment_directory / (
        f"measurements_tr_seed_{training_seed}_dd_seed_{distribution_seed}"
    )

    honest_rows = read_rows(measurement_directory / "honest_gradients.csv")
    clipping_rows = read_rows(measurement_directory / "clipping.csv")
    server_rows = read_rows(measurement_directory / "server.csv")
    if not (measurement_directory / "metadata.json").is_file():
        raise RuntimeError("Missing measurement metadata.json.")

    steps = manager.get_nb_steps()
    sampled_rounds = 1 + (steps - 1) // recorder.every_n_rounds
    expected_honest = sampled_rounds * len(recorder.honest_stages)
    expected_clipping = sampled_rounds * manager.get_nb_honest_clients()
    if len(honest_rows) != expected_honest:
        raise RuntimeError(f"Expected {expected_honest} honest rows, found {len(honest_rows)}.")
    if len(clipping_rows) != expected_clipping:
        raise RuntimeError(f"Expected {expected_clipping} clipping rows, found {len(clipping_rows)}.")
    if len(server_rows) != sampled_rounds:
        raise RuntimeError(f"Expected {sampled_rounds} server rows, found {len(server_rows)}.")

    required_honest = {
        "round", "stage", "heterogeneity", "norm_mean", "norm_median", "cosine_mean"
    }
    required_server = {"round", "aggregate_norm", "cosine_with_honest_mean"}
    if not required_honest.issubset(honest_rows[0]):
        raise RuntimeError("honest_gradients.csv is missing required metric columns.")
    if not required_server.issubset(server_rows[0]):
        raise RuntimeError("server.csv is missing required metric columns.")

    for row in clipping_rows:
        threshold = float(row["threshold"])
        output_norm = float(row["output_norm"])
        if not math.isfinite(threshold) or not math.isfinite(output_norm):
            raise RuntimeError("The configured moving-average clip produced nonfinite diagnostics.")
        if int(row["clipped"]) and output_norm > threshold * (1.0 + 1e-5) + 1e-8:
            raise RuntimeError("A clipped output norm exceeds the threshold recorded for it.")

    return experiment_directory, sampled_rounds


def main():
    args = parse_args()
    config = load_config(args.config, args.device, args.steps)
    manager, recorder = validate_config(config)
    if args.validate_only:
        print(f"Configuration is valid: {args.config.resolve()}")
        print(f"Model: {manager.get_model_name()} (ANN)")
        print(f"Clipping: {manager.get_honest_clients_clipping()['name']}")
        return

    device = manager.get_device()
    if not device.startswith("cuda"):
        raise ValueError(f"GPU smoke test requires a CUDA device, got {device!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")

    print(f"Running {manager.get_model_name()} on {device} for {manager.get_nb_steps()} rounds...")
    start_training(config)
    experiment_directory, sampled_rounds = verify_measurements(config, manager, recorder)
    print(f"Smoke test passed with {sampled_rounds} measured rounds.")
    print(f"Results: {experiment_directory}")


if __name__ == "__main__":
    main()
