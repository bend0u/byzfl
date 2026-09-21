"""Run the nine-configuration SNN MNIST GPU smoke benchmark."""

import json
import multiprocessing as mp
import os
import shutil
import subprocess
from pathlib import Path

# Keep nine worker processes from oversubscribing the host CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
from torchvision import datasets

from byzfl.benchmark import aggregated_test_heatmap, run_benchmark, test_heatmap


RUN_DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = RUN_DIRECTORY / "config.json"
NB_JOBS = 9


def _load_and_validate_config():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    benchmark = config["benchmark_config"]
    nb_configurations = (
        len(benchmark["f"])
        * len(benchmark["data_distribution"]["distribution_parameter"])
        * benchmark["nb_training_seeds"]
        * benchmark["nb_data_distribution_seeds"]
    )
    if nb_configurations != NB_JOBS:
        raise ValueError(
            f"Expected {NB_JOBS} training configurations, found {nb_configurations}."
        )
    return config


def _print_gpu_status():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this SSH session.")
    print(f"CUDA devices visible to PyTorch: {torch.cuda.device_count()}")
    print(f"Training device: {torch.cuda.get_device_name(0)}")
    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)


def _download_mnist(data_directory):
    """Download once before the worker pool starts."""
    datasets.MNIST(root=data_directory, train=True, download=True)
    datasets.MNIST(root=data_directory, train=False, download=True)


def main():
    config = _load_and_validate_config()
    os.chdir(RUN_DIRECTORY)
    _print_gpu_status()
    _download_mnist(config["evaluation_and_results"]["data_folder"])

    print(f"Launching {NB_JOBS} SNN trainings with {NB_JOBS} parallel workers.")
    run_benchmark(nb_jobs=NB_JOBS)

    results_directory = config["evaluation_and_results"]["results_directory"]
    plots_directory = "./plots"
    test_heatmap(results_directory, plots_directory)
    aggregated_test_heatmap(results_directory, plots_directory)
    print(f"Results: {RUN_DIRECTORY / 'results'}")
    print(f"Heatmaps: {RUN_DIRECTORY / 'plots'}")


if __name__ == "__main__":
    # CUDA subprocesses require a clean interpreter instead of Linux fork state.
    mp.set_start_method("spawn", force=True)
    main()
