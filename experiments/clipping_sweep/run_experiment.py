"""Launch the CNN/MNIST clipping sweep on exactly two visible GPUs."""

import argparse
import json
import os
from pathlib import Path
import sys


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIRECTORY / "config.json"
REPOSITORY_ROOT = EXPERIMENT_DIRECTORY.parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the CNN/MNIST constant-10 versus no-clipping sweep on two GPUs."
        )
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=16,
        help="Concurrent training processes across both GPUs (default: 16).",
    )
    parser.add_argument(
        "--gpus",
        default="0,1",
        help="Two physical GPU IDs to expose to the sweep (default: 0,1).",
    )
    return parser.parse_args()


def validate_launcher_inputs(jobs, gpu_ids):
    if jobs <= 0:
        raise ValueError("--jobs must be a positive integer.")

    parsed_gpu_ids = [gpu_id.strip() for gpu_id in gpu_ids.split(",") if gpu_id.strip()]
    if len(parsed_gpu_ids) != 2 or len(set(parsed_gpu_ids)) != 2:
        raise ValueError("--gpus must contain exactly two distinct GPU IDs.")
    return ",".join(parsed_gpu_ids)


def validate_config():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if config["benchmark_config"].get("device") != "cuda":
        raise ValueError("benchmark_config.device must be 'cuda' for GPU distribution.")

    expected_clipping = [
        {"name": "constant", "parameters": {"max_norm": 10.0}},
        {"name": "none", "parameters": {}},
    ]
    if config["honest_clients"].get("clipping") != expected_clipping:
        raise ValueError(
            "This launcher requires exactly constant(max_norm=10.0) and none clipping."
        )

    measurements = config.get("measurements", {})
    if measurements.get("enabled", False):
        raise ValueError("Measurements must stay disabled for this first sweep.")


def main():
    args = parse_args()
    visible_gpus = validate_launcher_inputs(args.jobs, args.gpus)
    validate_config()

    # Set visibility before importing PyTorch or ByzFL. PyTorch will expose the
    # selected physical devices locally as cuda:0 and cuda:1.
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    import torch
    from byzfl.benchmark import run_benchmark

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(
            f"Expected exactly two visible GPUs, found {torch.cuda.device_count()}."
        )

    print(f"Configuration: {CONFIG_PATH}")
    print(f"Physical GPUs: {visible_gpus}")
    print(f"Concurrent trainings: {args.jobs}")

    previous_directory = Path.cwd()
    try:
        os.chdir(EXPERIMENT_DIRECTORY)
        run_benchmark(nb_jobs=args.jobs, distribute_gpus=True)
    finally:
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
