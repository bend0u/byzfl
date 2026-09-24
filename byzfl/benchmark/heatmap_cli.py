"""Generic command-line interface for benchmark heatmap generation."""

import argparse
import json
from pathlib import Path

from byzfl.benchmark.heatmaps import (
    CAPTION_FIELDS,
    HEATMAP_VIEWS,
    generate_heatmaps,
)


def build_parser(default_results=None):
    """Build the CLI parser, optionally with an experiment-specific default."""
    default_results = (
        Path(default_results).expanduser().resolve()
        if default_results is not None
        else None
    )
    parser = argparse.ArgumentParser(
        description=(
            "Generate robust-best, per-aggregator, and per-attack test-accuracy "
            "heatmaps from a ByzFL benchmark result directory."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results,
        required=default_results is None,
        help="Benchmark result directory containing config.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory (default: <results>/heatmaps or "
            "<results>/heatmaps_captioned)."
        ),
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=HEATMAP_VIEWS,
        default=HEATMAP_VIEWS,
        help="Heatmap products to generate (default: all three).",
    )
    parser.add_argument(
        "--caption",
        action="store_true",
        help="Add compact identifying captions and use heatmaps_captioned/.",
    )
    parser.add_argument(
        "--caption-fields",
        nargs="+",
        choices=CAPTION_FIELDS,
        default=CAPTION_FIELDS,
        help=(
            "Caption fields and display order "
            "(default: model clipping pipeline attacks)."
        ),
    )
    return parser


def validate_results(results_directory):
    """Check that training and hyperparameter selection have completed."""
    config_path = results_directory / "config.json"
    hyperparameters_path = results_directory / "best_hyperparameters"

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing benchmark configuration: {config_path}")
    if not hyperparameters_path.is_dir():
        raise FileNotFoundError(
            f"Missing selected hyperparameters: {hyperparameters_path}"
        )

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not config.get("evaluation_and_results", {}).get("evaluate_on_test", False):
        raise ValueError("The benchmark did not enable test-set evaluation.")


def main(default_results=None, argv=None):
    """Run the generic CLI, optionally using an experiment-specific default."""
    args = build_parser(default_results).parse_args(argv)
    results_directory = args.results.expanduser().resolve()
    validate_results(results_directory)

    output_directory = (
        args.output.expanduser().resolve() if args.output is not None else None
    )
    generated = generate_heatmaps(
        results_directory,
        output_directory,
        views=args.views,
        caption=args.caption,
        caption_fields=args.caption_fields,
    )

    output_root = (
        output_directory
        if output_directory is not None
        else results_directory / (
            "heatmaps_captioned" if args.caption else "heatmaps"
        )
    )
    print(f"Generated {len(generated)} heatmaps in {output_root}")
    for heatmap in generated:
        print(heatmap.relative_to(output_root))


if __name__ == "__main__":
    main()
