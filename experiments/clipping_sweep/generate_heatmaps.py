"""Generate heatmaps for this experiment using the generic ByzFL CLI."""

from pathlib import Path
import sys


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIRECTORY.parents[1]
DEFAULT_RESULTS = (
    REPOSITORY_ROOT
    / "results"
    / "clipping_sweep"
    / "cnn_mnist_constant10_vs_none"
)

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from byzfl.benchmark.heatmap_cli import main  # noqa: E402


if __name__ == "__main__":
    main(default_results=DEFAULT_RESULTS)
