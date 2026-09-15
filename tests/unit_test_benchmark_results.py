"""Focused tests for validation-selected test scores in original ByzFL results."""

import importlib
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
results = importlib.import_module("byzfl.benchmark.evaluate_results")


def write_curve(directory, kind, values, training_seed=3, dd_seed=7):
    path = directory / f"{kind}_accuracy_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt"
    np.savetxt(path, [values], delimiter=",")
    return path


@pytest.fixture
def experiment(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({
        "benchmark_config": {"nb_steps": 10},
        "evaluation_and_results": {"evaluation_delta": 4},
    }))
    return directory


def selected_accuracy(directory, training_seeds=1, dd_seeds=1):
    return results.get_accuracy_at_best_step(
        directory.parent, directory.name, dd_seeds, training_seeds, 3, 7, 10, 4
    )


@pytest.mark.parametrize("validation,test,expected", [
    ([0.1, 0.9, 0.5, 0.8], [0.95, 0.2, 0.99, 0.3], 0.2),
    ([0.1, 0.9, 0.5, 0.8], [0.95, np.nan, 0.99, 0.3], 0.3),
    ([0.1, 0.9, 0.5, 0.8], [0.95, np.inf, 0.99, 0.3], 0.3),
    ([0.1, 0.9, 0.5, 0.8], [0.95, -0.1, 0.99, 0.3], 0.3),
    ([0.1, 0.9, 0.5, 0.8], [0.95, 1.1, 0.99, 0.3], 0.3),
    ([0.1, 0.9, 0.5, 0.8], [0.95, 0.0, 0.99, 0.3], 0.0),
    ([0.1, 0.9, 0.9, 0.8], [0.95, 0.2, 0.99, 0.3], 0.2),
    ([0.1, 0.9, 0.9, 0.8], [0.95, np.nan, 0.7, 0.3], 0.7),
    ([0.1, np.nan, 0.5, 0.8], [0.95, 0.99, 0.7, 0.3], 0.3),
    ([0.1, 0.8, 0.5, 0.9], [0.95, 0.99, 0.7, 0.3], 0.3),
])
def test_validation_ranking_and_fallback(experiment, validation, test, expected):
    write_curve(experiment, "val", validation)
    write_curve(experiment, "test", test)
    assert selected_accuracy(experiment) == pytest.approx(expected)


def test_rank_after_averaging_all_training_and_distribution_seeds(experiment):
    for dd_seed in (7, 8):
        write_curve(experiment, "val", [0.0, 0.9, 0.7, 0.1], 3, dd_seed)
        write_curve(experiment, "val", [0.0, 0.3, 0.8, 0.1], 4, dd_seed)
        write_curve(experiment, "test", [0.0, 0.1, 0.2, 0.3], 3, dd_seed)
        write_curve(experiment, "test", [0.0, 0.4, 0.8, 0.9], 4, dd_seed)
    # Mean validation selects round 8 for everybody, not each seed's own peak.
    assert selected_accuracy(experiment, 2, 2) == pytest.approx(0.5)


def test_fallback_requires_test_measurement_from_every_seed(experiment):
    for training_seed in (3, 4):
        write_curve(experiment, "val", [0.1, 0.9, 0.8, 0.2], training_seed)
        write_curve(experiment, "test", [0.9, 0.1, 0.6, 0.3], training_seed)
    write_curve(experiment, "test", [0.9, np.nan, 0.6, 0.3], 4)
    assert selected_accuracy(experiment, 2) == pytest.approx(0.6)


@pytest.mark.parametrize("kind", ["val", "test"])
def test_missing_seed_is_unavailable_not_zero_or_another_seeds_curve(experiment, kind):
    for training_seed in (3, 4):
        for metric in ("val", "test"):
            path = write_curve(experiment, metric, [0.1, 0.9, 0.8, 0.2], training_seed)
            if metric == kind and training_seed == 4:
                path.unlink()
    with pytest.warns(RuntimeWarning):
        assert np.isnan(selected_accuracy(experiment, 2))


@pytest.mark.parametrize("contents", ["0.9\n", "0.1,0.2,0.3\n", "0.1,0.2\n0.3,0.4\n", "broken\n"])
def test_misaligned_or_malformed_curve_is_not_broadcast_or_clamped(experiment, contents):
    write_curve(experiment, "val", [0.1, 0.9, 0.8, 0.2])
    path = write_curve(experiment, "test", [0.1, 0.2, 0.3, 0.4])
    path.write_text(contents)
    with pytest.warns(RuntimeWarning):
        assert np.isnan(selected_accuracy(experiment))


@pytest.mark.parametrize("kind", ["val", "test"])
def test_no_valid_checkpoint_is_unavailable(experiment, kind):
    write_curve(experiment, "val", [0.1, 0.9, 0.8, 0.2])
    write_curve(experiment, "test", [0.2, 0.3, 0.4, 0.5])
    write_curve(experiment, kind, [np.nan] * 4)
    with pytest.warns(RuntimeWarning, match="No paired"):
        assert np.isnan(selected_accuracy(experiment))


def test_experiment_schedule_overrides_root_schedule(experiment):
    write_curve(experiment, "val", [0.1, 0.2, 0.3, 0.9])
    write_curve(experiment, "test", [0.8, 0.7, 0.6, 0.5])
    assert results.get_accuracy_at_best_step(
        experiment.parent, experiment.name, 1, 1, 3, 7, 40, 2
    ) == pytest.approx(0.5)


def test_legacy_root_schedule_without_experiment_config(experiment):
    (experiment / "config.json").unlink()
    write_curve(experiment, "val", [0.1, 0.2, 0.3, 0.9])
    write_curve(experiment, "test", [0.8, 0.7, 0.6, 0.5])
    assert selected_accuracy(experiment) == pytest.approx(0.5)


def test_corrupt_experiment_config_is_unavailable(experiment):
    (experiment / "config.json").write_text("broken")
    with pytest.warns(RuntimeWarning, match="Accuracy unavailable"):
        assert np.isnan(selected_accuracy(experiment))


@pytest.fixture
def benchmark_results(tmp_path):
    config = {
        "benchmark_config": {
            "training_seed": 3, "nb_training_seeds": 1,
            "data_distribution_seed": 7, "nb_data_distribution_seeds": 1,
            "nb_honest_clients": 2, "f": 1, "nb_steps": 10,
            "set_honest_clients_as_clients": False,
            "data_distribution": {"name": "iid", "distribution_parameter": 0.0},
        },
        "evaluation_and_results": {"evaluation_delta": 4},
        "model": {"name": "cnn_mnist", "dataset_name": "mnist", "learning_rate": [0.1, 0.2]},
        "honest_clients": {"momentum": 0.0, "weight_decay": 0.0},
        "aggregator": [{"name": "Average"}, {"name": "Median"}],
        "pre_aggregators": [],
        "attack": [{"name": "SignFlipping"}, {"name": "InnerProductManipulation"}],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    directories = {}
    scores = {("Average", "SignFlipping"): 0.6,
              ("Average", "InnerProductManipulation"): 0.4,
              ("Median", "SignFlipping"): 0.5,
              ("Median", "InnerProductManipulation"): 0.7}
    for (aggregator, attack), score in scores.items():
        for lr in (0.1, 0.2):
            directory = tmp_path / (
                f"mnist_cnn_mnist_n_3_f_1_d_1_iid_0.0_{aggregator}__{attack}"
                f"_lr_{lr}_mom_0.0_wd_0.0"
            )
            directory.mkdir()
            (directory / "config.json").write_text(json.dumps(config))
            # Validation selects lr=0.2 even though lr=0.1 has higher test scores.
            validation = [0.1, 0.2, 0.3, 0.9 if lr == 0.2 else 0.4]
            test = [0.99, 0.98, score, score] if lr == 0.2 else [0.99] * 4
            write_curve(directory, "val", validation)
            write_curve(directory, "test", test)
            directories[aggregator, attack, lr] = directory
    results.find_best_hyperparameters(str(tmp_path))
    return tmp_path, directories


def test_hyperparameters_remain_validation_selected_and_final_round_is_exact(benchmark_results):
    root, _ = benchmark_results
    directory = root / "best_hyperparameters"
    for aggregator in ("Average", "Median"):
        prefix = f"mnist_cnn_mnist_n_3_f_1_d_1_iid_0.0__{aggregator}"
        np.testing.assert_allclose(
            np.loadtxt(directory / "hyperparameters" / f"{prefix}.txt"), [0.2, 0.0, 0.0]
        )
        for attack in ("SignFlipping", "InnerProductManipulation"):
            assert np.loadtxt(directory / "better_step" / f"{prefix}_{attack}.txt") == 10


@pytest.mark.parametrize("plot_function,expected", [
    ("test_heatmap", [0.4, 0.5]),
    ("aggregated_test_heatmap", [0.5]),
])
def test_heatmaps_keep_worst_attack_then_best_aggregator(benchmark_results, monkeypatch, plot_function, expected):
    root, directories = benchmark_results
    # Force fallback at the final validation peak; saved best-step files still say 10.
    write_curve(directories["Average", "InnerProductManipulation", 0.2],
                "test", [0.99, 0.98, 0.4, np.nan])
    plotted = []
    original_heatmap = results.sns.heatmap

    def capture(table, **kwargs):
        plotted.append(np.asarray(table).copy())
        return original_heatmap(table, **kwargs)

    monkeypatch.setattr(results.sns, "heatmap", capture)
    plot_directory = root / "plots"
    getattr(results, plot_function)(str(root), str(plot_directory))
    np.testing.assert_allclose([table.item() for table in plotted], expected)
    assert len(list(plot_directory.glob("*.pdf"))) == len(expected)


@pytest.mark.parametrize("plot_function", ["test_heatmap", "aggregated_test_heatmap"])
def test_missing_attack_is_not_ignored_by_heatmap_reductions(benchmark_results, monkeypatch, plot_function):
    root, directories = benchmark_results
    write_curve(directories["Average", "InnerProductManipulation", 0.2], "test", [np.nan] * 4)
    plotted = []
    original_heatmap = results.sns.heatmap

    def capture(table, **kwargs):
        plotted.append(np.asarray(table).copy())
        return original_heatmap(table, **kwargs)

    monkeypatch.setattr(results.sns, "heatmap", capture)
    with pytest.warns(RuntimeWarning):
        getattr(results, plot_function)(str(root), str(root / "plots"))
    assert np.isnan(plotted[0].item())
    if plot_function == "test_heatmap":
        assert plotted[1].item() == pytest.approx(0.5)
