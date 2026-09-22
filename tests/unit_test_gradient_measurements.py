"""Tests for extensible honest-gradient and server measurements."""

import csv
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.benchmark.measurements import (
    HONEST_METRICS,
    MeasurementRecorder,
    compute_honest_metrics,
    compute_server_metrics,
)
from byzfl.benchmark.managers import ParamsManager, get_model_result_name
from byzfl.fed_framework.clipping import ClippingResult
from byzfl.fed_framework.server import Server


def test_honest_metrics_match_analytical_example():
    vectors = [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])]
    metrics = compute_honest_metrics(
        vectors, ["heterogeneity", "norms", "cosine_similarity"]
    )
    assert metrics["heterogeneity"] == pytest.approx(1.0)
    assert metrics["norm_mean"] == pytest.approx(2.0)
    assert metrics["norm_median"] == pytest.approx(2.0)
    assert metrics["mean_vector_norm"] == pytest.approx(2.0)
    assert metrics["norm_client_0"] == pytest.approx(1.0)
    assert metrics["norm_client_1"] == pytest.approx(3.0)
    assert metrics["cosine_mean"] == pytest.approx(1.0)
    assert metrics["cosine_median"] == pytest.approx(1.0)
    assert metrics["cosine_valid_pairs"] == 1


def test_opposite_honest_gradients_have_negative_cosine_and_zero_mean():
    metrics = compute_honest_metrics(
        [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])],
        ["heterogeneity", "norms", "cosine_similarity"],
    )
    assert metrics["heterogeneity"] == pytest.approx(1.0)
    assert metrics["mean_vector_norm"] == pytest.approx(0.0)
    assert metrics["cosine_mean"] == pytest.approx(-1.0)


def test_server_metrics_compare_actual_aggregate_with_honest_mean():
    metrics = compute_server_metrics(
        aggregate=torch.tensor([2.0, 0.0]),
        honest_updates=[torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])],
        metric_names=["aggregate_norm", "cosine_with_honest_mean"],
    )
    assert metrics["aggregate_norm"] == pytest.approx(2.0)
    assert metrics["cosine_with_honest_mean"] == pytest.approx(1.0)
    assert metrics["cosine_with_honest_mean_valid"] == 1


def test_recorder_samples_requested_rounds_and_writes_stage_rows(tmp_path):
    recorder = MeasurementRecorder({
        "enabled": True,
        "every_n_rounds": 2,
        "honest_stages": ["raw_gradient", "client_update"],
        "honest_metrics": ["heterogeneity", "norms"],
        "server_metrics": ["aggregate_norm"],
    })
    diagnostic = ClippingResult(
        vector=torch.tensor([1.0, 0.0]), threshold=1.0,
        input_norm=2.0, output_norm=1.0, scale=0.5, clipped=True,
    )
    updates = [
        SimpleNamespace(
            raw_gradient=torch.tensor([2.0, 0.0]),
            clipped_gradient=torch.tensor([1.0, 0.0]),
            client_update=torch.tensor([1.0, 0.0]),
            clipping=diagnostic,
        ),
        SimpleNamespace(
            raw_gradient=torch.tensor([4.0, 0.0]),
            clipped_gradient=torch.tensor([2.0, 0.0]),
            client_update=torch.tensor([2.0, 0.0]),
            clipping=diagnostic,
        ),
    ]

    recorder.record_honest(1, updates)
    recorder.record_honest(2, updates)
    recorder.record_server(2, torch.tensor([1.5, 0.0]), [u.client_update for u in updates])
    recorder.write(tmp_path, training_seed=3, data_distribution_seed=7)

    directory = tmp_path / "measurements_tr_seed_3_dd_seed_7"
    with (directory / "honest_gradients.csv").open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [(row["round"], row["stage"]) for row in rows] == [
        ("2", "raw_gradient"), ("2", "client_update")
    ]
    with (directory / "clipping.csv").open(newline="") as csv_file:
        clipping_rows = list(csv.DictReader(csv_file))
    assert len(clipping_rows) == 2
    assert clipping_rows[0]["clipped"] == "1"
    assert (directory / "server.csv").is_file()
    assert (directory / "metadata.json").is_file()


def test_zero_vectors_are_excluded_from_pairwise_cosine():
    metrics = compute_honest_metrics(
        [torch.zeros(2), torch.tensor([1.0, 0.0])], ["cosine_similarity"]
    )
    assert metrics["cosine_valid_pairs"] == 0
    assert metrics["cosine_mean"] != metrics["cosine_mean"]  # NaN


def test_custom_metric_is_selected_by_registry_name(monkeypatch):
    monkeypatch.setitem(
        HONEST_METRICS,
        "maximum_norm_test",
        lambda stacked: {"maximum_norm": stacked.norm(dim=1).max().item()},
    )
    metrics = compute_honest_metrics(
        [torch.tensor([3.0, 4.0]), torch.tensor([1.0, 0.0])],
        ["maximum_norm_test"],
    )
    assert metrics == {"maximum_norm": pytest.approx(5.0)}


def test_server_returns_the_exact_aggregate_it_applies():
    server = Server.__new__(Server)
    aggregate = torch.tensor([2.0, -1.0])
    calls = []
    server.aggregate = lambda vectors: calls.append("aggregate") or aggregate
    server.set_gradients = lambda vector: calls.append(("set", vector))
    server._step = lambda: calls.append("step")

    returned = server.update_model_with_gradients([torch.ones(2)])

    assert returned is aggregate
    assert calls == ["aggregate", ("set", aggregate), "step"]


def test_optional_config_is_preserved_and_changes_result_identity():
    plain = {"model": {"name": "cnn_mnist"}}
    clipping = {
        "model": {"name": "cnn_mnist"},
        "honest_clients": {
            "clipping": {"name": "constant", "parameters": {"max_norm": 2.0}}
        },
    }
    measured = {
        **clipping,
        "measurements": {"enabled": True, "honest_metrics": ["heterogeneity"]},
    }
    assert get_model_result_name(plain) == "cnn_mnist"
    assert get_model_result_name(clipping).startswith("cnn_mnist_clip-constant-")
    assert get_model_result_name(measured) != get_model_result_name(clipping)

    resolved = ParamsManager(measured).get_data()
    assert resolved["honest_clients"]["clipping"] == clipping["honest_clients"]["clipping"]
    assert resolved["measurements"] == measured["measurements"]
