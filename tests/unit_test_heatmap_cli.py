"""Tests for the reusable benchmark heatmap command-line interface."""

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.benchmark import heatmap_cli


def test_generic_cli_requires_a_results_directory():
    with pytest.raises(SystemExit):
        heatmap_cli.build_parser().parse_args([])


def test_experiment_wrapper_can_supply_a_default_results_directory(tmp_path):
    args = heatmap_cli.build_parser(tmp_path).parse_args([])
    assert args.results == tmp_path.resolve()


def test_cli_forwards_selected_views_and_caption_mode(tmp_path, monkeypatch):
    (tmp_path / "best_hyperparameters").mkdir()
    (tmp_path / "config.json").write_text(json.dumps({
        "evaluation_and_results": {"evaluate_on_test": True},
    }))
    captured = {}

    def fake_generate(results, output, **kwargs):
        captured.update(results=results, output=output, **kwargs)
        output_root = results / "heatmaps_captioned"
        output_root.mkdir()
        generated = output_root / "best.pdf"
        generated.touch()
        return [generated]

    monkeypatch.setattr(heatmap_cli, "generate_heatmaps", fake_generate)
    heatmap_cli.main(
        default_results=tmp_path,
        argv=["--views", "best", "per-attack", "--caption"],
    )

    assert captured["results"] == tmp_path.resolve()
    assert captured["output"] is None
    assert captured["views"] == ["best", "per-attack"]
    assert captured["caption"] is True
    assert captured["caption_fields"] == heatmap_cli.CAPTION_FIELDS
