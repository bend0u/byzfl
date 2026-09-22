"""Tests for client-side clipping and its position before momentum."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from byzfl.fed_framework.client import Client
from byzfl.fed_framework.clipping import (
    CLIPPING_METHODS,
    ClippingMethod,
    ConstantClipping,
    FirstGradientClipping,
    MovingAverageClipping,
    create_clipping_method,
    decode_clipping_identity,
    encode_clipping_identity,
    get_clipping_readable_label,
    normalize_clipping_config,
)


def test_constant_clipping_caps_global_l2_norm():
    result = ConstantClipping(max_norm=2.5)(torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(result.vector, torch.tensor([1.5, 2.0]))
    assert result.input_norm == pytest.approx(5.0)
    assert result.output_norm == pytest.approx(2.5)
    assert result.scale == pytest.approx(0.5)
    assert result.clipped is True


def test_first_gradient_uses_independent_per_client_state():
    client_a = FirstGradientClipping()
    client_b = FirstGradientClipping()

    first_a = client_a(torch.tensor([3.0, 4.0]))
    first_b = client_b(torch.tensor([0.0, 2.0]))
    assert first_a.clipped is False
    assert first_b.clipped is False

    second_a = client_a(torch.tensor([0.0, 10.0]))
    second_b = client_b(torch.tensor([0.0, 10.0]))
    assert torch.linalg.vector_norm(second_a.vector).item() == pytest.approx(5.0)
    assert torch.linalg.vector_norm(second_b.vector).item() == pytest.approx(2.0)
    assert second_a.threshold == pytest.approx(5.0)
    assert second_b.threshold == pytest.approx(2.0)


def test_first_gradient_multiplier_does_not_clip_calibration_gradient():
    clipper = FirstGradientClipping(multiplier=0.5)
    first = clipper(torch.tensor([3.0, 4.0]))
    second = clipper(torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(first.vector, torch.tensor([3.0, 4.0]))
    assert torch.linalg.vector_norm(second.vector).item() == pytest.approx(2.5)


def test_moving_average_uses_past_threshold_then_updates_next_threshold():
    clipper = MovingAverageClipping(window=4, multiplier=1.0)

    first = clipper(torch.tensor([0.0, 2.0]))
    assert first.clipped is False
    assert first.threshold == pytest.approx(2.0)
    assert clipper.threshold == pytest.approx(2.0)

    second = clipper(torch.tensor([0.0, 4.0]))
    assert second.threshold == pytest.approx(2.0)
    assert torch.linalg.vector_norm(second.vector).item() == pytest.approx(2.0)
    # reference <- 0.75 * 2 + 0.25 * 4 = 2.5, used next round.
    assert clipper.reference_norm == pytest.approx(2.5)
    assert clipper.threshold == pytest.approx(2.5)

    third = clipper(torch.tensor([0.0, 4.0]))
    assert third.threshold == pytest.approx(2.5)
    assert torch.linalg.vector_norm(third.vector).item() == pytest.approx(2.5)


@pytest.mark.parametrize("parameters", [
    {"window": 0},
    {"window": 1.5},
    {"multiplier": 0},
])
def test_moving_average_rejects_invalid_parameters(parameters):
    with pytest.raises((TypeError, ValueError)):
        MovingAverageClipping(**parameters)


def test_prepare_gradient_update_clips_before_advancing_momentum():
    client = Client.__new__(Client)
    client.clipping = ConstantClipping(max_norm=5.0)
    client.momentum = 0.5
    client.momentum_gradient = torch.zeros(2)
    client.get_flat_gradients = lambda: torch.tensor([6.0, 8.0])

    update = client.prepare_gradient_update()

    torch.testing.assert_close(update.raw_gradient, torch.tensor([6.0, 8.0]))
    torch.testing.assert_close(update.clipped_gradient, torch.tensor([3.0, 4.0]))
    # m <- 0.5 * 0 + (1 - 0.5) * clipped_gradient
    torch.testing.assert_close(update.client_update, torch.tensor([1.5, 2.0]))


def test_clipping_factory_defaults_to_identity_and_rejects_unknown_methods():
    vector = torch.tensor([3.0, 4.0])
    result = create_clipping_method()(vector)
    assert result.vector is vector
    with pytest.raises(ValueError, match="Unknown clipping method"):
        create_clipping_method({"name": "missing", "parameters": {}})


def test_clipping_identity_is_canonical_reversible_and_url_safe():
    implicit = {"name": "moving_average", "parameters": {}}
    explicit = {
        "parameters": {"multiplier": 1, "window": 100},
        "name": "moving_average",
    }

    token = encode_clipping_identity(implicit)

    assert token == encode_clipping_identity(explicit)
    assert token == "cv1.m.100.1.0"
    assert set(token) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_."
    )
    assert decode_clipping_identity(token) == {
        "name": "moving_average",
        "parameters": {"multiplier": 1.0, "window": 100},
    }
    assert get_clipping_readable_label(implicit) == (
        "clip-moving-average-w100-mult-1.0"
    )


def test_clipping_identity_changes_with_method_or_parameter():
    configurations = [
        {"name": "none"},
        {"name": "constant", "parameters": {"max_norm": 0.5}},
        {"name": "constant", "parameters": {"max_norm": 1.0}},
        {"name": "first_gradient", "parameters": {"multiplier": 1.0}},
        {
            "name": "moving_average",
            "parameters": {"window": 100, "multiplier": 1.0},
        },
    ]
    tokens = [encode_clipping_identity(config) for config in configurations]
    assert len(tokens) == len(set(tokens))


def test_missing_and_explicit_none_have_one_canonical_identity():
    assert normalize_clipping_config() == {
        "name": "none",
        "parameters": {},
    }
    assert encode_clipping_identity() == encode_clipping_identity({"name": "none"})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_clipping_rejects_nonfinite_parameters(value):
    with pytest.raises(ValueError):
        ConstantClipping(value)


def test_custom_method_can_apply_a_non_norm_transformation(monkeypatch):
    class CoordinateClipping(ClippingMethod):
        def __init__(self, bound):
            self.bound = bound

        def _clip(self, gradient, gradient_norm):
            return gradient.clamp(-self.bound, self.bound), self.bound

    monkeypatch.setitem(CLIPPING_METHODS, "coordinate_test", CoordinateClipping)
    clipper = create_clipping_method({
        "name": "coordinate_test", "parameters": {"bound": 2.0}
    })
    result = clipper(torch.tensor([-3.0, 1.0, 4.0]))
    torch.testing.assert_close(result.vector, torch.tensor([-2.0, 1.0, 2.0]))
    assert result.clipped is True


@pytest.mark.parametrize("vector", [
    torch.tensor([float("nan")]),
    torch.tensor([float("inf")]),
])
def test_clipping_rejects_nonfinite_gradients(vector):
    with pytest.raises(ValueError, match="NaN or infinity"):
        ConstantClipping(1.0)(vector)
