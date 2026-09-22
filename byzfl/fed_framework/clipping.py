"""Client-side gradient clipping methods.

Clipping methods operate on one client's flat raw gradient before client
momentum is updated.  A method instance belongs to one client, which lets a
stateful method (such as ``first_gradient``) retain its own threshold across
federated rounds.
"""

import base64
from dataclasses import dataclass
import inspect
import json
import math
import re

import torch


@dataclass(frozen=True)
class ClippingResult:
    """A clipped vector and the diagnostics needed by experiment recorders."""

    vector: torch.Tensor
    threshold: float | None
    input_norm: float
    output_norm: float
    scale: float
    clipped: bool


class ClippingMethod:
    """Base class for client-side clipping methods.

    Subclasses implement :meth:`_clip` and return ``(vector, threshold)``.
    This supports transformations beyond global norm clipping while the base
    class enforces a common shape/device/dtype contract and diagnostics.
    """

    def _clip(self, gradient, gradient_norm):
        raise NotImplementedError

    def __call__(self, gradient):
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("A clipping method expects a torch.Tensor.")
        if gradient.ndim != 1:
            raise ValueError("A clipping method expects a flat one-dimensional gradient.")
        if not torch.isfinite(gradient).all():
            raise ValueError("Cannot clip a gradient containing NaN or infinity.")

        input_norm = torch.linalg.vector_norm(gradient).item()
        clipped_gradient, threshold = self._clip(gradient, input_norm)
        if not isinstance(clipped_gradient, torch.Tensor):
            raise TypeError("A clipping method must return a torch.Tensor.")
        if clipped_gradient.shape != gradient.shape:
            raise ValueError("A clipping method must preserve the gradient shape.")
        if clipped_gradient.device != gradient.device or clipped_gradient.dtype != gradient.dtype:
            raise ValueError("A clipping method must preserve the gradient device and dtype.")
        if not torch.isfinite(clipped_gradient).all():
            raise ValueError("A clipping method returned NaN or infinity.")

        output_norm = torch.linalg.vector_norm(clipped_gradient).item()
        scale = 1.0 if input_norm == 0 else output_norm / input_norm
        return ClippingResult(
            vector=clipped_gradient,
            threshold=threshold,
            input_norm=input_norm,
            output_norm=output_norm,
            scale=scale,
            clipped=not torch.equal(clipped_gradient, gradient),
        )

    @staticmethod
    def _clip_to_threshold(gradient, gradient_norm, threshold):
        if gradient_norm <= threshold:
            return gradient
        scale = 0.0 if gradient_norm == 0 else threshold / gradient_norm
        return gradient * scale


CLIPPING_METHODS = {}


def register_clipping_method(name):
    """Register a clipping class under the name used in configuration files."""
    if not isinstance(name, str) or not name:
        raise ValueError("A clipping method name must be a non-empty string.")

    def register(method_class):
        if name in CLIPPING_METHODS:
            raise ValueError(f"Clipping method '{name}' is already registered.")
        if not issubclass(method_class, ClippingMethod):
            raise TypeError("Registered clipping methods must inherit ClippingMethod.")
        CLIPPING_METHODS[name] = method_class
        return method_class

    return register


@register_clipping_method("none")
class NoClipping(ClippingMethod):
    """Leave every gradient unchanged."""

    def __call__(self, gradient):
        # Keep the disabled path as cheap as the original client code.  The
        # measurement recorder computes diagnostics only on sampled rounds.
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("A clipping method expects a torch.Tensor.")
        if gradient.ndim != 1:
            raise ValueError("A clipping method expects a flat one-dimensional gradient.")
        return ClippingResult(
            vector=gradient,
            threshold=None,
            input_norm=float("nan"),
            output_norm=float("nan"),
            scale=1.0,
            clipped=False,
        )

    def _clip(self, gradient, gradient_norm):
        return gradient, None


@register_clipping_method("constant")
class ConstantClipping(ClippingMethod):
    """Clip every raw gradient to one fixed global L2-norm bound."""

    def __init__(self, max_norm):
        if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
            raise TypeError("constant.max_norm must be a positive number.")
        if not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError("constant.max_norm must be positive.")
        self.max_norm = float(max_norm)

    def _clip(self, gradient, gradient_norm):
        return self._clip_to_threshold(gradient, gradient_norm, self.max_norm), self.max_norm


@register_clipping_method("first_gradient")
class FirstGradientClipping(ClippingMethod):
    """Freeze a per-client cap from that client's first raw gradient norm.

    The calibration gradient is returned unchanged.  Starting with the next
    round, the cap is ``multiplier * first_gradient_norm``.
    """

    def __init__(self, multiplier=1.0):
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise TypeError("first_gradient.multiplier must be a positive number.")
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("first_gradient.multiplier must be positive.")
        self.multiplier = float(multiplier)
        self.threshold = None

    def _clip(self, gradient, gradient_norm):
        if self.threshold is None:
            self.threshold = self.multiplier * gradient_norm
            # The first gradient calibrates the method; clipping starts on the
            # following call even when multiplier is smaller than one.
            return gradient, self.threshold
        return self._clip_to_threshold(gradient, gradient_norm, self.threshold), self.threshold


@register_clipping_method("moving_average")
class MovingAverageClipping(ClippingMethod):
    """Adapt a per-client L2 cap from an EMA of past raw gradient norms.

    ``window`` controls the EMA rate through ``alpha = 1 / window``.  The
    current gradient is clipped using the threshold computed at the end of the
    previous call, so a spike cannot immediately loosen its own bound.  Its raw
    norm is then incorporated into the reference used by the next round.
    """

    def __init__(self, window=100, multiplier=1.0):
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError("moving_average.window must be a positive integer.")
        if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
            raise TypeError("moving_average.multiplier must be a positive number.")
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("moving_average.multiplier must be positive.")
        self.window = window
        self.alpha = 1.0 / window
        self.multiplier = float(multiplier)
        self.reference_norm = None
        self.threshold = None

    def _clip(self, gradient, gradient_norm):
        if self.reference_norm is None:
            self.reference_norm = gradient_norm
            self.threshold = self.multiplier * self.reference_norm
            return gradient, self.threshold

        threshold_used = self.threshold
        clipped = self._clip_to_threshold(gradient, gradient_norm, threshold_used)

        self.reference_norm = (
            (1.0 - self.alpha) * self.reference_norm
            + self.alpha * gradient_norm
        )
        self.threshold = self.multiplier * self.reference_norm
        return clipped, threshold_used


def create_clipping_method(config=None):
    """Build one clipping method from an optional configuration dictionary."""
    if config is None:
        return NoClipping()
    if not isinstance(config, dict):
        raise TypeError("honest_clients.clipping must be a dictionary.")

    unknown = set(config) - {"name", "parameters"}
    if unknown:
        raise ValueError(f"Unknown clipping configuration fields: {sorted(unknown)}")
    name = config.get("name", "none")
    parameters = config.get("parameters", {})
    if not isinstance(name, str):
        raise TypeError("honest_clients.clipping.name must be a string.")
    if not isinstance(parameters, dict):
        raise TypeError("honest_clients.clipping.parameters must be a dictionary.")
    if name not in CLIPPING_METHODS:
        available = ", ".join(sorted(CLIPPING_METHODS))
        raise ValueError(f"Unknown clipping method '{name}'. Available methods: {available}.")

    try:
        return CLIPPING_METHODS[name](**parameters)
    except TypeError as error:
        raise TypeError(f"Invalid parameters for clipping method '{name}': {error}") from error


CLIPPING_IDENTITY_VERSION = "cv1"


def normalize_clipping_config(config=None):
    """Return the canonical, fully explicit identity of a clipping policy.

    Canonicalization deliberately maps omitted and explicit defaults to the
    same object. This makes result names stable across dictionary ordering and
    harmless spelling differences such as max_norm=1 versus 1.0.
    """
    method = create_clipping_method(config)
    name = "none" if config is None else config.get("name", "none")

    if isinstance(method, NoClipping):
        parameters = {}
    elif isinstance(method, ConstantClipping):
        parameters = {"max_norm": method.max_norm}
    elif isinstance(method, FirstGradientClipping):
        parameters = {"multiplier": method.multiplier}
    elif isinstance(method, MovingAverageClipping):
        parameters = {
            "multiplier": method.multiplier,
            "window": method.window,
        }
    else:
        # Registered extensions get stable constructor defaults even when the
        # author did not provide a custom identity hook.
        supplied = {} if config is None else config.get("parameters", {})
        try:
            bound = inspect.signature(CLIPPING_METHODS[name]).bind(**supplied)
            bound.apply_defaults()
        except TypeError as error:
            raise TypeError(
                f"Invalid parameters for clipping method '{name}': {error}"
            ) from error
        parameters = dict(bound.arguments)

    canonical = {"name": name, "parameters": parameters}
    try:
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Clipping parameters must have a finite JSON representation to be "
            "used in a result identity."
        ) from error
    return canonical


def _canonical_clipping_json(config=None):
    return json.dumps(
        normalize_clipping_config(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def encode_clipping_identity(config=None):
    """Encode a canonical clipping configuration as a compact path token."""
    clipping = normalize_clipping_config(config)
    name = clipping["name"]
    parameters = clipping["parameters"]
    if name == "none":
        payload = "n"
    elif name == "constant":
        payload = f"c.{_readable_number(parameters['max_norm'])}"
    elif name == "first_gradient":
        payload = f"f.{_readable_number(parameters['multiplier'])}"
    elif name == "moving_average":
        payload = (
            f"m.{parameters['window']}."
            f"{_readable_number(parameters['multiplier'])}"
        )
    else:
        # Extensible fallback: custom names and parameters remain lossless.
        encoded = base64.urlsafe_b64encode(
            _canonical_clipping_json(clipping).encode("utf-8")
        ).decode("ascii").rstrip("=")
        payload = f"x.{encoded}"
    return f"{CLIPPING_IDENTITY_VERSION}.{payload}"


def decode_clipping_identity(token):
    """Decode and validate a token created by encode_clipping_identity."""
    prefix = f"{CLIPPING_IDENTITY_VERSION}."
    if not isinstance(token, str) or not token.startswith(prefix):
        raise ValueError(f"A clipping identity must start with '{prefix}'.")
    payload = token[len(prefix):]
    try:
        if payload == "n":
            decoded = {"name": "none", "parameters": {}}
        elif payload.startswith("c."):
            decoded = {
                "name": "constant",
                "parameters": {"max_norm": float(payload[2:])},
            }
        elif payload.startswith("f."):
            decoded = {
                "name": "first_gradient",
                "parameters": {"multiplier": float(payload[2:])},
            }
        elif payload.startswith("m."):
            window, multiplier = payload[2:].split(".", 1)
            decoded = {
                "name": "moving_average",
                "parameters": {
                    "window": int(window),
                    "multiplier": float(multiplier),
                },
            }
        elif payload.startswith("x."):
            encoded = payload[2:]
            padding = "=" * (-len(encoded) % 4)
            raw = base64.b64decode(
                encoded + padding, altchars=b"-_", validate=True
            )
            decoded = json.loads(raw.decode("utf-8"))
        else:
            raise ValueError("Unknown clipping identity mode.")
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid clipping identity payload.") from error

    normalized = normalize_clipping_config(decoded)
    if encode_clipping_identity(normalized) != token:
        raise ValueError("The clipping identity payload is not canonical.")
    return normalized


def _readable_number(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _safe_label(value):
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.")
    return label or "value"


def get_clipping_readable_label(config=None):
    """Return a short human-readable label derived from canonical settings."""
    clipping = normalize_clipping_config(config)
    name = clipping["name"]
    parameters = clipping["parameters"]
    if name == "none":
        return "clip-none"
    if name == "constant":
        return f"clip-constant-maxnorm-{_readable_number(parameters['max_norm'])}"
    if name == "first_gradient":
        return f"clip-first-gradient-mult-{_readable_number(parameters['multiplier'])}"
    if name == "moving_average":
        return (
            f"clip-moving-average-w{parameters['window']}-"
            f"mult-{_readable_number(parameters['multiplier'])}"
        )

    # The complete custom configuration remains in the reversible token. The
    # readable part is a hint and is intentionally bounded.
    parts = [f"clip-{_safe_label(name)}"]
    for key, value in sorted(parameters.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            parts.append(f"{_safe_label(key)}-{_safe_label(_readable_number(value))}")
    return "-".join(parts)[:96].rstrip("-.")


def get_clipping_result_suffix(config=None):
    """Return the readable label followed by the reversible identity token."""
    return f"{get_clipping_readable_label(config)}__{encode_clipping_identity(config)}"
