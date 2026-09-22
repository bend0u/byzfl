# Client clipping and gradient measurements

Client clipping is optional and currently applies to DSGD raw honest gradients
before client momentum.  Each client owns its clipping-method instance, so a
stateful method keeps independent state across federated rounds.

## Configuration

Add a `clipping` object under `honest_clients`.  A constant global L2 cap is:

```json
"clipping": {
    "name": "constant",
    "parameters": {"max_norm": 10.0}
}
```

To use each client's first raw-gradient norm as that client's fixed cap:

```json
"clipping": {
    "name": "first_gradient",
    "parameters": {"multiplier": 1.0}
}
```

The first gradient calibrates `first_gradient` and is left unchanged.  Clipping
starts on the following round.  Omitting `clipping`, or selecting `none`, keeps
the original client behavior.

To adapt the cap slowly from an exponential moving average of past raw norms:

```json
"clipping": {
    "name": "moving_average",
    "parameters": {
        "window": 100,
        "multiplier": 1.0
    }
}
```

Each client keeps its own `reference_norm` and `threshold`.  A gradient is
clipped with the threshold from the preceding round; afterward its raw norm
updates the reference with `alpha = 1 / window`.  Thus, the current gradient
cannot increase its own bound.

Measurements are also optional:

```json
"measurements": {
    "enabled": true,
    "every_n_rounds": 10,
    "honest_stages": [
        "raw_gradient",
        "clipped_gradient",
        "client_update"
    ],
    "honest_metrics": [
        "heterogeneity",
        "norms",
        "cosine_similarity"
    ],
    "server_metrics": [
        "aggregate_norm",
        "cosine_with_honest_mean"
    ]
}
```

Here `client_update` is the post-momentum vector sent by an honest client.  The
server cosine compares the actual final aggregate with the mean of those honest
transmitted vectors.  It is only a measurement: the server still applies its
configured aggregate.

The measured heterogeneity is

```text
mean_i ||g_i - mean_j(g_j)||_2^2
```

computed among honest clients at each requested stage.  The complete runnable
configuration is `configs/clipping_and_measurements_mnist.json` in this docs
directory.

After placing the configuration at the repository root as `config.json`, run:

```python
from byzfl import run_benchmark

run_benchmark(nb_jobs=1)
```

Each experiment directory contains a seed-specific measurement directory with
`honest_gradients.csv`, `clipping.csv`, `server.csv`, and `metadata.json`.

## Adding a clipping method

Implement `_clip` and register the class.  It receives the current flat raw
gradient and its L2 norm, and returns the transformed vector plus an optional
scalar threshold for diagnostics.

```python
from byzfl import ClippingMethod, register_clipping_method


@register_clipping_method("my_method")
class MyMethod(ClippingMethod):
    def __init__(self, strength):
        self.strength = strength

    def _clip(self, gradient, gradient_norm):
        transformed = gradient.clamp(-self.strength, self.strength)
        return transformed, self.strength
```

The base class checks that a custom method preserves shape, device, and dtype
and rejects nonfinite input or output.

## Adding a metric

An honest metric receives one tensor shaped `(number_of_clients, parameters)`
and returns named scalar values:

```python
from byzfl.benchmark.measurements import register_honest_metric


@register_honest_metric("maximum_norm")
def maximum_norm(stacked_vectors):
    value = stacked_vectors.norm(dim=1).max().item()
    return {"maximum_norm": value}
```

Add `maximum_norm` to `honest_metrics` to record it at every selected stage.
