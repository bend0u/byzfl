"""Configurable measurements for honest DSGD gradients and server aggregates."""

import csv
import json
import os

import torch


HONEST_METRICS = {}
SERVER_METRICS = {}


def register_honest_metric(name):
    """Register a metric that receives stacked honest vectors ``(clients, d)``."""
    def register(function):
        if name in HONEST_METRICS:
            raise ValueError(f"Honest metric '{name}' is already registered.")
        HONEST_METRICS[name] = function
        return function
    return register


def register_server_metric(name):
    """Register a metric of the aggregate and mean honest client update."""
    def register(function):
        if name in SERVER_METRICS:
            raise ValueError(f"Server metric '{name}' is already registered.")
        SERVER_METRICS[name] = function
        return function
    return register


def _stack_honest_vectors(vectors):
    if not vectors:
        raise ValueError("Honest-gradient measurements require at least one client.")
    if not all(isinstance(vector, torch.Tensor) and vector.ndim == 1 for vector in vectors):
        raise TypeError("Honest-gradient measurements expect flat torch tensors.")
    stacked = torch.stack(vectors)
    if not torch.isfinite(stacked).all():
        raise ValueError("Cannot measure honest vectors containing NaN or infinity.")
    return stacked


@register_honest_metric("heterogeneity")
def heterogeneity(stacked):
    """Empirical honest-gradient dissimilarity around the honest mean.

    This is ``mean_i ||g_i - mean_j(g_j)||_2^2``, the per-round empirical
    quantity corresponding to the gradient-dissimilarity term used in robust
    federated-learning analyses.
    """
    honest_mean = stacked.mean(dim=0)
    value = (stacked - honest_mean).pow(2).sum(dim=1).mean()
    return {"heterogeneity": value.item()}


@register_honest_metric("norms")
def norms(stacked):
    client_norms = torch.linalg.vector_norm(stacked, dim=1)
    row = {
        "norm_mean": client_norms.mean().item(),
        "norm_median": torch.quantile(client_norms, 0.5).item(),
        "mean_vector_norm": torch.linalg.vector_norm(stacked.mean(dim=0)).item(),
    }
    row.update({f"norm_client_{index}": value for index, value in enumerate(client_norms.tolist())})
    return row


@register_honest_metric("cosine_similarity")
def cosine_similarity(stacked):
    """Mean and median pairwise cosine among nonzero honest vectors."""
    vector_norms = torch.linalg.vector_norm(stacked, dim=1)
    pairs = torch.triu_indices(stacked.shape[0], stacked.shape[0], offset=1, device=stacked.device)
    valid = (vector_norms[pairs[0]] > 0) & (vector_norms[pairs[1]] > 0)
    valid_pairs = int(valid.sum().item())
    if valid_pairs == 0:
        return {
            "cosine_mean": float("nan"),
            "cosine_median": float("nan"),
            "cosine_valid_pairs": 0,
        }

    left, right = pairs[0][valid], pairs[1][valid]
    cosine = (stacked[left] * stacked[right]).sum(dim=1)
    cosine = cosine / (vector_norms[left] * vector_norms[right])
    return {
        "cosine_mean": cosine.mean().item(),
        "cosine_median": torch.quantile(cosine, 0.5).item(),
        "cosine_valid_pairs": valid_pairs,
    }


@register_server_metric("aggregate_norm")
def aggregate_norm(aggregate, honest_mean):
    return {"aggregate_norm": torch.linalg.vector_norm(aggregate).item()}


@register_server_metric("cosine_with_honest_mean")
def cosine_with_honest_mean(aggregate, honest_mean):
    aggregate_size = torch.linalg.vector_norm(aggregate)
    honest_size = torch.linalg.vector_norm(honest_mean)
    if aggregate_size == 0 or honest_size == 0:
        value = float("nan")
        valid = 0
    else:
        value = torch.dot(aggregate, honest_mean).div(aggregate_size * honest_size).item()
        valid = 1
    return {"cosine_with_honest_mean": value, "cosine_with_honest_mean_valid": valid}


def compute_honest_metrics(vectors, metric_names):
    """Compute registered metrics for a collection of honest vectors."""
    stacked = _stack_honest_vectors(vectors)
    result = {}
    for name in metric_names:
        if name not in HONEST_METRICS:
            available = ", ".join(sorted(HONEST_METRICS))
            raise ValueError(f"Unknown honest metric '{name}'. Available metrics: {available}.")
        result.update(HONEST_METRICS[name](stacked))
    return result


def compute_server_metrics(aggregate, honest_updates, metric_names):
    """Measure the actual aggregate without changing the server update."""
    stacked = _stack_honest_vectors(honest_updates)
    if not isinstance(aggregate, torch.Tensor) or aggregate.ndim != 1:
        raise TypeError("Server measurements expect a flat aggregate tensor.")
    if not torch.isfinite(aggregate).all():
        raise ValueError("Cannot measure an aggregate containing NaN or infinity.")
    honest_mean = stacked.mean(dim=0)
    result = {}
    for name in metric_names:
        if name not in SERVER_METRICS:
            available = ", ".join(sorted(SERVER_METRICS))
            raise ValueError(f"Unknown server metric '{name}'. Available metrics: {available}.")
        result.update(SERVER_METRICS[name](aggregate, honest_mean))
    return result


class MeasurementRecorder:
    """Collect requested metrics and write one compact CSV per measurement kind."""

    STAGES = ("raw_gradient", "clipped_gradient", "client_update")

    def __init__(self, config=None):
        config = {} if config is None else config
        if not isinstance(config, dict):
            raise TypeError("measurements must be a dictionary.")
        unknown = set(config) - {
            "enabled", "every_n_rounds", "honest_stages", "honest_metrics", "server_metrics"
        }
        if unknown:
            raise ValueError(f"Unknown measurement configuration fields: {sorted(unknown)}")

        self.enabled = config.get("enabled", False)
        if not isinstance(self.enabled, bool):
            raise TypeError("measurements.enabled must be a bool.")
        self.every_n_rounds = config.get("every_n_rounds", 1)
        if (isinstance(self.every_n_rounds, bool)
                or not isinstance(self.every_n_rounds, int)
                or self.every_n_rounds <= 0):
            raise ValueError("measurements.every_n_rounds must be a positive integer.")

        self.honest_stages = config.get("honest_stages", list(self.STAGES))
        self.honest_metrics = config.get(
            "honest_metrics", ["heterogeneity", "norms", "cosine_similarity"]
        )
        self.server_metrics = config.get(
            "server_metrics", ["aggregate_norm", "cosine_with_honest_mean"]
        )
        self._validate_name_list("honest_stages", self.honest_stages, set(self.STAGES))
        self._validate_name_list("honest_metrics", self.honest_metrics, set(HONEST_METRICS))
        self._validate_name_list("server_metrics", self.server_metrics, set(SERVER_METRICS))

        self.config = dict(config)
        self.honest_rows = []
        self.clipping_rows = []
        self.server_rows = []

    @staticmethod
    def _validate_name_list(field, values, available):
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise TypeError(f"measurements.{field} must be a list of strings.")
        unknown = set(values) - available
        if unknown:
            raise ValueError(f"Unknown values in measurements.{field}: {sorted(unknown)}")

    def should_measure(self, round_index):
        return self.enabled and round_index % self.every_n_rounds == 0

    def record_honest(self, round_index, updates):
        if not self.should_measure(round_index):
            return
        for stage in self.honest_stages:
            vectors = [getattr(update, stage) for update in updates]
            row = {"round": round_index, "stage": stage}
            row.update(compute_honest_metrics(vectors, self.honest_metrics))
            self.honest_rows.append(row)

        for client_id, update in enumerate(updates):
            diagnostic = update.clipping
            input_norm = torch.linalg.vector_norm(update.raw_gradient).item()
            output_norm = torch.linalg.vector_norm(update.clipped_gradient).item()
            scale = 1.0 if input_norm == 0 else output_norm / input_norm
            self.clipping_rows.append({
                "round": round_index,
                "client_id": client_id,
                "input_norm": input_norm,
                "output_norm": output_norm,
                "threshold": (
                    diagnostic.threshold if diagnostic.threshold is not None else float("nan")
                ),
                "scale": scale,
                "clipped": int(diagnostic.clipped),
            })

    def record_server(self, round_index, aggregate, honest_updates):
        if not self.should_measure(round_index):
            return
        row = {"round": round_index}
        row.update(compute_server_metrics(aggregate, honest_updates, self.server_metrics))
        self.server_rows.append(row)

    def write(self, experiment_path, training_seed, data_distribution_seed):
        if not self.enabled:
            return
        directory = os.path.join(
            experiment_path,
            f"measurements_tr_seed_{training_seed}_dd_seed_{data_distribution_seed}",
        )
        os.makedirs(directory, exist_ok=True)
        self._write_csv(os.path.join(directory, "honest_gradients.csv"), self.honest_rows)
        self._write_csv(os.path.join(directory, "clipping.csv"), self.clipping_rows)
        self._write_csv(os.path.join(directory, "server.csv"), self.server_rows)
        with open(os.path.join(directory, "metadata.json"), "w") as metadata_file:
            json.dump({
                "schema_version": 1,
                "training_seed": training_seed,
                "data_distribution_seed": data_distribution_seed,
                "configuration": self.config,
                "heterogeneity_definition": "mean_i ||g_i - mean_j(g_j)||_2^2",
            }, metadata_file, indent=4, allow_nan=False)

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        fieldnames = []
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
