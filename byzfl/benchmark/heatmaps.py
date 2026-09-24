"""Configurable test-accuracy heatmaps for benchmark result directories."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from byzfl.benchmark.evaluate_results import (
    _attacks_for_f,
    _for_each_result_identity,
    _test_accuracy_at_best_validation,
    ensure_list,
)
from byzfl.benchmark.managers import get_model_result_name
from byzfl.fed_framework.clipping import normalize_clipping_config


HEATMAP_VIEWS = ("best", "per-aggregator", "per-attack")
CAPTION_FIELDS = ("model", "clipping", "pipeline", "attacks")

_ATTACK_LABELS = {
    "Optimal_ALittleIsEnough": "ALIE",
    "ALittleIsEnough": "ALIE",
    "SignFlipping": "SignFlip",
    "Optimal_InnerProductManipulation": "IPM",
    "InnerProductManipulation": "IPM",
}

_AGGREGATOR_LABELS = {
    "GeometricMedian": "GeomMedian",
    "CenteredClipping": "CenteredClip",
}


def _number_label(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def _model_label(config):
    name = config.get("model", {}).get("name", "model")
    labels = {
        "cnn_mnist": "CNN-MNIST",
        "cnn_cifar": "CNN-CIFAR",
        "fc_mnist": "FC-MNIST",
    }
    return labels.get(name, name.replace("_", "-"))


def _clipping_label(config):
    clipping = normalize_clipping_config(
        config.get("honest_clients", {}).get("clipping")
    )
    name = clipping["name"]
    parameters = clipping["parameters"]
    if name == "none":
        return "no clipping"
    if name == "constant":
        return f"constant clip ({_number_label(parameters['max_norm'])})"
    if name == "first_gradient":
        return f"first-gradient clip (×{_number_label(parameters['multiplier'])})"
    if name == "moving_average":
        return (
            f"moving-average clip (w={parameters['window']}, "
            f"×{_number_label(parameters['multiplier'])})"
        )
    return name.replace("_", " ")


def _aggregator_label(name):
    return _AGGREGATOR_LABELS.get(name, name)


def _attack_label(name):
    return _ATTACK_LABELS.get(name, name)


def _caption_text(config, pre_aggregators, aggregators, attacks, fields):
    """Build the compact, visible caption for one concrete heatmap."""
    fields = tuple(fields)
    unknown = set(fields) - set(CAPTION_FIELDS)
    if unknown:
        raise ValueError(f"Unknown caption fields: {sorted(unknown)}")

    aggregation = (
        _aggregator_label(aggregators[0]["name"])
        if len(aggregators) == 1
        else "{" + ", ".join(
            _aggregator_label(aggregator["name"])
            for aggregator in aggregators
        ) + "}"
    )
    pipeline = " → ".join(
        [pre_aggregator["name"] for pre_aggregator in pre_aggregators]
        + [aggregation]
    )
    segments = {
        "model": _model_label(config),
        "clipping": _clipping_label(config),
        "pipeline": pipeline,
        "attacks": ", ".join(
            _attack_label(attack["name"]) for attack in attacks
        ),
    }
    return " | ".join(segments[field] for field in fields)


def _caption_for_figure(caption):
    """Wrap a long caption only at semantic separators."""
    if len(caption) <= 110:
        return caption, 0.075
    segments = caption.split(" | ")
    midpoint = max(1, (len(segments) + 1) // 2)
    return (
        " | ".join(segments[:midpoint])
        + "\n"
        + " | ".join(segments[midpoint:]),
        0.115,
    )


def _save_heatmap(table, x_labels, y_labels, output_path, caption=None):
    sns.heatmap(table, xticklabels=x_labels, yticklabels=y_labels, annot=True)
    plt.xlabel("Number of Byzantine clients")
    plt.ylabel("Data heterogeneity level")

    save_kwargs = {}
    if caption:
        visible_caption, bottom_margin = _caption_for_figure(caption)
        plt.gcf().text(
            0.5,
            0.015,
            visible_caption,
            ha="center",
            va="bottom",
            fontsize=8,
        )
        plt.tight_layout(rect=(0, bottom_margin, 1, 1))
        save_kwargs["metadata"] = {"Title": caption, "Subject": caption}
    else:
        plt.tight_layout()

    plt.savefig(output_path, **save_kwargs)
    plt.close()


def _pre_aggregator_pipelines(pre_aggregators):
    if not pre_aggregators or isinstance(pre_aggregators[0], dict):
        return [pre_aggregators]
    return pre_aggregators


def _load_hyperparameters(
    path_to_results,
    dataset_name,
    model_name,
    nb_nodes,
    nb_byzantine,
    nb_declared,
    data_distribution,
    distribution_parameter,
    pre_aggregator_names,
    aggregator,
    defaults,
):
    file_name = (
        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
        f"d_{nb_declared}_{data_distribution['name']}_"
        f"{distribution_parameter}_{pre_aggregator_names}_"
        f"{aggregator['name']}.txt"
    )
    path = path_to_results / "best_hyperparameters" / "hyperparameters" / file_name
    if not path.exists():
        return defaults

    values = np.atleast_1d(np.loadtxt(path))
    return values[0], values[1], values[2]


def _selected_accuracy(
    path_to_results,
    dataset_name,
    model_name,
    nb_nodes,
    nb_byzantine,
    nb_declared,
    data_distribution,
    distribution_parameter,
    aggregator,
    pre_aggregator_names,
    attack,
    hyperparameters,
    nb_data_distribution_seeds,
    nb_training_seeds,
    training_seed,
    data_distribution_seed,
    evaluation_delta,
):
    learning_rate, momentum, weight_decay = hyperparameters
    config_name = (
        f"{dataset_name}_{model_name}_n_{nb_nodes}_f_{nb_byzantine}_"
        f"d_{nb_declared}_{data_distribution['name']}_"
        f"{distribution_parameter}_{aggregator['name']}_"
        f"{pre_aggregator_names}_{attack['name']}_"
        f"lr_{learning_rate}_mom_{momentum}_wd_{weight_decay}"
    )
    return _test_accuracy_at_best_validation(
        path_to_results,
        config_name,
        nb_data_distribution_seeds,
        nb_training_seeds,
        training_seed,
        data_distribution_seed,
        evaluation_delta,
    )


@_for_each_result_identity
def _generate_view(
    path_to_results,
    path_to_plot,
    view,
    caption=False,
    caption_fields=CAPTION_FIELDS,
    *,
    _config=None,
):
    path_to_results = Path(path_to_results)
    path_to_plot = Path(path_to_plot)
    path_to_plot.mkdir(parents=True, exist_ok=True)

    if _config is None:
        with (path_to_results / "config.json").open(encoding="utf-8") as file:
            config = json.load(file)
    else:
        config = _config

    benchmark = config["benchmark_config"]
    result_config = config["evaluation_and_results"]
    model = config["model"]
    honest_clients = config["honest_clients"]

    nb_honest_clients = ensure_list(benchmark["nb_honest_clients"])
    nb_byzantine_values = ensure_list(benchmark["f"])
    data_distributions = ensure_list(benchmark["data_distribution"])
    aggregators = ensure_list(config["aggregator"])
    attacks = ensure_list(config["attack"])
    pre_aggregator_pipelines = _pre_aggregator_pipelines(
        config.get("pre_aggregators", [])
    )
    defaults = (
        ensure_list(model["learning_rate"])[0],
        ensure_list(honest_clients["momentum"])[0],
        ensure_list(honest_clients["weight_decay"])[0],
    )
    dataset_name = model["dataset_name"]
    model_name = get_model_result_name(config)

    tolerated_values = benchmark.get("tolerated_f")
    if tolerated_values is None:
        declared_groups = [(None, nb_byzantine_values)]
    else:
        declared_groups = [
            (declared, [value for value in nb_byzantine_values if value <= declared])
            for declared in ensure_list(tolerated_values)
        ]

    if view == "per-attack":
        attack_groups = [(attack["name"], [attack]) for attack in attacks]
        aggregation_groups = [("best", aggregators)]
    elif view == "per-aggregator":
        attack_groups = [("worst", attacks)]
        aggregation_groups = [
            (aggregator["name"], [aggregator]) for aggregator in aggregators
        ]
    elif view == "best":
        attack_groups = [("worst", attacks)]
        aggregation_groups = [("best", aggregators)]
    else:
        raise ValueError(f"Unknown heatmap view: {view!r}")

    for pre_aggregators in pre_aggregator_pipelines:
        pre_aggregator_names = "_".join(
            pre_aggregator["name"] for pre_aggregator in pre_aggregators
        )
        for nb_honest in nb_honest_clients:
            for declared, actual_nb_byzantine in declared_groups:
                for data_distribution in data_distributions:
                    distribution_parameters = ensure_list(
                        data_distribution["distribution_parameter"]
                    )
                    for attack_group_name, attack_group in attack_groups:
                        for aggregation_group_name, aggregation_group in aggregation_groups:
                            table = np.zeros(
                                (len(distribution_parameters), len(actual_nb_byzantine))
                            )

                            for y, nb_byzantine in enumerate(actual_nb_byzantine):
                                nb_declared = nb_byzantine if declared is None else declared
                                nb_nodes = (
                                    nb_honest
                                    if benchmark["set_honest_clients_as_clients"]
                                    else nb_honest + nb_byzantine
                                )
                                for x, distribution_parameter in enumerate(
                                    distribution_parameters
                                ):
                                    aggregator_scores = []
                                    for aggregator in aggregation_group:
                                        hyperparameters = _load_hyperparameters(
                                            path_to_results,
                                            dataset_name,
                                            model_name,
                                            nb_nodes,
                                            nb_byzantine,
                                            nb_declared,
                                            data_distribution,
                                            distribution_parameter,
                                            pre_aggregator_names,
                                            aggregator,
                                            defaults,
                                        )
                                        attack_scores = [
                                            _selected_accuracy(
                                                path_to_results,
                                                dataset_name,
                                                model_name,
                                                nb_nodes,
                                                nb_byzantine,
                                                nb_declared,
                                                data_distribution,
                                                distribution_parameter,
                                                aggregator,
                                                pre_aggregator_names,
                                                active_attack,
                                                hyperparameters,
                                                benchmark["nb_data_distribution_seeds"],
                                                benchmark["nb_training_seeds"],
                                                benchmark["training_seed"],
                                                benchmark["data_distribution_seed"],
                                                result_config["evaluation_delta"],
                                            )
                                            for active_attack in _attacks_for_f(
                                                attack_group, nb_byzantine
                                            )
                                        ]
                                        aggregator_scores.append(min(attack_scores))

                                    table[len(table) - 1 - x][y] = max(
                                        aggregator_scores
                                    )

                            end_name = (
                                "tolerated_f_equal_real.pdf"
                                if declared is None
                                else f"tolerated_f_{declared}.pdf"
                            )
                            base_name = (
                                f"{dataset_name}_{model_name}_"
                                f"{data_distribution['name']}_{pre_aggregator_names}_"
                            )
                            if view == "best":
                                file_name = (
                                    f"best_test_{base_name}"
                                    f"nb_honest_clients_{nb_honest}_{end_name}"
                                )
                            elif view == "per-aggregator":
                                file_name = (
                                    f"test_{base_name}{aggregation_group_name}_"
                                    f"nb_honest_clients_{nb_honest}_{end_name}"
                                )
                            else:
                                file_name = (
                                    f"best_per_attack_test_{base_name}"
                                    f"{attack_group_name}_"
                                    f"nb_honest_clients_{nb_honest}_{end_name}"
                                )

                            visible_caption = None
                            if caption:
                                visible_caption = _caption_text(
                                    config,
                                    pre_aggregators,
                                    aggregation_group,
                                    attack_group,
                                    caption_fields,
                                )

                            _save_heatmap(
                                table,
                                [str(value) for value in actual_nb_byzantine],
                                [str(value) for value in reversed(distribution_parameters)],
                                path_to_plot / file_name,
                                visible_caption,
                            )


def generate_heatmaps(
    path_to_results,
    path_to_plot=None,
    views=HEATMAP_VIEWS,
    caption=False,
    caption_fields=CAPTION_FIELDS,
):
    """Generate selected heatmaps and return their PDF paths.

    ``best`` computes max over aggregators after min over attacks.
    ``per-aggregator`` computes min over attacks for each aggregator.
    ``per-attack`` computes max over aggregators for each attack.
    """
    path_to_results = Path(path_to_results).expanduser().resolve()
    selected_views = tuple(views)
    unknown_views = set(selected_views) - set(HEATMAP_VIEWS)
    if unknown_views:
        raise ValueError(f"Unknown heatmap views: {sorted(unknown_views)}")
    if not selected_views:
        raise ValueError("At least one heatmap view must be selected.")

    caption_fields = tuple(caption_fields)
    unknown_fields = set(caption_fields) - set(CAPTION_FIELDS)
    if unknown_fields:
        raise ValueError(f"Unknown caption fields: {sorted(unknown_fields)}")

    if path_to_plot is None:
        root = path_to_results / (
            "heatmaps_captioned" if caption else "heatmaps"
        )
    else:
        root = Path(path_to_plot).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    for view in selected_views:
        output = root
        if view == "per-aggregator":
            output = root / "per_aggregator"
        elif view == "per-attack":
            output = root / "per_attack"
        _generate_view(
            str(path_to_results),
            str(output),
            view,
            caption,
            caption_fields,
        )

    return sorted(root.rglob("*.pdf"))
