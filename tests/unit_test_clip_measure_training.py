"""Small end-to-end DSGD run for clipping and measurement integration."""

import csv
import importlib
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
train_module = importlib.import_module("byzfl.benchmark.train")


class TinyMNIST(Dataset):
    def __init__(self, root, train, download, transform):
        count = 12 if train else 6
        generator = torch.Generator().manual_seed(10 if train else 11)
        self.images = torch.rand(count, 1, 28, 28, generator=generator)
        self.targets = torch.arange(count) % 10
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def test_training_records_the_vectors_used_by_the_real_dsgd_path(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module.datasets, "MNIST", TinyMNIST)
    monkeypatch.setitem(
        train_module.dict_datasets,
        "mnist",
        ("MNIST", lambda value: value, lambda value: value),
    )
    config = {
        "benchmark_config": {
            "device": "cpu", "training_seed": 3, "nb_training_seeds": 1,
            "nb_workers": 2, "nb_honest_clients": 2, "f": 0, "tolerated_f": 0,
            "size_train_set": 0.75, "data_distribution_seed": 4,
            "nb_data_distribution_seeds": 1,
            "data_distribution": {"name": "iid", "distribution_parameter": 1.0},
            "training_algorithm": {"name": "DSGD", "parameters": {}},
            "nb_steps": 2,
        },
        "model": {
            "name": "fc_mnist", "dataset_name": "mnist", "nb_labels": 10,
            "loss": "NLLLoss", "optimizer_name": "SGD", "learning_rate": 0.1,
            "learning_rate_decay": 1.0, "milestones": [],
        },
        "aggregator": {"name": "Average", "parameters": {}},
        "pre_aggregators": [],
        "honest_clients": {
            "momentum": 0.5, "weight_decay": 0.0, "batch_size": 2,
            "clipping": {"name": "constant", "parameters": {"max_norm": 0.01}},
        },
        "attack": {"name": "NoAttack", "parameters": {}},
        "measurements": {
            "enabled": True, "every_n_rounds": 1,
            "honest_stages": ["raw_gradient", "clipped_gradient", "client_update"],
            "honest_metrics": ["heterogeneity", "norms", "cosine_similarity"],
            "server_metrics": ["aggregate_norm", "cosine_with_honest_mean"],
        },
        "evaluation_and_results": {
            "evaluation_delta": 1, "batch_size_evaluation": 4,
            "evaluate_on_test": False, "store_per_client_metrics": False,
            "store_models": False, "data_folder": str(tmp_path / "data"),
            "results_directory": str(tmp_path / "results"),
        },
    }

    train_module.start_training(config)

    experiment_directories = [path for path in (tmp_path / "results").iterdir() if path.is_dir()]
    assert len(experiment_directories) == 1
    measurement_directory = (
        experiment_directories[0] / "measurements_tr_seed_3_dd_seed_4"
    )
    with (measurement_directory / "honest_gradients.csv").open(newline="") as file:
        honest_rows = list(csv.DictReader(file))
    with (measurement_directory / "clipping.csv").open(newline="") as file:
        clipping_rows = list(csv.DictReader(file))
    with (measurement_directory / "server.csv").open(newline="") as file:
        server_rows = list(csv.DictReader(file))

    assert len(honest_rows) == 2 * 3
    assert len(clipping_rows) == 2 * 2
    assert len(server_rows) == 2
    assert {row["stage"] for row in honest_rows} == {
        "raw_gradient", "clipped_gradient", "client_update"
    }
    assert all(float(row["output_norm"]) <= 0.0100001 for row in clipping_rows)
    assert all(row["cosine_with_honest_mean_valid"] == "1" for row in server_rows)
