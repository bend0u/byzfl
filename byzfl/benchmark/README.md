# ByzFL benchmark: experiments and visualization

The benchmark runs a Cartesian sweep described by a JSON file, stores every
training separately, selects hyperparameters from validation accuracy, and then
generates comparable test-accuracy heatmaps.

This guide covers the complete workflow: define an experiment, estimate its
size, run or resume it, and visualize the results.

## What can be varied?

| Dimension | Configuration field | Typical use |
| --- | --- | --- |
| Training algorithm | `benchmark_config.training_algorithm` | `DSGD` or `FedAvg` |
| Model and dataset | `model.name`, `model.dataset_name` | Compare architectures or datasets |
| Byzantine clients | `benchmark_config.f` | Sweep attack strength |
| Declared tolerance | `benchmark_config.tolerated_f` | Test misspecified Byzantine counts |
| Data heterogeneity | `benchmark_config.data_distribution` | IID/non-IID comparisons |
| Aggregator | `aggregator` | Compare robust aggregation rules |
| Pre-aggregation pipeline | `pre_aggregators` | Apply transformations such as NNM then ARC |
| Attack | `attack` | Compare Byzantine strategies |
| Optimizer settings | learning rate, momentum, weight decay | Hyperparameter selection |
| Client clipping | `honest_clients.clipping` | Compare clipping policies and parameters |
| Randomness | training and data-distribution seeds | Repeat experiments |
| Diagnostics | `measurements` | Record gradient, clipping, and aggregate statistics |

A list normally creates a sweep axis. For example, four values of `f`, two
learning rates, and three attacks create `4 x 2 x 3` configurations before the
other dimensions and seeds are counted. The important exceptions are:

- `pre_aggregators` is one ordered pipeline, not a set of alternatives;
- `milestones` is one scheduler setting;
- measurement lists (`honest_stages`, `honest_metrics`, and `server_metrics`)
  select diagnostics and are not sweep axes.

Use a separate configuration and results directory to compare different
pre-aggregation pipelines. This keeps experiment identities unambiguous.

### Workers and Byzantine counts

With `set_honest_clients_as_clients: false` (the default),
`nb_honest_clients` is fixed and the total number of workers is
`nb_honest_clients + f`.

With `set_honest_clients_as_clients: true`, `nb_honest_clients` is interpreted
as the fixed total number of workers, so the number of honest workers becomes
`nb_honest_clients - f`.

If `tolerated_f` is omitted, robust methods are configured with the true `f`.
If it is supplied, the benchmark tests declared tolerances and removes cases
where `f > tolerated_f`.

At `f = 0`, attacks cannot affect training. The benchmark runs one canonical
`NoAttack` baseline instead of repeating it for every configured attack. All
clipping, aggregation, data, and seed variants still run at `f = 0`.

### Training algorithms

DSGD sends one gradient or update per client at each step:

```json
{"name": "DSGD", "parameters": {}}
```

FedAvg performs local client steps and samples a configurable client fraction:

```json
{
  "name": "FedAvg",
  "parameters": {
    "proportion_selected_clients": 0.6,
    "local_steps_per_client": 5
  }
}
```

### Client-side clipping

Clipping is applied to each honest client's raw gradient. Active client-side
clipping currently requires `DSGD`.

| Mode | Configuration | Meaning |
| --- | --- | --- |
| None | `{"name": "none", "parameters": {}}` | Leave the gradient unchanged |
| Constant | `{"name": "constant", "parameters": {"max_norm": 10.0}}` | Fixed global L2-norm cap |
| First gradient | `{"name": "first_gradient", "parameters": {"multiplier": 1.0}}` | Per-client cap calibrated from its first gradient |
| Moving average | `{"name": "moving_average", "parameters": {"window": 100, "multiplier": 1.0}}` | Per-client cap based on past raw-gradient norms |

To compare policies or parameter values in one experiment, provide a list:

```json
"clipping": [
  {"name": "constant", "parameters": {"max_norm": 10.0}},
  {"name": "none", "parameters": {}}
]
```

Each clipping mode and parameter combination receives a distinct, stable result
identity. This makes a controlled clipping comparison possible while the model
and every other setting stay unchanged.

### Optional gradient measurements

Measurements are disabled unless explicitly enabled and currently require
`DSGD`. They observe training without changing the model update.

```json
"measurements": {
  "enabled": true,
  "every_n_rounds": 10,
  "honest_stages": ["raw_gradient", "clipped_gradient", "client_update"],
  "honest_metrics": ["heterogeneity", "norms", "cosine_similarity"],
  "server_metrics": ["aggregate_norm", "cosine_with_honest_mean"]
}
```

Each measured seed pair produces:

```text
measurements_tr_seed_<training-seed>_dd_seed_<distribution-seed>/
├── clipping.csv
├── honest_gradients.csv
├── metadata.json
└── server.csv
```

Measurements can be expensive and produce substantial data. A useful workflow
is to run the full accuracy sweep without measurements, identify interesting
regions, and then run a smaller diagnostic experiment with measurements.

## Example configuration

This example compares constant clipping at 10 against no clipping on CNN/MNIST.
It sweeps four aggregators, three attacks, four heterogeneity levels, and
`f = 0..4`, using two training seeds.

```json
{
  "benchmark_config": {
    "device": "cuda",
    "training_seed": 42,
    "nb_training_seeds": 2,
    "nb_honest_clients": 10,
    "set_honest_clients_as_clients": false,
    "f": [0, 1, 2, 3, 4],
    "size_train_set": 0.8,
    "data_distribution_seed": 42,
    "nb_data_distribution_seeds": 1,
    "data_distribution": [{
      "name": "gamma_similarity_niid",
      "distribution_parameter": [1.0, 0.66, 0.33, 0.0]
    }],
    "training_algorithm": {"name": "DSGD", "parameters": {}},
    "nb_steps": 500
  },
  "model": {
    "name": "cnn_mnist",
    "is_snn": false,
    "dataset_name": "mnist",
    "nb_labels": 10,
    "loss": "NLLLoss",
    "accuracy_name": null,
    "optimizer_name": "SGD",
    "learning_rate": 0.15,
    "learning_rate_decay": 1.0,
    "milestones": []
  },
  "aggregator": [
    {"name": "GeometricMedian", "parameters": {"nu": 0.1, "T": 3}},
    {"name": "CenteredClipping", "parameters": {}},
    {"name": "TrMean", "parameters": {}},
    {"name": "MultiKrum", "parameters": {}}
  ],
  "pre_aggregators": [
    {"name": "NNM", "parameters": {}},
    {"name": "ARC", "parameters": {}}
  ],
  "honest_clients": {
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "clipping": [
      {"name": "constant", "parameters": {"max_norm": 10.0}},
      {"name": "none", "parameters": {}}
    ]
  },
  "attack": [
    {"name": "Optimal_ALittleIsEnough", "parameters": {}},
    {"name": "SignFlipping", "parameters": {}},
    {"name": "Optimal_InnerProductManipulation", "parameters": {}}
  ],
  "evaluation_and_results": {
    "evaluation_delta": 50,
    "batch_size_evaluation": 128,
    "evaluate_on_test": true,
    "store_models": false,
    "store_per_client_metrics": false,
    "data_folder": "../../data",
    "results_directory": "../../results/clipping_sweep/cnn_mnist_constant10_vs_none"
  }
}
```

The pre-aggregator list above is the ordered pipeline
`NNM -> ARC -> aggregator`.

### Estimate the number of trainings

Multiply the sizes of independent axes and both seed counts. When `f` includes
zero, use one attack at zero and every configured attack for each positive `f`:

```text
attack/f combinations = 1 + (# positive f values x # attacks)
```

For the example:

```text
2 clipping policies
+x 4 aggregators
+x 4 heterogeneity levels
+x 2 training seeds
+x 1 data-distribution seed
+x (1 + 4 x 3) attack/f combinations
+= 832 trainings
```

Learning-rate, momentum, weight-decay, model, dataset, distribution, and other
sweeps add their own multiplicative factors.

## Run an experiment

`run_benchmark` reads `config.json` from the current working directory. A
minimal launcher placed next to that file is:

```python
from byzfl.benchmark import run_benchmark


if __name__ == "__main__":
    run_benchmark(nb_jobs=16, distribute_gpus=True)
```

For CPU or one configured device, leave distribution disabled:

```python
run_benchmark(nb_jobs=4, distribute_gpus=False)
```

- `nb_jobs` is the total number of concurrent trainings, not jobs per GPU.
- With `distribute_gpus=True`, complete trainings are assigned round-robin to
  all CUDA devices visible to the process.
- Each individual training stays on one GPU.
- If fewer than two GPUs are visible, the configured device is retained.

The included clipping launcher validates that exactly two GPUs are visible and
that its intended comparison is unchanged:

```bash
.venv/bin/python experiments/clipping_sweep/run_experiment.py \
  --gpus 0,1 \
  --jobs 16
```

To run it in the background, create the log directory first because shell
redirection happens before Python starts:

```bash
mkdir -p results/clipping_sweep/cnn_mnist_constant10_vs_none

nohup .venv/bin/python -u experiments/clipping_sweep/run_experiment.py \
  --gpus 0,1 \
  --jobs 16 \
  > results/clipping_sweep/cnn_mnist_constant10_vs_none/run.log 2>&1 &
```

Monitor it with the PID printed by the shell:

```bash
tail -f results/clipping_sweep/cnn_mnist_constant10_vs_none/run.log
ps -p <PID> -o pid,etime,cmd
nvidia-smi
```

Re-running an experiment skips seed combinations with a completion marker. Use
a new results directory when materially changing a configuration; do not mix
different studies in one directory.

When all trainings finish and `size_train_set < 1`, the benchmark creates
`best_hyperparameters/`. Hyperparameters are selected from validation accuracy
under the worst configured attack. The test set is not used for selection.

## Generate heatmaps

Heatmap generation requires:

- the copied `config.json` in the results directory;
- the completed `best_hyperparameters/` selection;
- `evaluation_and_results.evaluate_on_test: true`.

Generate every view without captions:

```bash
.venv/bin/python -m byzfl.benchmark.heatmap_cli \
  --results results/clipping_sweep/cnn_mnist_constant10_vs_none \
  --views best per-aggregator per-attack
```

Generate every view with an identifying caption embedded in each PDF:

```bash
.venv/bin/python -m byzfl.benchmark.heatmap_cli \
  --results results/clipping_sweep/cnn_mnist_constant10_vs_none \
  --views best per-aggregator per-attack \
  --caption
```

Captioned and uncaptioned figures are separate products:

```text
<results>/
├── heatmaps/
│   ├── best_*.pdf
│   ├── per_aggregator/
│   └── per_attack/
└── heatmaps_captioned/
    ├── best_*.pdf
    ├── per_aggregator/
    └── per_attack/
```

The experiment-specific shortcut uses its known results directory:

```bash
.venv/bin/python experiments/clipping_sweep/generate_heatmaps.py \
  --views best per-aggregator per-attack \
  --caption
```

### Heatmap meanings

Every cell is mean test accuracy across the configured training and
data-distribution seeds, evaluated at the checkpoint with the highest mean
validation accuracy. Ties select the earliest checkpoint. A cell is not the
maximum test accuracy seen during training.

| View | Cell calculation | Question answered | Output location |
| --- | --- | --- | --- |
| `best` | Best aggregator after taking each aggregator's worst attack | What is the strongest robust pipeline available? | Heatmap root |
| `per-aggregator` | One aggregator under its worst attack | How robust is each aggregator? | `per_aggregator/` |
| `per-attack` | Best aggregator for one attack | What is achievable against each attack? | `per_attack/` |

In formulas, where `a` is an aggregator and `t` is an attack:

```text
best            = max_a min_t accuracy(a, t)
per-aggregator  = min_t accuracy(fixed a, t)
per-attack      = max_a accuracy(a, fixed t)
```

Heatmaps are generated separately for every clipping identity. Constant
clipping and no clipping are never combined into one cell.

### Select views, captions, and output

```bash
# Only robust-best heatmaps
.venv/bin/python -m byzfl.benchmark.heatmap_cli \
  --results results/my_experiment \
  --views best

# Captioned per-aggregator and per-attack heatmaps
.venv/bin/python -m byzfl.benchmark.heatmap_cli \
  --results results/my_experiment \
  --views per-aggregator per-attack \
  --caption

# A custom output directory and shorter captions
.venv/bin/python -m byzfl.benchmark.heatmap_cli \
  --results results/my_experiment \
  --output /tmp/my_heatmaps \
  --views best per-aggregator per-attack \
  --caption \
  --caption-fields model clipping pipeline
```

Default caption order is `model | clipping | pipeline | attacks`. Available
fields are `model`, `clipping`, `pipeline`, and `attacks`; they can be selected
and reordered. Captions are also stored in PDF metadata. `--caption-fields`
only matters with `--caption`.

### Python API

```python
from byzfl.benchmark import generate_heatmaps


generated_files = generate_heatmaps(
    "results/my_experiment",
    views=("best", "per-aggregator", "per-attack"),
    caption=True,
    caption_fields=("model", "clipping", "pipeline", "attacks"),
)
```

The function returns the generated PDF paths.

## Other visualizations

The existing evaluation module also exposes accuracy curves and legacy
heatmaps:

```python
from byzfl.benchmark.evaluate_results import (
    aggregated_test_heatmap,
    loss_heatmap,
    test_accuracy_curve,
    test_heatmap,
)


test_accuracy_curve("results/my_experiment", "plots/curves")
loss_heatmap("results/my_experiment", "plots/loss")
test_heatmap("results/my_experiment", "plots/test")
aggregated_test_heatmap("results/my_experiment", "plots/aggregated")
```

For new summaries, prefer `generate_heatmaps` or its CLI: they understand
clipping identities and expose robust-best, per-aggregator, and per-attack
views explicitly.

## Recommended workflow

1. Create an experiment directory containing `config.json` and a launcher.
2. Give it a new, descriptive `results_directory`.
3. Start with one seed and few steps to validate the setup.
4. Estimate the full run count before launching the sweep.
5. Run the full accuracy sweep without measurements.
6. Confirm the log ends with `All trainings finished` and that
   `best_hyperparameters/` exists.
7. Generate captioned `best`, `per-aggregator`, and `per-attack` heatmaps.
8. Run a smaller measurement-enabled follow-up where gradient-level diagnosis
   is useful.

Keep the configuration copied into each results directory. Together with
captioned figures and clipping-aware result identities, it provides the record
needed to reproduce and identify results outside the repository.
