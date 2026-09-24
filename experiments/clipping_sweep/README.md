# CNN/MNIST constant-10 versus no-clipping sweep

This experiment is the first clean baseline for comparing client-side clipping
while keeping the model, data, optimization, aggregation, and attacks fixed.

It compares constant client-gradient clipping with `max_norm = 10` against no
client-gradient clipping. The configuration uses two training seeds, four
aggregators, three attacks, four data-heterogeneity levels, and `f = 0..4`.
Because attacks are deduplicated at `f = 0`, the complete sweep contains
**832 trainings**.

## Run

From the repository root:

```bash
mkdir -p results/clipping_sweep/cnn_mnist_constant10_vs_none

nohup .venv/bin/python -u experiments/clipping_sweep/run_experiment.py \
  --gpus 0,1 \
  --jobs 16 \
  > results/clipping_sweep/cnn_mnist_constant10_vs_none/run.log 2>&1 &
```

`--jobs 16` means 16 concurrent trainings in total, distributed round-robin
over the two GPUs. It does not mean 16 jobs per GPU.

Monitor the run:

```bash
tail -f results/clipping_sweep/cnn_mnist_constant10_vs_none/run.log
ps -p <PID> -o pid,etime,cmd
nvidia-smi
```

The benchmark can be restarted with the same command; completed seed runs are
skipped.

## Generate the visualizations

Wait until the log reports `All trainings finished` and
`best_hyperparameters/` exists. Then generate all captioned heatmaps:

```bash
.venv/bin/python experiments/clipping_sweep/generate_heatmaps.py \
  --views best per-aggregator per-attack \
  --caption
```

This creates:

```text
results/clipping_sweep/cnn_mnist_constant10_vs_none/heatmaps_captioned/
├── best_*.pdf
├── per_aggregator/
└── per_attack/
```

- `best`: best aggregator after considering its worst attack;
- `per-aggregator`: each aggregator under its worst attack;
- `per-attack`: best aggregator for each attack.

Each clipping policy gets separate heatmaps. A cell is mean test accuracy at
the checkpoint selected by mean validation accuracy, not the maximum test
accuracy seen during training.

Generate uncaptioned figures by omitting `--caption`. Generate a subset by
listing only the desired views, for example `--views best per-attack`.

For all configuration possibilities, clipping modes, measurement diagnostics,
run-count rules, caption fields, output options, and the Python API, see the
[benchmark guide](../../byzfl/benchmark/README.md).
