# CNN clipping and measurement GPU smoke test

This experiment runs `cnn_mnist` with ten honest clients, two Byzantine
clients, moving-average clipping before client momentum, NNM followed by
trimmed mean, and a sign-flipping attack.  It records honest statistics every
five rounds and verifies the generated CSV files after training.

On the EPFL server:

```bash
cd ~/byzfl
git fetch origin
git switch clip-and-measure
git pull --ff-only
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python experiments/clip_measure/run_gpu_smoke.py
```

For a shorter first check:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/clip_measure/run_gpu_smoke.py --steps 20
```

To use the second visible GPU instead:

```bash
CUDA_VISIBLE_DEVICES=0,1 python experiments/clip_measure/run_gpu_smoke.py --device cuda:1
```

The runner exits with an error when expected measurement files, rows, columns,
or finite clipping diagnostics are missing. Results are written below
`results/clip_measure_gpu_smoke/`, which should remain uncommitted.
