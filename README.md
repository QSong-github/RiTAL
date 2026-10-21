# RiTAL

A dual-encoder model for RNA–small-molecule virtual screening. An
E(n)-equivariant graph network encodes the 3-D RNA pocket, a GIN encodes the
ligand, and a cross-attention block scores the pair. Training optimises a
differentiable surrogate of rank percentile against a curriculum of
cross-pocket and chemically-similar decoys.

14.8 M parameters.

## Results

Mean rank percentile (RP) of the native ligand against the decoy pool, averaged
over folds. Per-fold numbers in [RESULTS.md](RESULTS.md).

| Benchmark | Folds | Mean RP |
|---|---|---|
| RNAmigos, 10-fold | 10 | **0.835 ± 0.044** |
| HARIBOSS, 5-fold, sequence-clustered split | 5 | **0.770 ± 0.016** |

> **Model selection caveat.** The training loop splits data into train and test
> only — there is no held-out validation set, and `best.pt` is the epoch with the
> highest RP *on the test fold*. These numbers are therefore optimistically
> biased and should not be read as unbiased generalisation estimates. Anyone
> building on this should carve a validation split out of the training folds and
> re-select. The fold assignments needed to do that are in the data pickles.

The two benchmarks are not comparable to each other: the HARIBOSS split clusters
by sequence identity so that test pockets are dissimilar to training pockets,
which is the harder setting.

## Install

```bash
conda create -n rital python=3.12
conda activate rital
pip install -e .
```

Needs PyTorch ≥ 2.1 and RDKit ≥ 2023.3; see `pyproject.toml`. A GPU is required
for training but not for inference on a single pocket.

## Data

Not included — see [data/README.md](data/README.md) for the expected layout and
upstream sources. If you already have a copy:

```bash
./scripts/link_local_data.sh /path/to/your/data
```

This symlinks the five required items into `data/`. Nothing is copied. To keep
the data elsewhere, set `RITAL_DATA`:

```bash
export RITAL_DATA=/scratch/you/rital-data
```

Configs refer to data and outputs only through `${RITAL_DATA}` and
`${RITAL_RUNS}` (default `./data` and `./runs`), so the repository contains no
machine-specific paths.

## Weights

Also not in the repository — 165 MB per checkpoint exceeds GitHub's blob limit.
See [checkpoints/README.md](checkpoints/README.md) for the layout and
`checkpoints/SHA256SUMS` to verify a download.

## Usage

Score a pocket–ligand pair:

```python
import yaml, torch
from rital import RiTALModel

cfg = yaml.safe_load(open("configs/rnamigos10fd.yaml"))
model = RiTALModel(model_cfg=cfg["model"])
sd = torch.load("checkpoints/rnamigos10fd/fold6/best.pt", map_location="cpu")
model.load_state_dict(sd["model_state_dict"], strict=True)
model.eval()
```

Check that data and model come up without training anything:

```bash
python scripts/train.py --config configs/hariboss5fd.yaml --fold 0 --dry-run
```

Train one fold:

```bash
python scripts/train.py --config configs/hariboss5fd.yaml --fold 0
```

On SLURM, one array job per fold. Cluster-specific settings are deliberately not
baked into the scripts:

```bash
sbatch -A <account> --array=0-4 scripts/slurm_train_hariboss5fd.sh
sbatch -A <account> --array=0-9 scripts/slurm_train_rnamigos10fd.sh
```

Both scripts resubmit themselves until `MAX_RESUBMITS` is reached, resuming from
`last.pt`. To stop a fold early, `touch $RITAL_RUNS/<benchmark>/fold<N>/STOP`.

Verify an unpacked checkpoint set against the architecture:

```bash
python _build/smoke_test.py
```

## Training setup

Identical for both benchmarks apart from the dataset and fold count.

- **Pocket tower** — 6-layer EGNN over the 3-D pocket graph, 80 nodes max.
- **Ligand tower** — 4-layer GIN, 64 atoms max.
- **Interaction** — 4-layer cross-attention with a distance bias, 256-dim
  normalised projections, 4 heads.
- **Objective** — soft rank-percentile surrogate as the primary term
  (`λ=2.0`, `τ=0.07`), with InfoNCE (0.1), a triplet rank loss (0.5), an
  RNA-side contrastive term (0.3) and a binding-site auxiliary (0.3, from epoch 7).
- **Decoys** — curriculum over K = 48 → 64 → 72 negatives per positive, two
  thirds cross-pocket and one third FP2-similar hard negatives. Hard negatives
  are re-mined every 3 epochs from epoch 5.
- **Margin** — 0.6 with tanh decay and restarts every 8 epochs.
- **Seed rotation** — when RP fails to improve for 3 consecutive epochs, reload
  `best.pt`, advance the random seed and restart LR warm-up. This replaces early
  stopping; total epochs are bounded by the resubmit cap instead. Per-fold
  rotation counts are in [RESULTS.md](RESULTS.md).
- Batch 8, 1500 steps/epoch, 30 epochs per job, AdamW at 5e-5 with cosine decay.

## Layout

```
src/rital/          model package
  model.py            RiTALModel — the dual encoder
  egnn.py             E(n)-equivariant pocket encoder
  gin_encoder.py      GIN / FP2 ligand encoders
  featurizers.py      pocket and ligand graph construction
  graph_data.py       batching
  paths.py            ${RITAL_DATA} / ${RITAL_RUNS} resolution
  schemas.py          RiTALSample
  legacy.py           stubs for retired RNA-FM variants
  optimol.py          stub for the optional OptiMol ligand encoder
configs/            one YAML per benchmark
scripts/
  train.py            training entry point
  _trainer_lib.py     shared routines, loaded by train.py (not run directly)
  link_local_data.sh  populate data/ with symlinks
  slurm_*.sh          SLURM array jobs with auto-resubmit
checkpoints/        released weights go here (not in git)
data/               training data goes here (not in git)
_build/             smoke test
```

## Naming

This code was developed as **BridgeBind3D** (`bridgebind3d`, `BB3D`) and renamed
for release. If you are comparing against internal notes or earlier runs:

| Before | Now |
|---|---|
| package `bridgebind3d` | package `rital` |
| `unified_model_v3.py` | `model.py` |
| `UnifiedInteractionModelV3` | `RiTALModel` |
| `UnifiedInteractionModelV4` / `V10` (stubs) | `RiTALModelRNAFM` / `RiTALModelRNAFMPlus` |
| `BridgeSample` | `RiTALSample` |

The rename does not affect existing weights: checkpoints store a `state_dict`,
whose keys follow attribute names rather than class names. All 15 released
checkpoints load with `strict=True` under the new names — `_build/smoke_test.py`
re-checks this.
