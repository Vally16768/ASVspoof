# ASVspoof 2019 LA — Features, Splits & Combos (updated)

**Purpose**

This repository prepares the ASVspoof 2019 Logical Access (LA) dataset and extracts engineered audio features for each file, builds train/val/test splits, and materializes feature combinations as `.npz` blobs (ready-to-train `X, y, columns`). Model training is intentionally out of scope — this repo focuses on *data preparation*.

**Layout**

```
.
├── asvspoofy
│   └──    # main CLI: build index, extract features, splits, combos
├── constants.py                    # single source of repo configuration (paths, defaults)
├── Makefile                         # convenience targets to run common tasks
├── database/
│   └── download_asvspoof_la.sh
├── features_tabular/                # helpers for tabular feature extraction & combos
├── scripts/
├── results/                         # optional examples (safe to delete)
└── requirements.txt
```

**Important:** `constants.py` is the single canonical configuration file. All scripts should read configuration from `constants.py` (e.g. `from constants import DATA_ROOT, INDEX_DIRNAME`) — do not duplicate constants across scripts.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make dataset           # verify dataset layout and unpack if needed
make extract WORKERS=8 # choose workers according to your CPU
```

Expected final layout (relative to repo root):

```
database/data/asvspoof2019/
  ASVspoof2019_LA_train/flac/*.flac
  ASVspoof2019_LA_dev/flac/*.flac
  ASVspoof2019_LA_eval/flac/*.flac
  ASVspoof2019_LA_cm_protocols/*.txt
```

## Steps & targets

- `make check` — run a small script that verifies `constants.py` paths and dataset layout.
- `make extract` — run `asvspoof_features_pipeline.py` to extract features and produce `features_all.parquet` and index files.
- `make combos` — build selected combos, e.g. `make combos codes="AB M"`
- `make combos_all` — materialize all combos (heavy: 2^N - 1).
- `make clean` — remove generated artifacts.

## Usage examples

Minimal scikit-learn baseline (example):

```bash
python scripts/train_baseline.py ABL
python scripts/eval_baseline.py ABL
```

## Developer notes

- Avoid embedding Python evaluation into the Makefile. Use `Makefile` variables with `?=` defaults and allow overrides via environment variables or make arguments.
- Prefer package-style imports: make code importable as `import asvspoof.features` or `from constants import DATA_ROOT`.
- Add smoke tests to run in CI: `make check` should run quickly and fail fast if dataset layout or critical constants are missing.