# ASVspoof 2019 LA — Features, Splits & Combos

This repo prepares the **ASVspoof 2019 Logical Access (LA)** dataset and extracts a rich set of classical audio features for every file. It also builds **train / val / test** splits (with **eval** kept separate) and materializes **feature combinations** as ready-to-train `.npz` blobs (X, y, columns).

> ⚠️ Scope: this repository focuses on **data prep + features + combos**. Model training/evaluation can be done in a separate repo or simple scripts that load the generated `.npz` files. (See “Training & Evaluation (optional)” below.)

## Contents

- `database/download_asvspoof_la.sh` — helper for dataset placement/unpack  
- `asvspoof_features_pipeline.py` — end-to-end features & splits pipeline  
- `Makefile` — convenience targets (`dataset`, `extract`, `combos`, `clean`, …)  
- `constants.py` — central config (paths, defaults)  
- `features_tabular/` — helper logic for tabular features / combos  
- `results/` — optional outputs/examples (safe to delete)  
- `requirements.txt` — Python deps

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make dataset   # verifies folder layout; unpacks if needed
```

Expected final layout:

```
database/data/asvspoof2019/
  ASVspoof2019_LA_train/flac/*.flac
  ASVspoof2019_LA_dev/flac/*.flac
  ASVspoof2019_LA_eval/flac/*.flac
  ASVspoof2019_LA_cm_protocols/*.txt
```

> If you already have `LA.zip`, place it in `database/` before running `make dataset`.

## Feature Extraction & Splits

Run:

```bash
# choose workers to match your CPU
make extract WORKERS=36
```

Artifacts are written to:

```
database/data/asvspoof2019/index/
  features_all.parquet
  features_all.csv
  features_meta.json
  train.csv / val.csv / test.csv   # ids & labels; eval is label-free
```

## Feature Combos → `.npz` (X, y, columns)

Materialize **all** combos (heavy: 2^15 - 1):

```bash
make combos_all
```

Or only specific codes:

```bash
# examples:
make combos codes="AB ABL M AEMNO"
```

Outputs:

```
database/data/asvspoof2019/index/combos/
  train/<CODE>.npz
  val/<CODE>.npz
  test/<CODE>.npz
```

### Feature code map

Each letter groups a family:

```
A: mfcc_mean          H: chroma_cens_mean   N: pitch_mean
B: mfcc_std           I: chroma_cqt_mean    O: pitch_std
C: spec_centroid_mean J: wavelet_mean
D: spec_bw_mean       K: chroma_stft_mean
E: spec_contrast_mean L: wavelet_std
F: spec_rolloff_mean  M: zcr_mean
G: rms_mean
```

## Training & Evaluation (optional)

This repo produces `.npz` blobs you can train with any ML framework. Minimal scikit-learn baseline:

```python
# save as scripts/train_baseline.py (example)
import sys, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import dump

combo = sys.argv[1] if len(sys.argv) > 1 else "AB"
root = Path("database/data/asvspoof2019/index/combos")
Xtr = np.load(root/"train"/f"{combo}.npz"); Xv = np.load(root/"val"/f"{combo}.npz")
clf = LogisticRegression(max_iter=1000, n_jobs=-1).fit(Xtr["X"], Xtr["y"])
proba = clf.predict_proba(Xv["X"])[:,1]
auc = roc_auc_score(Xv["y"], proba)
acc = accuracy_score(Xv["y"], (proba>=0.5).astype(int))
Path("models").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
dump(clf, f"models/{combo}.joblib")
json.dump({"combo":combo, "val_auc":auc, "val_acc":acc}, open(f"results/{combo}_metrics.json","w"), indent=2)
print(f"{combo}: AUC={auc:.4f} ACC={acc:.4f}")
```

Run it:

```bash
python scripts/train_baseline.py ABL
```

Evaluate on `test`:

```python
# scripts/eval_baseline.py (example)
import sys, json, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import load

combo = sys.argv[1] if len(sys.argv) > 1 else "ABL"
root = Path("database/data/asvspoof2019/index/combos")
Xt = np.load(root/"test"/f"{combo}.npz")
clf = load(f"models/{combo}.joblib")
proba = clf.predict_proba(Xt["X"])[:,1]
auc = roc_auc_score(Xt["y"], proba)
acc = accuracy_score(Xt["y"], (proba>=0.5).astype(int))
Path("results").mkdir(exist_ok=True)
json.dump({"combo":combo, "test_auc":auc, "test_acc":acc}, open(f"results/{combo}_test_metrics.json","w"), indent=2)
print(f"{combo} TEST: AUC={auc:.4f} ACC={acc:.4f}")
```

Run:

```bash
python scripts/eval_baseline.py ABL
```

## Make targets

```bash
make help
make dataset         # verify dataset layout (and unpack if needed)
make extract         # run features pipeline
make combos_all      # build all combos (heavy)
make combos codes="AB M ..."   # only specific ones
make clean           # remove generated artifacts (safe)
make clean_combos    # remove only combos
```

## Repro tips

- Pin your Python deps (`requirements.txt`) for stability.
- Log pipeline params in `features_meta.json`.
- Set random seeds for any stochastic step (if you add them later).

## License

For the code in this repository: MIT (or your choice). Dataset follows ASVspoof license/terms.