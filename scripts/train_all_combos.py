#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def load_npz(p: Path):
    d = np.load(p, allow_pickle=True)
    X, y = d["X"], d["y"]
    return X, y

def train_all(root: Path, out_csv: Path, max_iter=200, batch_size=256, seed=42):
    tr = root/"index"/"combos"/"train"
    va = root/"index"/"combos"/"val"
    te = root/"index"/"combos"/"test"
    if not tr.exists(): raise SystemExit(f"Missing combos at {tr}. Run make combos_all first.")
    codes = sorted(p.stem for p in tr.glob("*.npz"))
    rows = []; out_csv.parent.mkdir(parents=True, exist_ok=True)
    for i, code in enumerate(codes, 1):
        try:
            Xtr, ytr = load_npz(tr/f"{code}.npz")
            Xva = yva = None
            if va.exists() and (va/f"{code}.npz").exists():
                Xva, yva = load_npz(va/f"{code}.npz")
            else:
                Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.1, random_state=seed, stratify=ytr)
            if te.exists() and (te/f"{code}.npz").exists():
                Xte, yte = load_npz(te/f"{code}.npz")
            else:
                Xte, yte = Xva, yva

            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(256,128,64),
                              activation="relu", solver="adam",
                              batch_size=batch_size, max_iter=max_iter,
                              random_state=seed, early_stopping=True,
                              n_iter_no_change=10, validation_fraction=0.1 if Xva is None else 0.0,
                              verbose=False)
            )
            pipe.fit(Xtr, ytr)
            acc = accuracy_score(yte, pipe.predict(Xte)) if yte is not None else np.nan
            rows.append({"combo": code, "accuracy": acc, "n_train": len(ytr), "n_test": (len(yte) if yte is not None else 0)})
        except Exception as e:
            rows.append({"combo": code, "accuracy": np.nan, "n_train": 0, "n_test": 0, "error": str(e)})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[{i}/{len(codes)}] {code} done")
    print(f"[✓] wrote {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="results/combos_accuracy.csv")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    train_all(Path(a.data_root), Path(a.out), a.max_iter, a.batch_size, a.seed)

if __name__ == "__main__":
    main()
