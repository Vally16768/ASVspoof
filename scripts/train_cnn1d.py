#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, callbacks, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# ---- project config (your defaults) ----
try:
    # If this file lives inside the same package as cli.py
    from .config import DEFAULTS
except Exception:
    # Fallback for running as a standalone script
    from config import DEFAULTS  # noqa: F401

# ---------- IO helpers ----------
def npz_path(out_dir: Path, split: str, code: str) -> Path:
    return out_dir / "combos" / split / f"{code}.npz"

def load_combo_split(out_dir: Path, code: str, split: str):
    p = npz_path(out_dir, split, code)
    if not p.exists():
        raise SystemExit(
            f"Missing NPZ for split='{split}', code='{code}': {p}\n"
            f"Tip: materialize combos first, e.g.:\n"
            f"  python -m cli combos --data-root '{out_dir.parent}' --codes {code}"
        )
    data = np.load(p, allow_pickle=True)
    X = data["X"].astype("float32")
    y = data["y"].astype("int32") if "y" in data.files and data["y"].size else None
    cols = list(map(str, data["columns"])) if "columns" in data.files else []
    code_read = str(data["combo_code"]) if "combo_code" in data.files else code
    return X, y, cols, code_read

# ---------- Model ----------
def build_mlp(input_dim: int, lr: float = 1e-3, dropout: float = 0.2):
    inp = L.Input(shape=(input_dim,), name="features")

    # Normalize inside the graph so the layer is saved with the model
    norm = L.Normalization(axis=-1, name="norm")
    # Will be adapted on training data outside
    x = norm(inp)

    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="sigmoid", name="prob_bonafide")(x)

    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return m, norm

# ---------- Training ----------
def main():
    ap = argparse.ArgumentParser(description="Train a tabular MLP on a materialized feature combination (NPZ).")
    ap.add_argument("--data-root", type=str, default="database/data/asvspoof2019",
                    help="ASVspoof2019 root. Combos live in <data-root>/index/combos/.")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Override index dir (defaults to <data-root>/index).")
    ap.add_argument("--code", type=str, required=True,
                    help="Combo code (e.g., A, AB, AEM, ...).")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--results-root", type=str, default="results",
                    help="Folder to save run artifacts.")
    args = ap.parse_args()

    tf.random.set_seed(DEFAULTS["random_state"])
    np.random.seed(DEFAULTS["random_state"])

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (data_root / "index")
    code = args.code.upper()

    # Load splits from NPZs produced by `cli.py combos`
    Xtr, ytr, cols, code_read = load_combo_split(out_dir, code, "train")
    Xva, yva, _, _            = load_combo_split(out_dir, code, "val")
    Xte, yte, _, _            = load_combo_split(out_dir, code, "test")

    if ytr is None or yva is None or yte is None:
        raise SystemExit("This trainer expects labeled splits (train/val/test). Got unlabeled data.")

    # Build model
    model, norm = build_mlp(Xtr.shape[1], lr=args.lr, dropout=args.dropout)
    # Adapt normalization on training data only
    norm.adapt(Xtr)

    # Results dir
    results_dir = (Path(args.results_root) / f"combo_{code_read}").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=str(results_dir / "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger(str(results_dir / "train_log.csv")),
    ]

    # Train
    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=cbs,
    )

    # Evaluate
    prob_te = model.predict(Xte, batch_size=args.batch_size).ravel()
    ypred   = (prob_te >= 0.5).astype("int32")

    # Metrics
    acc = float(accuracy_score(yte, ypred))
    try:
        auc = float(roc_auc_score(yte, prob_te))
    except Exception:
        auc = float("nan")

    # Write artifacts
    (results_dir / "accuracy.txt").write_text(f"Test accuracy: {acc:.4f}\nAUC: {auc:.4f}\nN={len(yte)}\n")
    (results_dir / "classification_report.txt").write_text(
        classification_report(yte, ypred, target_names=["spoof", "bonafide"])
    )

    # Confusion matrix plot (small helper)
    try:
        import matplotlib.pyplot as plt
        cm = confusion_matrix(yte, ypred, labels=[0, 1])
        fig = plt.figure(figsize=(4, 4), dpi=120)
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix — {code_read}")
        plt.xticks([0,1], ["spoof","bonafide"])
        plt.yticks([0,1], ["spoof","bonafide"])
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.tight_layout()
        fig.savefig(results_dir / "confusion_matrix.png")
        plt.close(fig)
    except Exception:
        pass

    # Predictions CSV (path not stored in NPZ; we output just labels + proba)
    np.savetxt(results_dir / "predictions.csv",
               np.c_[prob_te, ypred, yte],
               delimiter=",", header="prob_bonafide,pred,label", comments="", fmt="%.6f")

    # Save final model
    model.save(results_dir / "final_model.keras")

    # Meta (document columns used)
    meta = {
        "code": code_read,
        "num_features": int(Xtr.shape[1]),
        "columns": cols,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "random_state": DEFAULTS["random_state"],
        "data_root": str(data_root),
        "index_dir": str(out_dir),
        "results_dir": str(results_dir),
    }
    (results_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[✓] Saved artifacts to: {results_dir}")

if __name__ == "__main__":
    main()
