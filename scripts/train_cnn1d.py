#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Train a tabular MLP on a materialized feature combination (NPZ).

This version:
- Reliably imports `constants.py` by walking up from this file until it finds it,
  then adds that directory to sys.path BEFORE importing.
- Leaves the training/eval logic unchanged otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, callbacks, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------
# Locate repo root so we can import `constants.py` no matter where we run from
# ---------------------------------------------------------------------
def _locate_repo_root(start: Path, marker: str = "constants.py", max_up: int = 6) -> Path | None:
    """Walk upward from `start` for up to `max_up` levels looking for `marker`."""
    cur = start.resolve()
    for _ in range(max_up):
        candidate = cur / marker
        if candidate.exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Also try CWD (useful when launching via `python scripts/train_cnn1d.py` from repo root)
    cwd_candidate = Path.cwd() / marker
    if cwd_candidate.exists():
        return Path.cwd().resolve()
    return None

HERE = Path(__file__).resolve().parent
_repo_root = _locate_repo_root(HERE)
if _repo_root is None:
    raise SystemExit(
        "[!] Could not find 'constants.py' by walking up from scripts/.\n"
        "    Make sure 'constants.py' is at the repo root.\n"
        "    Options:\n"
        "      - Run from the repo root:  PYTHONPATH=\"$PWD\" python scripts/train_cnn1d.py --code A\n"
        "      - Or set PYTHONPATH to the repo root.\n"
    )

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ---- Project constants (single source of truth) ----
import constants as C  # noqa: E402  (import after path fix)

# ---------- IO helpers ----------
def npz_path(index_dir: Path, split: str, code: str) -> Path:
    return index_dir / "combos" / split / f"{code}.npz"

def load_combo_split(index_dir: Path, code: str, split: str):
    p = npz_path(index_dir, split, code)
    if not p.exists():
        raise SystemExit(
            f"Missing NPZ for split='{split}', code='{code}': {p}\n"
            f"Tip: materialize combos first."
        )
    data = np.load(p, allow_pickle=True)
    X = data["X"].astype("float32")
    y = data["y"].astype("int32") if "y" in data.files and data["y"].size else None
    cols = list(map(str, data["columns"])) if "columns" in data.files else []
    code_read = str(data["combo_code"]) if "combo_code" in data.files else code
    return X, y, cols, code_read

# ---------- Model ----------
def build_mlp(input_dim: int, lr: float, dropout: float):
    inp = L.Input(shape=(input_dim,), name="features")
    norm = L.Normalization(axis=-1, name="norm")
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
    # Only required argument: combo code
    ap = argparse.ArgumentParser(
        description="Train a tabular MLP on a materialized feature combination (NPZ)."
    )
    ap.add_argument(
        "--code", "-c",
        type=str,
        required=True,
        help="Combo code (e.g., A, AB, AEM, ...).",
    )
    args = ap.parse_args()
    code = args.code.upper()

    # ---- Read everything else from constants.py ----
    rng = int(getattr(C, "random_state", 42))
    tf.random.set_seed(rng)
    np.random.seed(rng)

    data_root    = Path(getattr(C, "directory")).resolve()
    index_dir    = data_root / getattr(C, "index_folder_name", "index")
    results_root = Path(getattr(C, "results_folder", "results")).resolve()

    # Hyperparameters (prefer MLP-specific if present; otherwise reuse cnn1d ones)
    epochs     = int(getattr(C, "mlp_epochs", getattr(C, "cnn1d_epochs", 100)))
    batch_size = int(getattr(C, "mlp_batch_size", getattr(C, "cnn1d_batch_size", 64)))
    lr         = float(getattr(C, "mlp_optimizer_lr", getattr(C, "cnn1d_optimizer_lr", 1e-3)))
    dropout    = float(getattr(C, "mlp_dropout", getattr(C, "cnn1d_dropout2", 0.20)))

    # Filenames from constants
    fname_best   = getattr(C, "best_model_filename", "best_model.keras")
    fname_final  = getattr(C, "final_model_filename", "final_model.keras")
    fname_log    = getattr(C, "train_log_filename", "train_log.csv")
    fname_acc    = getattr(C, "accuracy_txt_filename", "accuracy.txt")
    fname_clfrep = getattr(C, "classification_report_filename", "classification_report.txt")
    fname_preds  = getattr(C, "predictions_csv_filename", "predictions.csv")
    fname_cm_png = getattr(C, "confusion_matrix_png_filename", "confusion_matrix.png")

    # Callback settings
    ckpt_monitor = getattr(C, "cb_model_checkpoint_monitor", "val_loss")
    es_monitor   = getattr(C, "cb_early_stopping_monitor", "val_loss")
    es_patience  = int(getattr(C, "cb_early_stopping_patience", 10))
    es_restore   = bool(getattr(C, "cb_early_stopping_restore", True))
    rlrop_monitor= getattr(C, "cb_reduce_lr_monitor", "val_loss")
    rlrop_factor = float(getattr(C, "cb_reduce_lr_factor", 0.5))
    rlrop_pat    = int(getattr(C, "cb_reduce_lr_patience", 3))
    rlrop_min_lr = float(getattr(C, "cb_reduce_lr_min_lr", 1e-6))

    # Load splits from NPZs
    Xtr, ytr, cols, code_read = load_combo_split(index_dir, code, "train")
    Xva, yva, _, _            = load_combo_split(index_dir, code, "val")
    Xte, yte, _, _            = load_combo_split(index_dir, code, "test")
    if ytr is None or yva is None or yte is None:
        raise SystemExit("This trainer expects labeled splits (train/val/test). Got unlabeled data.")

    # Build & adapt model
    model, norm = build_mlp(Xtr.shape[1], lr=lr, dropout=dropout)
    norm.adapt(Xtr)

    # Results dir
    results_dir = (results_root / f"combo_{code_read}").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    cbs = [
        callbacks.ModelCheckpoint(
            filepath=str(results_dir / fname_best),
            monitor=ckpt_monitor, save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor=es_monitor, patience=es_patience,
            restore_best_weights=es_restore, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor=rlrop_monitor, factor=rlrop_factor,
            patience=rlrop_pat, min_lr=rlrop_min_lr, verbose=1
        ),
        callbacks.CSVLogger(str(results_dir / fname_log)),
    ]

    # Train
    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=cbs,
    )

    # Evaluate
    prob_te = model.predict(Xte, batch_size=batch_size).ravel()
    ypred   = (prob_te >= 0.5).astype("int32")
    acc = float(accuracy_score(yte, ypred))
    try:
        auc = float(roc_auc_score(yte, prob_te))
    except Exception:
        auc = float("nan")

    # Write artifacts
    (results_dir / fname_acc).write_text(f"Test accuracy: {acc:.4f}\nAUC: {auc:.4f}\nN={len(yte)}\n")
    (results_dir / fname_clfrep).write_text(
        classification_report(yte, ypred, target_names=["spoof", "bonafide"])
    )

    # Confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        cm = confusion_matrix(yte, ypred, labels=[0, 1])
        fig = plt.figure(figsize=(4, 4), dpi=120)
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix — {code_read}")
        plt.xticks([0, 1], ["spoof", "bonafide"])
        plt.yticks([0, 1], ["spoof", "bonafide"])
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.tight_layout()
        fig.savefig(results_dir / fname_cm_png)
        plt.close(fig)
    except Exception:
        pass

    # Predictions CSV (labels + proba)
    np.savetxt(
        results_dir / fname_preds,
        np.c_[prob_te, ypred, yte],
        delimiter=",",
        header="prob_bonafide,pred,label",
        comments="",
        fmt="%.6f",
    )

    # Save final model
    model.save(results_dir / fname_final)

    # Meta
    meta = {
        "code": code_read,
        "num_features": int(Xtr.shape[1]),
        "columns": cols,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "dropout": dropout,
        "random_state": rng,
        "data_root": str(data_root),
        "index_dir": str(index_dir),
        "results_dir": str(results_dir),
        "repo_root": str(_repo_root),
    }
    (results_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[✓] Saved artifacts to: {results_dir}")

if __name__ == "__main__":
    main()
