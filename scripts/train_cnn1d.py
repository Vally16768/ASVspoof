#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, argparse, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np, pandas as pd, soundfile as sf, librosa
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Robust import for the audio loader
try:
    from scripts.audio_io import load_audio_any
except ModuleNotFoundError:
    from pathlib import Path as _P
    import sys as _sys
    _repo = _P(__file__).resolve().parents[1]
    for _p in (str(_repo), str(_repo/"scripts")):
        if _p not in _sys.path: _sys.path.insert(0, _p)
    from scripts.audio_io import load_audio_any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras import layers as L, models, callbacks

# --- Import ALL config from constants.py (single source of truth) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import (
    # paths/folders
    directory as CFG_DATA_ROOT,
    index_folder_name as INDEX_DIRNAME,
    results_folder as RESULTS_ROOT,
    # core data/sampling
    sampling_rate as SR,
    random_state as SEED,
    # run naming
    cnn1d_default_combo_name as DEFAULT_COMBO_NAME,
    # model/training defaults
    cnn1d_n_mfcc as N_MFCC,
    cnn1d_duration_seconds as DURATION_S,
    cnn1d_batch_size as BATCH_SIZE,
    cnn1d_epochs as EPOCHS,
    cnn1d_dropout1 as DROPOUT1,
    cnn1d_dropout2 as DROPOUT2,
    cnn1d_optimizer_lr as LR,
    # callback settings
    cb_model_checkpoint_monitor as CB_CKPT_MON,
    cb_early_stopping_monitor as CB_ES_MON,
    cb_early_stopping_patience as CB_ES_PATIENCE,
    cb_early_stopping_restore as CB_ES_RESTORE,
    cb_reduce_lr_monitor as CB_RLR_MON,
    cb_reduce_lr_factor as CB_RLR_FACTOR,
    cb_reduce_lr_patience as CB_RLR_PATIENCE,
    cb_reduce_lr_min_lr as CB_RLR_MIN_LR,
    # artifact filenames
    train_log_filename as TRAIN_LOG_NAME,
    best_model_filename as BEST_MODEL_NAME,
    final_model_filename as FINAL_MODEL_NAME,
    accuracy_txt_filename as ACC_TXT_NAME,
    classification_report_filename as CLFREP_NAME,
    predictions_csv_filename as PRED_CSV_NAME,
    confusion_matrix_png_filename as CM_PNG_NAME,
)

# --- helpers ---
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_index(csv_path: Path) -> List[Tuple[str, str]]:
    items = []
    if not csv_path.exists():
        return items
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        first = next(r, None)
        if first is None:
            return items
        # header-aware
        if any(h.lower() in ("path","filepath","file","relpath") for h in first):
            hdr = [h.lower() for h in first]
            for row in r:
                d = {hdr[i]: row[i] for i in range(min(len(hdr), len(row)))}
                p = d.get("path") or d.get("filepath") or d.get("file") or d.get("relpath") or row[0]
                lab = d.get("label") or (row[1] if len(row) > 1 else "")
                items.append((p, lab))
        else:
            # plain 2-column CSV
            if len(first) >= 2:
                items.append((first[0], first[1]))
            for row in r:
                if len(row) >= 2:
                    items.append((row[0], row[1]))
    return items

def load_audio(path: Path, target_sr: int):
    y, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y

def extract_mfcc(y, sr, n_mfcc, n_fft, hop_length, win_length):
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
        hop_length=hop_length, win_length=win_length,
        center=True, htk=True
    )

def fix_length(feat, T):  # (n_mfcc, T) -> (T, n_mfcc)
    n_feat, t = feat.shape
    if t < T:
        feat = np.concatenate([feat, np.zeros((n_feat, T - t), feat.dtype)], axis=1)
    elif t > T:
        feat = feat[:, :T]
    return feat.T

def to_xy(items, data_root, sr, n_mfcc, n_fft, hop, win, T):
    X, y, paths = [], [], []
    for rel, lab in items:
        fpath = (data_root / rel) if (data_root / rel).exists() else Path(rel)
        try:
            sig = load_audio_any(fpath, sr)
            mfcc = extract_mfcc(sig, sr, n_mfcc, n_fft, hop, win)
            X.append(fix_length(mfcc, T).astype("float32"))
            y.append(1 if str(lab).strip().lower() in ("bonafide", "real", "1", "genuine") else 0)
            paths.append(str(rel))
        except Exception as e:
            print(f"[!] {fpath}: {e}")
    X = np.stack(X) if X else np.zeros((0, T, n_mfcc), dtype="float32")
    y = np.array(y, dtype="int64")
    return X, y, paths

def build_model(input_shape, d1=DROPOUT1, d2=DROPOUT2, lr=LR):
    inp = L.Input(shape=input_shape)
    x = L.Conv1D(256, 3, padding="same", activation="relu")(inp)
    x = L.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = L.Dropout(d1)(x)
    x = L.MaxPooling1D(8)(x)
    x = L.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = L.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = L.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = L.Dropout(d2)(x)
    x = L.Flatten()(x)
    out = L.Dense(1, activation="sigmoid")(x)

    m = models.Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return m

def load_split(index_dir: Path):
    train = read_index(index_dir / "train.csv")
    val   = read_index(index_dir / "val.csv")
    test  = (read_index(index_dir / "test.csv")
             or read_index(index_dir / "dev.csv")
             or read_index(index_dir / "eval.csv"))
    return train, val, test

# --- training ---
def main():
    ap = argparse.ArgumentParser(description="ASVspoof LA – CNN 1D (MFCC) driven entirely by constants.py")
    ap.add_argument("--data-root",  type=str, required=False, help="Override dataset root (optional)")
    ap.add_argument("--index-dir",  type=str, required=False, help="Override index dir (optional)")
    ap.add_argument("--combo-name", type=str, required=False, help="Override run tag (optional)")
    args = ap.parse_args()

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    root_cfg = Path(CFG_DATA_ROOT).resolve()
    root = Path(args.data_root).resolve() if args.data_root else root_cfg

    alt1 = (REPO_ROOT / "data/asvspoof2019").resolve()
    alt2 = (REPO_ROOT / "database/data/asvspoof2019").resolve()
    if not (root / INDEX_DIRNAME).exists():
        if (alt1 / INDEX_DIRNAME).exists():
            root = alt1
        elif (alt2 / INDEX_DIRNAME).exists():
            root = alt2

    index_dir = Path(args.index_dir).resolve() if args.index_dir else (root / INDEX_DIRNAME)
    if not (index_dir / "train.csv").exists():
        msg = (f"[!] Missing {index_dir/'train.csv'}.\n"
               f"    Current data root: {root}\n"
               f"    Try: python scripts/make_index_from_protocols.py --data-root '{root}'")
        print(msg); sys.exit(2)

    combo_name = args.combo_name if args.combo_name else DEFAULT_COMBO_NAME

    # MFCC frame params derived from constants.SR and constants.DURATION_S
    n_fft = int(0.025 * SR)
    win   = n_fft
    hop   = int(0.010 * SR)
    T     = int(DURATION_S * SR / hop)

    # Load splits
    train, val, test = load_split(index_dir)
    if not val or not test:
        Xtmp, ytmp, _ = to_xy(train, root, SR, N_MFCC, n_fft, hop, win, T)
        if len(Xtmp) == 0:
            print("[!] Could not load any data. Check your CSVs/paths."); sys.exit(2)
        Xtr, Xte, ytr, yte = train_test_split(Xtmp, ytmp, test_size=0.2, random_state=SEED, stratify=ytmp)
        Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.1, random_state=SEED, stratify=ytr)
        test_paths = [""] * len(yte)
    else:
        Xtr, ytr, _ = to_xy(train, root, SR, N_MFCC, n_fft, hop, win, T)
        Xva, yva, _ = to_xy(val,   root, SR, N_MFCC, n_fft, hop, win, T) if val else (np.empty((0, T, N_MFCC), 'float32'), np.empty((0,), 'int64'), [])
        Xte, yte, test_paths = to_xy(test, root, SR, N_MFCC, n_fft, hop, win, T)

    model = build_model((Xtr.shape[1], Xtr.shape[2]))

    results_dir = (REPO_ROOT / RESULTS_ROOT / combo_name).resolve()
    ensure_dir(results_dir)

    cbs = [
        callbacks.ModelCheckpoint(
            filepath=str(results_dir / BEST_MODEL_NAME),
            monitor=CB_CKPT_MON, save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor=CB_ES_MON, patience=CB_ES_PATIENCE,
            restore_best_weights=CB_ES_RESTORE, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor=CB_RLR_MON, factor=CB_RLR_FACTOR,
            patience=CB_RLR_PATIENCE, min_lr=CB_RLR_MIN_LR, verbose=1
        ),
        callbacks.CSVLogger(str(results_dir / TRAIN_LOG_NAME)),
    ]

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva) if len(yva) else None,
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=cbs
    )

    # Plots
    from scripts.plot_utils import plot_history, plot_confusion
    plot_history(hist, results_dir)

    # Evaluation on test
    yprob = model.predict(Xte, batch_size=BATCH_SIZE).ravel() if len(Xte) else np.array([])
    ypred = (yprob >= 0.5).astype(int) if len(yprob) else np.array([])

    if len(yte):
        (results_dir / ACC_TXT_NAME).write_text(
            f"Test accuracy: {accuracy_score(yte, ypred):.4f}\nSamples: {len(yte)}\n"
        )
        (results_dir / CLFREP_NAME).write_text(
            classification_report(yte, ypred, target_names=["spoof", "bonafide"])
        )
        plot_confusion(
            confusion_matrix(yte, ypred, labels=[0, 1]),
            ["spoof", "bonafide"],
            results_dir / CM_PNG_NAME
        )
        pd.DataFrame({
            "path": test_paths, "label": yte, "prob_bonafide": yprob, "pred": ypred
        }).to_csv(results_dir / PRED_CSV_NAME, index=False)

    model.save(results_dir / FINAL_MODEL_NAME)
    print(f"[✓] Results saved to: {results_dir}")

if __name__ == "__main__":
    main()
