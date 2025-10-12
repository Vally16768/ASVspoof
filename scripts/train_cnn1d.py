#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Train a tabular MLP on a materialized feature combination (NPZ) and report:
- Accuracy, Balanced Accuracy, ROC-AUC, F1 per class, EER
- min t-DCF (ASVspoof 2019, non-legacy) via tDCF_python_v2/eval_metrics.py

Outputs in results/<CODE>/:
  - metrics.json
  - tdcf_metrics.json
  - cm_scores_test.csv           (utt_id,label,p_bonafide)
  - cm_scores_eval_format.txt    (6-col format like ASVspoof CM file)
  - predictions.csv
  - classification_report.txt
  - confusion_matrix.png
  - accuracy.txt
  - train_log.csv
  - best_model.keras, final_model.keras

Strict requirements (no defaults, no skipping):
  1) Folder "<REPO_ROOT>/tDCF_python_v2" must exist and be importable.
  2) File "<REPO_ROOT>/tDCF_python_v2/scores/ASVspoof2019_LA_eval_asv_scores.txt" must exist.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# ---- NumPy compatibility shim for legacy ASVspoof scripts (np.float, np.int, etc.) ----
for _alias, _target in (("float", float), ("int", int), ("bool", bool), ("complex", complex), ("object", object)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _target)

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers as L, models, callbacks, optimizers

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# ---------- Locate repo root & import constants ----------
def _locate_repo_root(start: Path) -> Optional[Path]:
    cur = start
    for _ in range(8):
        if (cur / 'constants.py').exists():
            return cur
        cur = cur.parent
    return None

HERE = Path(__file__).resolve().parent
REPO_ROOT = _locate_repo_root(HERE)
if REPO_ROOT is None:
    raise SystemExit("[!] Could not find 'constants.py' by walking up from scripts/. Place constants.py at repo root.")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib
C = importlib.import_module('constants')

# ----- Config / filenames -----
DATA_ROOT     = Path(getattr(C, 'directory', 'database/data/asvspoof2019')).resolve()
INDEX_NAME    = getattr(C, 'index_folder_name', 'index')
RESULTS_ROOT  = Path(getattr(C, 'results_folder', 'results')).resolve()

BEST_MODEL_FILENAME   = getattr(C, 'best_model_filename', 'best_model.keras')
FINAL_MODEL_FILENAME  = getattr(C, 'final_model_filename', 'final_model.keras')
TRAIN_LOG_CSV         = getattr(C, 'train_log_filename', 'train_log.csv')
ACCURACY_TXT_FILENAME = getattr(C, 'accuracy_txt_filename', 'accuracy.txt')
CLASS_REPORT_TXT      = getattr(C, 'classification_report_filename', 'classification_report.txt')
PREDICTIONS_CSV       = getattr(C, 'predictions_csv_filename', 'predictions.csv')
CONF_MAT_PNG          = getattr(C, 'confusion_matrix_png_filename', 'confusion_matrix.png')

# MLP hyperparams
EPOCHS      = int(getattr(C, 'mlp_epochs', getattr(C, 'cnn1d_epochs', 80)))
BATCH_SIZE  = int(getattr(C, 'mlp_batch_size', getattr(C, 'cnn1d_batch_size', 64)))
LR          = float(getattr(C, 'mlp_optimizer_lr', getattr(C, 'cnn1d_optimizer_lr', 1e-3)))
DROPOUT     = float(getattr(C, 'mlp_dropout', getattr(C, 'cnn1d_dropout2', 0.20)))
RANDOM_SEED = int(getattr(C, 'random_state', 42))

CB_ES_PATIENCE   = int(getattr(C, 'cb_early_stopping_patience', 10))
CB_RLR_PATIENCE  = int(getattr(C, 'cb_reduce_lr_patience', 3))
CB_RLR_MIN_LR    = float(getattr(C, 'cb_reduce_lr_min_lr', 1e-6))

# ---------- NPZ helpers ----------
def npz_path(index_dir: Path, split: str, code: str) -> Path:
    return index_dir / 'combos' / split / f"{code}.npz"

def _load_npz(f: Path) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    if not f.exists():
        raise FileNotFoundError(f"[!] Missing NPZ: {f}")
    data = np.load(f, allow_pickle=True)
    X = data.get('X')
    y = data.get('y')
    cols = None
    if X is None or y is None:
        arrs = [data[k] for k in data.files]
        if len(arrs) >= 2:
            X, y = arrs[0], arrs[1]
        else:
            raise ValueError(f"[!] NPZ {f} doesn't contain X and y")
    if 'columns' in data.files:
        cols = list(map(str, data['columns'].tolist()))
    elif 'cols' in data.files:
        cols = list(map(str, data['cols'].tolist()))
    return X, y, cols

# ---------- Model ----------
def build_mlp(input_dim: int, dropout: float = 0.2, lr: float = 1e-3) -> tf.keras.Model:
    inp = L.Input(shape=(input_dim,), name='in')
    x = L.Normalization(name='norm')(inp)  # adapt on train only
    x = L.Dense(256, activation=None)(x); x = L.BatchNormalization()(x); x = L.Activation('relu')(x); x = L.Dropout(dropout)(x)
    x = L.Dense(128, activation=None)(x); x = L.BatchNormalization()(x); x = L.Activation('relu')(x); x = L.Dropout(dropout)(x)
    x = L.Dense(64,  activation=None)(x); x = L.BatchNormalization()(x); x = L.Activation('relu')(x); x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation='sigmoid', name='out')(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------- Metrics ----------
def compute_eer_from_scores(y_true: np.ndarray, scores_bona: np.ndarray) -> Tuple[float, float]:
    """Return (eer, threshold). y_true in {0,1}; scores are P(bonafide)."""
    fpr, tpr, thr = roc_curve(y_true, scores_bona, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float(max(fpr[idx], fnr[idx]))
    eer_thr = float(thr[idx])
    return eer, eer_thr

def save_confusion_matrix_png(cm: np.ndarray, labels: list[str], out_png: Path) -> None:
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest')
    ax.set_xticks([0,1]); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks([0,1]); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i,j]), ha='center', va='center')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

# ---------- t-DCF via eval_metrics.py (strict) ----------
def _load_asv_eval_scores(asv_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ASV EVAL scores and split into (target, nontarget, spoof) arrays.
    Expected columns: <spk> <file_id> <sys> <attk> <key> <score>
    """
    if not asv_file.exists():
        raise SystemExit(f"[!] Missing ASV score file: {asv_file}")
    data = np.genfromtxt(str(asv_file), dtype=str)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise SystemExit(f"[!] ASV score file has unexpected format: {asv_file}")
    keys = data[:, 4].astype(str)
    scores = data[:, 5].astype(float)
    tar = scores[keys == 'target']
    non = scores[keys == 'nontarget']
    spoof = scores[keys == 'spoof']
    if tar.size == 0 or non.size == 0 or spoof.size == 0:
        raise SystemExit(f"[!] ASV score file missing one of target/nontarget/spoof subsets: {asv_file}")
    return tar, non, spoof

def _load_cm_eval_scores(cm_eval_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read our CM 6-col file: <spk> <file_id> <sys> <attk> <key> <score>
    Return bona_scores (key=bonafide), spoof_scores (key=spoof).
    """
    if not cm_eval_file.exists():
        raise SystemExit(f"[!] Missing CM score file: {cm_eval_file}")
    data = np.genfromtxt(str(cm_eval_file), dtype=str)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        raise SystemExit(f"[!] CM score file has unexpected format: {cm_eval_file}")
    keys = data[:, 4].astype(str)
    scores = data[:, 5].astype(float)
    cm_bona = scores[keys == 'bonafide']
    cm_spoof = scores[keys == 'spoof']
    if cm_bona.size == 0 or cm_spoof.size == 0:
        raise SystemExit("[!] CM file missing bonafide or spoof rows.")
    return cm_bona, cm_spoof

def compute_min_tDCF_strict(cm_eval_file: Path, tdcf_root: Path) -> Dict[str, Any]:
    """
    Compute min t-DCF using tDCF_python_v2/eval_metrics.py (official math),
    with ASV file: tDCF_python_v2/scores/ASVspoof2019_LA_eval_asv_scores.txt
    """
    if not tdcf_root.exists():
        raise SystemExit(f"[!] tDCF folder missing: {tdcf_root}")
    if str(tdcf_root) not in sys.path:
        sys.path.insert(0, str(tdcf_root))
    try:
        em = importlib.import_module('eval_metrics')
    except Exception as e:
        raise SystemExit(f"[!] Could not import eval_metrics from {tdcf_root}: {e}")

    asv_file = tdcf_root / "scores" / "ASVspoof2019_LA_eval_asv_scores.txt"
    tar_asv, non_asv, spoof_asv = _load_asv_eval_scores(asv_file)
    cm_bona, cm_spoof = _load_cm_eval_scores(cm_eval_file)

    # Cost model (ASVspoof 2019, non-legacy)
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,
        'Ptar': (1 - Pspoof) * 0.99,
        'Pnon': (1 - Pspoof) * 0.01,
        'Cmiss': 1,
        'Cfa': 10,
        'Cfa_spoof': 10,
    }

    # ASV operating point: EER
    eer_asv, asv_thr = em.compute_eer(tar_asv, non_asv)
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = em.obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_thr
    )

    # CM EER (for reporting)
    eer_cm, _ = em.compute_eer(cm_bona, cm_spoof)

    # t-DCF curve & min
    tDCF_curve, CM_thresholds = em.compute_tDCF(
        cm_bona, cm_spoof,
        Pfa_asv, Pmiss_asv, Pfa_spoof_asv,
        cost_model, False
    )
    min_idx = int(np.argmin(tDCF_curve))
    min_tDCF = float(tDCF_curve[min_idx])
    cm_thr_at_min = float(CM_thresholds[min_idx])

    return {
        'asv_score_file': str(asv_file),
        'eer_asv': float(eer_asv),
        'asv_threshold': float(asv_thr),
        'Pfa_asv': float(Pfa_asv),
        'Pmiss_asv': float(Pmiss_asv),
        'Pfa_spoof_asv': float(Pfa_spoof_asv),
        'eer_cm': float(eer_cm),
        'min_tDCF': min_tDCF,
        'min_tDCF_threshold_index': min_idx,
        'cm_threshold_at_min_tDCF': cm_thr_at_min,
        'Nbona': int(cm_bona.size),
        'Nspoof': int(cm_spoof.size),
    }

# ---------- Training ----------
def train_and_eval(code: str) -> None:
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    index_dir = DATA_ROOT / INDEX_NAME
    results_dir = RESULTS_ROOT / code
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load NPZs
    tr_X, tr_y, _ = _load_npz(npz_path(index_dir, 'train', code))
    va_X, va_y, _ = _load_npz(npz_path(index_dir, 'val',   code))
    te_X, te_y, cols = _load_npz(npz_path(index_dir, 'test',  code))

    # Build model
    model = build_mlp(tr_X.shape[1], dropout=DROPOUT, lr=LR)
    # Adapt normalization to train only
    norm_layer = model.get_layer('norm')
    norm_layer.adapt(tr_X)

    # Class weights (counter imbalance)
    counts = np.bincount(tr_y.astype(int), minlength=2)
    total = counts.sum()
    class_weights = {i: float(total / (2.0 * c)) if c > 0 else 0.0 for i, c in enumerate(counts)}
    print('[*] Class weights:', class_weights)

    # Callbacks
    cbs = [
        callbacks.EarlyStopping(monitor='val_loss', patience=CB_ES_PATIENCE, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', patience=CB_RLR_PATIENCE, min_lr=CB_RLR_MIN_LR),
        callbacks.ModelCheckpoint(filepath=str(results_dir / BEST_MODEL_FILENAME),
                                  monitor='val_loss', save_best_only=True, save_weights_only=False),
        callbacks.CSVLogger(str(results_dir / TRAIN_LOG_CSV), append=False),
    ]

    # Train
    model.fit(
        tr_X, tr_y,
        validation_data=(va_X, va_y),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        class_weight=class_weights,
        callbacks=cbs
    )

    # Save final + reload best
    model.save(str(results_dir / FINAL_MODEL_FILENAME))
    best_model = tf.keras.models.load_model(str(results_dir / BEST_MODEL_FILENAME))

    # Predict on test (LA_dev)
    te_scores = best_model.predict(te_X, batch_size=BATCH_SIZE, verbose=0).ravel()  # P(bonafide)
    te_pred = (te_scores >= 0.5).astype(int)

    # Metrics (general)
    acc = float(accuracy_score(te_y, te_pred))
    bal_acc = float(balanced_accuracy_score(te_y, te_pred))
    f1_bona = float(f1_score(te_y, te_pred, pos_label=1))
    f1_spoof = float(f1_score(te_y, te_pred, pos_label=0))
    try:
        roc_auc = float(roc_auc_score(te_y, te_scores))
    except Exception:
        roc_auc = float('nan')

    eer, eer_thr = compute_eer_from_scores(te_y, te_scores)
    cls_rep = classification_report(te_y, te_pred, target_names=['spoof(0)','bonafide(1)'])
    cm = confusion_matrix(te_y, te_pred, labels=[0,1])

    # Save general artifacts
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'roc_auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_thr,
        'f1_bonafide': f1_bona,
        'f1_spoof': f1_spoof,
        'n_test': int(te_y.shape[0]),
        'n_test_pos_bonafide': int((te_y==1).sum()),
        'n_test_neg_spoof': int((te_y==0).sum()),
    }
    (results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (results_dir / ACCURACY_TXT_FILENAME).write_text(f"{acc:.6f}\n")
    (results_dir / CLASS_REPORT_TXT).write_text(cls_rep)
    save_confusion_matrix_png(cm, ['spoof','bonafide'], results_dir / CONF_MAT_PNG)

    # Build predictions dataframe
    test_csv = DATA_ROOT / INDEX_NAME / 'test.csv'
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        utt_ids = df_test['path'].map(lambda p: Path(p).stem).tolist()
        if len(utt_ids) != len(te_y):
            utt_ids = [f'utt_{i}' for i in range(len(te_y))]
    else:
        utt_ids = [f'utt_{i}' for i in range(len(te_y))]

    preds_df = pd.DataFrame({
        'utt_id': utt_ids,
        'label': te_y.astype(int),
        'p_bonafide': te_scores,
        'pred': te_pred.astype(int),
    })
    preds_df.to_csv(results_dir / PREDICTIONS_CSV, index=False)
    preds_df[['utt_id','label','p_bonafide']].to_csv(results_dir / 'cm_scores_test.csv', index=False)

    # Build CM file in 6-col *official* format for evaluator:
    # <spk> <file_id> <system_id> <attack_id> <key> <score>
    cm_eval_file = results_dir / 'cm_scores_eval_format.txt'
    with cm_eval_file.open('w', encoding='utf-8') as f:
        for uid, y, s in zip(preds_df['utt_id'], preds_df['label'], preds_df['p_bonafide']):
            key = 'bonafide' if int(y) == 1 else 'spoof'
            f.write(f"{uid} {uid} - - {key} {float(s):.8f}\n")

    # ---------- t-DCF (official math via eval_metrics, MUST succeed) ----------
    tdcf_root = REPO_ROOT / 'tDCF_python_v2'
    tdcf = compute_min_tDCF_strict(cm_eval_file, tdcf_root)
    (results_dir / 'tdcf_metrics.json').write_text(json.dumps(tdcf, indent=2))

    print('[*] t-DCF computed successfully.')
    print(f"[✓] Saved artifacts to: {results_dir}")
    out = {**metrics, **{'min_tDCF': tdcf['min_tDCF']}}
    print(json.dumps(out, indent=2))

def main():
    ap = argparse.ArgumentParser(description='Train MLP on combo NPZ and report metrics incl. EER and min t-DCF (official math).')
    ap.add_argument('--code', required=True, help='Combo code (letters), e.g., A, AH, ABC')
    args = ap.parse_args()
    code = args.code.strip().upper().replace('+','')
    train_and_eval(code)

if __name__ == '__main__':
    main()
