#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, argparse, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np, pandas as pd, soundfile as sf, librosa
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras import layers as L, models, callbacks

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from constants import (
    directory as DEFAULT_DATA_ROOT,
    sampling_rate as DEFAULT_SR,
    random_state as DEFAULT_SEED,
    results_folder as RESULTS_ROOT,
)
from scripts.plot_utils import plot_history, plot_confusion

# ----------------- I/O helpers -----------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_index(csv_path: Path) -> List[Tuple[str, str]]:
    items=[]
    if not csv_path.exists(): return items
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        r=csv.reader(f); first=next(r, None)
        if first is None: return items
        if any(h.lower() in ("path","filepath","file","relpath") for h in first):
            hdr=[h.lower() for h in first]
            for row in r:
                d={hdr[i]:row[i] for i in range(min(len(hdr),len(row)))}
                p=d.get("path") or d.get("filepath") or d.get("file") or d.get("relpath") or row[0]
                lab=d.get("label") or (row[1] if len(row)>1 else "")
                items.append((p,lab))
        else:
            if len(first)>=2: items.append((first[0], first[1]))
            for row in r:
                if len(row)>=2: items.append((row[0],row[1]))
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
    )  # (n_mfcc, T)

def fix_length(feat, target_frames):  # (n_mfcc,T)->(T,n_mfcc)
    n_feat, T = feat.shape
    if T < target_frames:
        feat = np.concatenate([feat, np.zeros((n_feat, target_frames-T), feat.dtype)], axis=1)
    elif T > target_frames:
        feat = feat[:, :target_frames]
    return feat.T

def to_xy(items, data_root, sr, n_mfcc, n_fft, hop, win, T):
    X, y, paths = [], [], []
    for rel, lab in items:
        fpath = data_root / rel
        if not fpath.exists():
            fpath = Path(rel)  # acceptă căi absolute în CSV
        try:
            sig = load_audio(fpath, sr)
            mfcc = extract_mfcc(sig, sr, n_mfcc, n_fft, hop, win)
            X.append(fix_length(mfcc, T).astype("float32"))
            y.append(1 if str(lab).strip().lower() in ("bonafide","real","1","genuine") else 0)
            paths.append(str(rel))
        except Exception as e:
            print(f"[!] {fpath}: {e}")
    X = np.stack(X) if X else np.zeros((0, T, n_mfcc), dtype="float32")
    y = np.array(y, dtype="int64")
    return X, y, paths

# ----------------- Model -----------------
def build_model(input_shape, d1=0.1, d2=0.25):
    x_in = L.Input(shape=input_shape)
    x = L.Conv1D(256, 3, padding="same", activation="relu")(x_in)
    x = L.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = L.Dropout(d1)(x)
    x = L.MaxPooling1D(8)(x)
    x = L.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = L.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = L.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = L.Dropout(d2)(x)
    x = L.Flatten()(x)
    out = L.Dense(1, activation="sigmoid")(x)
    m = models.Model(x_in, out, name="cnn1d_asvspoof")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return m

def load_split(index_dir: Path):
    train = read_index(index_dir/"train.csv")
    val   = read_index(index_dir/"val.csv")
    test  = read_index(index_dir/"test.csv") or read_index(index_dir/"dev.csv") or read_index(index_dir/"eval.csv")
    return train, val, test

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="ASVspoof LA – CNN 1D (MFCC) + salvare rezultate")
    ap.add_argument("--data-root",  type=str, default=DEFAULT_DATA_ROOT, help="Rădăcina datasetului (implic.: constants.directory)")
    ap.add_argument("--index-dir",  type=str, default=None, help="Folderul cu CSV-urile de split (implic.: <data-root>/index)")
    ap.add_argument("--combo-name", dest="combo_name", type=str, default="A_mfcc40", help="Numele combinației (folosit în results/<combo>)")
    ap.add_argument("--sr",         type=int, default=DEFAULT_SR)
    ap.add_argument("--n-mfcc",     type=int, default=40)
    ap.add_argument("--duration",   type=float, default=4.0, help="secunde (se taie/padează la această durată)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=200)  # 200 epoci
    ap.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_root = Path(args.data_root)
    index_dir = Path(args.index_dir) if args.index_dir else (data_root / "index")

    # 25ms/10ms la 16kHz
    n_fft = int(0.025 * args.sr)
    win   = n_fft
    hop   = int(0.010 * args.sr)
    T     = int(args.duration * args.sr / hop)

    results_dir = Path(RESULTS_ROOT) / args.combo_name
    ensure_dir(results_dir)

    # Încărcare split
    train, val, test = load_split(index_dir)
    if not train:
        print(f"[!] Lipsesc CSV-urile de index ({index_dir}/train.csv).")
        sys.exit(2)

    # Dacă nu avem val/test, split din train
    if not val or not test:
        Xtmp, ytmp, _ = to_xy(train, data_root, args.sr, args.n_mfcc, n_fft, hop, win, T)
        if len(Xtmp) == 0:
            print("[!] N-am putut încărca datele. Verifică căile din CSV.")
            sys.exit(2)
        Xtr, Xte, ytr, yte = train_test_split(Xtmp, ytmp, test_size=0.2, random_state=args.seed, stratify=ytmp)
        Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.1, random_state=args.seed, stratify=ytr)
        test_paths = [""] * len(yte)
    else:
        Xtr, ytr, _ = to_xy(train, data_root, args.sr, args.n_mfcc, n_fft, hop, win, T)
        Xva, yva, _ = to_xy(val,   data_root, args.sr, args.n_mfcc, n_fft, hop, win, T) if val else \
                      (np.empty((0,T,args.n_mfcc),'float32'), np.empty((0,),'int64'), [])
        Xte, yte, test_paths = to_xy(test,  data_root, args.sr, args.n_mfcc, n_fft, hop, win, T)

    # Model + callbacks (EarlyStopping + ReduceLROnPlateau + ModelCheckpoint + CSVLogger)
    model = build_model((Xtr.shape[1], Xtr.shape[2]))
    ckpt  = results_dir / "best_model.keras"
    log   = results_dir / "train_log.csv"
    cbs = [
        callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        callbacks.CSVLogger(str(log))
    ]

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva) if len(yva) else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=cbs
    )

    # Plot-uri (apel către modulul dedicat)
    plot_history(history, results_dir)

    # Test
    yprob = model.predict(Xte, batch_size=args.batch_size).ravel() if len(Xte) else np.array([])
    ypred = (yprob >= 0.5).astype(int) if len(yprob) else np.array([])

    if len(yte):
        acc = accuracy_score(yte, ypred)
        (results_dir / "accuracy.txt").write_text(f"Test accuracy: {acc:.4f}\nSamples: {len(yte)}\n")
        (results_dir / "classification_report.txt").write_text(
            classification_report(yte, ypred, target_names=["spoof","bonafide"])
        )
        cm = confusion_matrix(yte, ypred, labels=[0,1])
        plot_confusion(cm, ["spoof","bonafide"], results_dir / "confusion_matrix.png")

        pd.DataFrame({
            "path": test_paths, "label": yte,
            "prob_bonafide": yprob, "pred": ypred
        }).to_csv(results_dir / "predictions.csv", index=False)

    model.save(results_dir / "final_model.keras")
    print(f"[✓] Rezultatele sunt în: {results_dir}")

if __name__ == "__main__":
    main()
