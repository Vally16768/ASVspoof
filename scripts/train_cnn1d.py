#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, csv, argparse, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np, pandas as pd, soundfile as sf, librosa
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from scripts.audio_io import load_audio_any

# import robust pentru loaderul audio
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

# --- import constants din rădăcina repo-ului ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from constants import directory as DEFAULT_DATA_ROOT, sampling_rate as DEFAULT_SR, random_state as DEFAULT_SEED, results_folder as RESULTS_ROOT

# --- helpers ---
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
    if sr != target_sr: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    if y.ndim > 1: y = y.mean(axis=1)
    return y

def extract_mfcc(y, sr, n_mfcc, n_fft, hop_length, win_length):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                hop_length=hop_length, win_length=win_length,
                                center=True, htk=True)

def fix_length(feat, T):  # (n_mfcc,T)->(T,n_mfcc)
    n_feat, t = feat.shape
    if t < T: feat = np.concatenate([feat, np.zeros((n_feat, T-t), feat.dtype)], axis=1)
    elif t > T: feat = feat[:, :T]
    return feat.T

def to_xy(items, data_root, sr, n_mfcc, n_fft, hop, win, T):
    X,y,paths=[],[],[]
    for rel,lab in items:
        fpath=data_root/rel
        if not fpath.exists(): fpath=Path(rel)
        try:
            sig = load_audio_any(fpath, sr)
            mfcc=extract_mfcc(sig, sr, n_mfcc, n_fft, hop, win)
            X.append(fix_length(mfcc, T).astype("float32"))
            y.append(1 if str(lab).strip().lower() in ("bonafide","real","1","genuine") else 0)
            paths.append(str(rel))
        except Exception as e:
            print(f"[!] {fpath}: {e}")
    X=np.stack(X) if X else np.zeros((0,T,n_mfcc),dtype="float32")
    y=np.array(y, dtype="int64")
    return X,y,paths

def build_model(input_shape, d1=0.1, d2=0.25):
    inp=L.Input(shape=input_shape)
    x=L.Conv1D(256,3,padding="same",activation="relu")(inp)
    x=L.Conv1D(128,3,padding="same",activation="relu")(x)
    x=L.Dropout(d1)(x)
    x=L.MaxPooling1D(8)(x)
    x=L.Conv1D(128,3,padding="same",activation="relu")(x)
    x=L.Conv1D(128,3,padding="same",activation="relu")(x)
    x=L.Conv1D(128,3,padding="same",activation="relu")(x)
    x=L.Dropout(d2)(x)
    x=L.Flatten()(x)
    out=L.Dense(1, activation="sigmoid")(x)
    m=models.Model(inp,out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return m

def load_split(index_dir: Path):
    train=read_index(index_dir/"train.csv")
    val  =read_index(index_dir/"val.csv")
    test =read_index(index_dir/"test.csv") or read_index(index_dir/"dev.csv") or read_index(index_dir/"eval.csv")
    return train,val,test

# --- training ---
def main():
    ap=argparse.ArgumentParser(description="ASVspoof LA – CNN 1D (MFCC) + salvare rezultate")
    ap.add_argument("--data-root",  type=str, default=DEFAULT_DATA_ROOT, help="Rădăcina datasetului (implic.: constants.directory)")
    ap.add_argument("--index-dir",  type=str, default=None, help="Folder CSV split (implic.: <data-root>/index)")
    ap.add_argument("--combo-name", type=str, default="A_mfcc40")
    ap.add_argument("--sr",         type=int, default=DEFAULT_SR)
    ap.add_argument("--n-mfcc",     type=int, default=40)
    ap.add_argument("--duration",   type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=200)  # 200 epoci
    ap.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    args=ap.parse_args()

    np.random.seed(args.seed); tf.random.set_seed(args.seed)

    # normalizează rădăcina: acceptă și <repo>/data/... sau <repo>/database/...
    root = Path(args.data_root).resolve()
    alt1 = (REPO_ROOT/"data/asvspoof2019").resolve()
    alt2 = (REPO_ROOT/"database/data/asvspoof2019").resolve()
    if not (root/"index").exists():
        if (alt1/"index").exists(): root = alt1
        elif (alt2/"index").exists(): root = alt2

    index_dir = Path(args.index_dir).resolve() if args.index_dir else (root/"index")
    if not (index_dir/"train.csv").exists():
        msg = f"[!] Nu găsesc {index_dir/'train.csv'}.\n" \
              f"    Current --data-root: {root}\n" \
              f"    Sugestii:\n" \
              f"      export ASVSPOOF_ROOT='{alt1}'  # dacă CSV-urile sunt în data/asvspoof2019/index\n" \
              f"      sau rulează: python scripts/make_index_from_protocols.py --data-root '{root}'"
        print(msg); sys.exit(2)

    # params MFCC
    n_fft=int(0.025*args.sr); win=n_fft; hop=int(0.010*args.sr); T=int(args.duration*args.sr/hop)

    # load data
    train,val,test=load_split(index_dir)
    if not val or not test:
        Xtmp,ytmp,_=to_xy(train, root, args.sr, args.n_mfcc, n_fft, hop, win, T)
        if len(Xtmp)==0: print("[!] N-am putut încărca datele. Verifică CSV-urile."); sys.exit(2)
        Xtr,Xte,ytr,yte=train_test_split(Xtmp,ytmp,test_size=0.2,random_state=args.seed,stratify=ytmp)
        Xtr,Xva,ytr,yva=train_test_split(Xtr,ytr,test_size=0.1,random_state=args.seed,stratify=ytr)
        test_paths=[""]*len(yte)
    else:
        Xtr,ytr,_=to_xy(train, root, args.sr, args.n_mfcc, n_fft, hop, win, T)
        Xva,yva,_=to_xy(val,   root, args.sr, args.n_mfcc, n_fft, hop, win, T) if val else (np.empty((0,T,args.n_mfcc),'float32'), np.empty((0,),'int64'),[])
        Xte,yte,test_paths=to_xy(test, root, args.sr, args.n_mfcc, n_fft, hop, win, T)

    model=build_model((Xtr.shape[1], Xtr.shape[2]))
    results_dir = (REPO_ROOT/RESULTS_ROOT/args.combo_name).resolve()
    ensure_dir(results_dir)

    cbs=[
        callbacks.ModelCheckpoint(str(results_dir/"best_model.keras"), monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        callbacks.CSVLogger(str(results_dir/"train_log.csv"))
    ]

    hist=model.fit(Xtr,ytr, validation_data=(Xva,yva) if len(yva) else None,
                   epochs=args.epochs, batch_size=args.batch_size, verbose=1, callbacks=cbs)

    # ploturi – folosim matplotlib direct aici ca să nu stricăm importurile
    from scripts.plot_utils import plot_history, plot_confusion
    plot_history(hist, results_dir)

    yprob=model.predict(Xte, batch_size=args.batch_size).ravel() if len(Xte) else np.array([])
    ypred=(yprob>=0.5).astype(int) if len(yprob) else np.array([])

    if len(yte):
        (results_dir/"accuracy.txt").write_text(f"Test accuracy: {accuracy_score(yte, ypred):.4f}\nSamples: {len(yte)}\n")
        (results_dir/"classification_report.txt").write_text(classification_report(yte, ypred, target_names=["spoof","bonafide"]))
        plot_confusion(confusion_matrix(yte, ypred, labels=[0,1]), ["spoof","bonafide"], results_dir/"confusion_matrix.png")
        pd.DataFrame({"path":test_paths,"label":yte,"prob_bonafide":yprob,"pred":ypred}).to_csv(results_dir/"predictions.csv", index=False)

    model.save(results_dir/"final_model.keras")
    print(f"[✓] Rezultatele sunt în: {results_dir}")

if __name__=="__main__":
    main()
