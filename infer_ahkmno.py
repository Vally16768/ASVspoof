# infer_combo.py
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ENV anti-segfault (sigur pe CPU)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

# === proiect: folosim exact pipeline-ul tău ===
from asvspoof.features import extract_features_for_path, _load_audio_strict
from asvspoof.config import ExtractConfig
from asvspoof.combos import (
    group_columns_from_df,
    columns_for_code,
    normalize_codes_to_sorted_unique,
    _effective_letter_maps,  # ca să raportăm mappingul efectiv
)

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def try_load_scaler(path: Path):
    try:
        import joblib
        if path.exists():
            return joblib.load(path)
    except Exception:
        return None
    return None

def load_model_robust(model_path: Path):
    print("[stage] loading model ...")
    try:
        import keras
        # oprește GPU dacă există, și limitează thread-urile
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
        m = keras.models.load_model(str(model_path), compile=False)
        print("[stage] model loaded via keras")
        return m
    except Exception as e:
        print(f"[warn] keras load failed: {type(e).__name__}: {e}")
        import tensorflow as tf
        try:
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
        m = tf.keras.models.load_model(str(model_path), compile=False)
        print("[stage] model loaded via tf.keras")
        return m

def main():
    ap = argparse.ArgumentParser(description="Inferență cu combo pe litere (folosind combos.py).")
    ap.add_argument("--combo",  type=str, required=True, help="Ex: AHKMNO")
    ap.add_argument("--model",  type=str, default="final_model/best_model.keras")
    ap.add_argument("--audio",  type=str, default="final_model/fake.wav")
    ap.add_argument("--labels", type=str, default="")
    ap.add_argument("--sr",     type=int, default=16000)
    ap.add_argument("--scaler", type=str, default="final_model/scaler.pkl")
    args = ap.parse_args()

    model_path  = Path(args.model)
    audio_path  = Path(args.audio)
    labels_path = Path(args.labels) if args.labels else None
    scaler_path = Path(args.scaler) if args.scaler else None

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    print("[stage] extract features (features.py) ...")
    cfg = ExtractConfig(sampling_rate=args.sr)  # folosește valorile default din config (window_length_ms, fmax, n_mels etc.)
    # Folosim funcția oficială, care reîncarcă audio din path cu _load_audio_strict, ca la train
    feats = extract_features_for_path(audio_path, cfg)

    # Construim un DataFrame cu ordinea corectă a coloanelor (ca la train)
    row = {
        "split": "infer",
        "file_id": audio_path.stem,
        "path": str(audio_path),
        "label": None,
        "target": None,
    }
    row.update(feats)
    df = pd.DataFrame([row])

    # Harta de grupuri -> coloane și maparea efectivă literă->grup
    groups = group_columns_from_df(df)
    forward, reverse = _effective_letter_maps()  # forward: group->letter, reverse: letter->group

    # Normalizează și validează codul
    code = normalize_codes_to_sorted_unique([args.combo])
    if not code:
        raise SystemExit(f"Invalid combo code: {args.combo}")
    code = code[0]  # codul sortat (ex: AHKMNO -> AHKMNO)

    # Coloanele exacte pentru combo (în ordinea definită de combos.py)
    try:
        cols = columns_for_code(code, groups)
    except KeyError as e:
        print("[error] Unknown letter in combo. Effective mapping was:")
        for L in sorted(reverse.keys()):
            print(f"  {L} -> {reverse[L]}")
        raise

    if not cols:
        raise SystemExit(f"No columns resolved for combo {code} — check GROUP_ALIASES / FEATURES_LIST.")

    # X: selectează din df în ordinea corectă
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    # (opțional) scaler
    if scaler_path and scaler_path.exists():
        print("[stage] apply scaler ...")
        scaler = try_load_scaler(scaler_path)
        if scaler is not None:
            X = scaler.transform(X)

    # Load model + predict
    model = load_model_robust(model_path)
    need = getattr(model, "inputs", [None])[0].shape[-1] if getattr(model, "inputs", None) else None
    if need is not None and X.shape[1] != int(need):
        print(f"[warn] feature dim mismatch: X={X.shape[1]} vs model expects {int(need)} — trimming/padding.")
        v = X[0]
        need = int(need)
        if v.size > need:
            v = v[:need]
        elif v.size < need:
            v = np.pad(v, (0, need - v.size))
        X = v[None, :]

    preds = model.predict(X, verbose=0)
    preds = np.array(preds)
    if preds.ndim == 1:
        preds = preds[None, :]
    probs = softmax(preds)

    # labels (opțional)
    if labels_path and labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        if len(labels) == probs.shape[1]:
            pairs = sorted(zip(labels, probs[0].tolist()), key=lambda t: t[1], reverse=True)
            print("\n=== Rezultat inferență (etichetă : p) ===")
            for lab, p_ in pairs:
                print(f"{lab:>20s} : {p_:.4f}")
            print(f"\nPredicție: {pairs[0][0]} (p={pairs[0][1]:.4f})")
        else:
            print("[warn] labels size mismatch -> indici:")
            for i, p_ in enumerate(probs[0]):
                print(f"{i:>3d} : {p_:.4f}")
            top = int(np.argmax(probs[0]))
            print(f"\nPredicție: clasa {top} (p={probs[0, top]:.4f})")
    else:
        print("\n=== Rezultat inferență (index : p) ===")
        for i, p_ in enumerate(probs[0]):
            print(f"{i:>3d} : {p_:.4f}")
        top = int(np.argmax(probs[0]))
        print(f"\nPredicție: clasa {top} (p={probs[0, top]:.4f})")

    # Raport mapping efectiv pentru literele DIN combo-ul curent
    print("\n[i] Effective letter → group mapping (for this run):")
    for L in code:
        g = reverse.get(L, "?")
        print(f"    {L} -> {g}")

    print(f"\n[i] Num features selected for {code}: {X.shape[1]}")

if __name__ == "__main__":
    main()
