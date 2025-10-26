# infer_combo_strict.py
import os
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# ===== ENV anti-segfault (CPU safe) =====
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

# --- Repo API (folosește exact pipeline-ul tău) ---
from asvspoof.features import extract_features_for_path
from asvspoof.config import ExtractConfig, FEATURE_NAME_REVERSE_MAPPING
from asvspoof.combos import (
    group_columns_from_df,
    columns_for_code,
    normalize_codes_to_sorted_unique,
    _effective_letter_maps,
)

def load_model_robust(model_path: Path):
    print("[stage] loading model ...")
    try:
        import keras
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        m = keras.models.load_model(str(model_path), compile=False)
        print("[stage] model loaded via keras")
        return m
    except Exception as e:
        print(f"[warn] keras load failed: {type(e).__name__}: {e}")
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        m = tf.keras.models.load_model(str(model_path), compile=False)
        print("[stage] model loaded via tf.keras")
        return m

def main():
    ap = argparse.ArgumentParser(description="Inferență STRICT (features.py + combos.py) pentru un combo de litere. Model: ieșire sigmoid = p(bonafide).")
    ap.add_argument("--combo",  type=str, required=True, help="Ex: AHKMNO")
    ap.add_argument("--model",  type=str, default="final_model/best_model.keras")
    ap.add_argument("--audio",  type=str, default="final_model/fake.wav")
    ap.add_argument("--sr",     type=int, default=16000)
    ap.add_argument("--print-cols", action="store_true", help="Afișează exact coloanele folosite (în ordinea de inferență).")
    args = ap.parse_args()

    model_path  = Path(args.model)
    audio_path  = Path(args.audio)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # 1) Extrage feature-urile EXACT ca la train (features.py)
    print("[stage] extract features (features.py) ...")
    cfg = ExtractConfig(sampling_rate=args.sr)
    feats = extract_features_for_path(audio_path, cfg)  # loghează STAGE-urile

    # 2) Construiește un DataFrame cu rândul de inferență
    row = {"split":"infer","file_id":audio_path.stem,"path":str(audio_path),"label":None,"target":None}
    row.update(feats)
    df = pd.DataFrame([row])

    # 3) Rezolvă combo-ul în coloane cu combos.py (maparea efectivă litere→grup vine din config)
    groups = group_columns_from_df(df)
    forward, reverse = _effective_letter_maps()  # forward: group->letter, reverse: letter->group

    code = normalize_codes_to_sorted_unique([args.combo])
    if not code:
        raise SystemExit(f"Invalid combo code: {args.combo}")
    code = code[0]  # ex. AHKMNO

    cols = columns_for_code(code, groups)
    if not cols:
        raise SystemExit(f"No columns resolved for combo {code}. Verifică GROUP_ALIASES / FEATURES_LIST în config.")

    if args.print_cols:
        print("\n[i] Columns used (ordered):")
        for c in cols:
            print("  -", c)

    X = df[cols].to_numpy(dtype=np.float32, copy=False)

    # 4) Încarcă modelul (MLP cu ieșire sigmoid = p(bonafide))
    model = load_model_robust(model_path)

    # Verifică dimensiunea de intrare (ar TREBUI să egaleze len(cols); nu facem trim/pad ca să nu alterăm scorul)
    need = getattr(model, "inputs", [None])[0].shape[-1] if getattr(model, "inputs", None) else None
    if need is not None and X.shape[1] != int(need):
        raise SystemExit(f"[!] Feature dim mismatch: X={X.shape[1]} vs model expects {int(need)}. Combo greșit sau features order/dim schimbate.")

    # 5) Predict — SCOARDELE sunt P(bonafide) (train_cnn1d.py așa antrenează/salvează)
    #    Deci clasa 0 = spoof, clasa 1 = bonafide; p_spoof = 1 - p_bonafide.
    y_proba = model.predict(X, verbose=0).ravel()
    if y_proba.size != 1:
        raise SystemExit(f"[!] Unexpected model output shape: {y_proba.shape}. Aștept un singur scor sigmoid.")
    p_bonafide = float(y_proba[0])
    p_spoof = float(1.0 - p_bonafide)
    pred_label = 1 if p_bonafide >= 0.5 else 0  # threshold 0.5 ca la train

    # 6) Raport
    print("\n=== Rezultat inferență (bin. sigmoid) ===")
    print(f"p_spoof(0)    : {p_spoof:.6f}")
    print(f"p_bonafide(1) : {p_bonafide:.6f}")
    print(f"\nPredicție     : {'bonafide(1)' if pred_label==1 else 'spoof(0)'}  (thr=0.5)")

    # mapping litere → grup pentru combo-ul cerut (din config, nu presupuneri)
    print("\n[i] Effective letter → group mapping (from config):")
    for L in code:
        print(f"    {L} -> {reverse.get(L,'?')}")

    print(f"\n[i] Num features selected for {code}: {X.shape[1]}")

if __name__ == "__main__":
    main()
