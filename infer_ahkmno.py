# infer_combo_strict.py
import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

# ===== ENV anti-segfault (CPU safe for TF ops) =====
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

# --- Repo API ---
from asvspoof.features import extract_features_for_path, load_pca_models
from asvspoof.config import ExtractConfig
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

        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        m = keras.models.load_model(str(model_path), compile=False)
        print("[stage] model loaded via keras")
        return m
    except Exception as e:
        print(f"[warn] keras load failed: {type(e).__name__}: {e}")
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        m = tf.keras.models.load_model(str(model_path), compile=False)
        print("[stage] model loaded via tf.keras")
        return m


def _load_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"[!] Failed to parse metadata file {meta_path}: {e}")


def main():
    ap = argparse.ArgumentParser(
        description="Inferență STRICT pentru combo de litere (model sigmoid -> p(bonafide))."
    )
    ap.add_argument("--combo", type=str, default="", help="Ex: AHKMNO (dacă lipsește, se ia din metadata)")
    ap.add_argument("--model", type=str, default="final_model/best_model.keras")
    ap.add_argument("--audio", type=str, default="final_model/fake.wav")
    ap.add_argument("--meta", type=str, default="final_model/metadata.json")
    ap.add_argument("--transforms-dir", type=str, default="final_model/transforms")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--print-cols", action="store_true", help="Afișează exact coloanele folosite.")
    args = ap.parse_args()

    model_path = Path(args.model)
    audio_path = Path(args.audio)
    meta_path = Path(args.meta)
    transforms_dir = Path(args.transforms_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    metadata = _load_metadata(meta_path)

    raw_code = args.combo.strip()
    if not raw_code:
        raw_code = str(metadata.get("combo", "")).strip()
    if not raw_code:
        raise SystemExit("[!] Combo not provided and no combo found in metadata.")

    code = normalize_codes_to_sorted_unique([raw_code])
    if not code:
        raise SystemExit(f"Invalid combo code: {raw_code}")
    code = code[0]

    # Resolve group subset directly from combo letters.
    _, reverse = _effective_letter_maps()
    combo_groups = []
    for ch in code:
        g = reverse.get(ch)
        if g and g not in combo_groups:
            combo_groups.append(g)

    ssl_meta = metadata.get("ssl", {}) if isinstance(metadata, dict) else {}
    base_cfg = ExtractConfig(sampling_rate=args.sr)
    cfg = ExtractConfig(
        sampling_rate=args.sr,
        ssl_wav2vec_model_id=str(ssl_meta.get("wav2vec_model_id", base_cfg.ssl_wav2vec_model_id)),
        ssl_wavlm_model_id=str(ssl_meta.get("wavlm_model_id", base_cfg.ssl_wavlm_model_id)),
        ssl_pca_components=int(ssl_meta.get("pca_components", base_cfg.ssl_pca_components)),
    )

    pca_models = load_pca_models(transforms_dir, groups=combo_groups)

    # 1) Extract features for combo groups only.
    print("[stage] extract features ...")
    feats = extract_features_for_path(audio_path, cfg, groups=combo_groups, pca_models=pca_models)

    # 2) Build one-row DataFrame.
    row = {"split": "infer", "file_id": audio_path.stem, "path": str(audio_path), "label": None, "target": None}
    row.update(feats)
    df = pd.DataFrame([row])

    # 3) Resolve combo columns via combos.py.
    groups = group_columns_from_df(df)
    _, reverse = _effective_letter_maps()
    cols = columns_for_code(code, groups)
    if not cols:
        raise SystemExit(f"No columns resolved for combo {code}. Verify GROUP_ALIASES / FEATURES_LIST in config.")

    if args.print_cols:
        print("\n[i] Columns used (ordered):")
        for c in cols:
            print("  -", c)

    X = df[cols].to_numpy(dtype=np.float32, copy=False)

    # 4) Load model.
    model = load_model_robust(model_path)

    need = getattr(model, "inputs", [None])[0].shape[-1] if getattr(model, "inputs", None) else None
    if need is not None and X.shape[1] != int(need):
        raise SystemExit(
            f"[!] Feature dim mismatch: X={X.shape[1]} vs model expects {int(need)}. "
            "Wrong combo, transforms, or feature config."
        )

    # 5) Predict: score is P(bonafide).
    y_proba = model.predict(X, verbose=0).ravel()
    if y_proba.size != 1:
        raise SystemExit(f"[!] Unexpected model output shape: {y_proba.shape}.")
    p_bonafide = float(y_proba[0])
    p_spoof = float(1.0 - p_bonafide)
    pred_label = 1 if p_bonafide >= 0.5 else 0

    # 6) Report.
    print("\n=== Rezultat inferență (bin. sigmoid) ===")
    print(f"p_spoof(0)    : {p_spoof:.6f}")
    print(f"p_bonafide(1) : {p_bonafide:.6f}")
    print(f"\nPredicție     : {'bonafide(1)' if pred_label == 1 else 'spoof(0)'}  (thr=0.5)")

    print("\n[i] Effective letter -> group mapping:")
    for L in code:
        print(f"    {L} -> {reverse.get(L, '?')}")
    print(f"\n[i] Num features selected for {code}: {X.shape[1]}")


if __name__ == "__main__":
    main()
