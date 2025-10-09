# === features.py (extract features only; no splitting) ===
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import ExtractConfig


def _lazy_import():
    import soundfile as sf
    import librosa
    import pywt
    return sf, librosa, pywt


def _frame_params(sr: int, window_length_ms: float) -> Tuple[int, int]:
    n_fft = int(round(sr * window_length_ms / 1000.0))
    n_fft_pow2 = 1 << (n_fft - 1).bit_length()
    hop = max(1, n_fft_pow2 // 4)
    return n_fft_pow2, hop


def extract_features_for_path(path: Path, cfg: ExtractConfig) -> Dict[str, float]:
    sf, librosa, pywt = _lazy_import()

    # read wav/flac robustly
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if hasattr(y, "ndim") and y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception:
        y, sr = librosa.load(str(path), sr=None, mono=True)

    if sr != cfg.sampling_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sampling_rate)
        sr = cfg.sampling_rate

    # trim silence; ensure non-empty
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) == 0:
        y = np.zeros(sr // 2, dtype=np.float32)

    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    feats: Dict[str, float] = {}

    # scalar features
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    feats["zcr_mean"] = float(np.mean(zcr))

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    feats["rms_mean"] = float(np.mean(rms))

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    feats["spec_centroid_mean"] = float(np.mean(spec_centroid))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    feats["spec_bw_mean"] = float(np.mean(spec_bw))

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
    feats["spec_rolloff_mean"] = float(np.mean(spec_rolloff))

    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    sc_means = np.mean(spec_contrast, axis=1)
    for i, v in enumerate(sc_means, start=1):
        feats[f"spec_contrast_mean_{i:02d}"] = float(v)

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i, v in enumerate(np.mean(chroma_stft, axis=1), start=1):
        feats[f"chroma_stft_mean_{i:02d}"] = float(v)

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i, v in enumerate(np.mean(chroma_cqt, axis=1), start=1):
        feats[f"chroma_cqt_mean_{i:02d}"] = float(v)

    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i, v in enumerate(np.mean(chroma_cens, axis=1), start=1):
        feats[f"chroma_cens_mean_{i:02d}"] = float(v)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop,
        n_mels=cfg.n_mels, fmax=cfg.fmax
    )
    for i, v in enumerate(np.mean(mfcc, axis=1), start=1):
        feats[f"mfcc_mean_{i:02d}"] = float(v)
    for i, v in enumerate(np.std(mfcc, axis=1), start=1):
        feats[f"mfcc_std_{i:02d}"] = float(v)

    # Pitch (robust)
    try:
        import librosa
        f0 = librosa.yin(y, fmin=50.0, fmax=min(1000.0, sr / 2.0), sr=sr, frame_length=n_fft, hop_length=hop)
        f0 = np.where(np.isfinite(f0), f0, np.nan)
        feats["pitch_mean"] = float(np.nanmean(f0)) if np.any(np.isfinite(f0)) else 0.0
        feats["pitch_std"]  = float(np.nanstd(f0))  if np.any(np.isfinite(f0)) else 0.0
    except Exception:
        feats["pitch_mean"] = 0.0
        feats["pitch_std"]  = 0.0

    # Wavelets
    try:
        import pywt
        coeffs = pywt.wavedec(y, "db4", level=5)
        w_means = [float(np.mean(np.abs(c))) for c in coeffs]
        w_stds  = [float(np.std(np.abs(c))) for c in coeffs]
        for i, v in enumerate(w_means, start=1):
            feats[f"wavelet_mean_{i:02d}"] = v
        for i, v in enumerate(w_stds, start=1):
            feats[f"wavelet_std_{i:02d}"]  = v
    except Exception:
        for i in range(1, 7):
            feats[f"wavelet_mean_{i:02d}"] = 0.0
            feats[f"wavelet_std_{i:02d}"]  = 0.0

    return feats


def extract_all_features(df_index: pd.DataFrame, cfg: ExtractConfig) -> pd.DataFrame:
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x

    jobs = [
        {
            "split": r.split,
            "file_id": r.file_id,
            "abs_path": r.abs_path,
            "label": (r.label if isinstance(r.label, str) else None),
            "target": (int(r.target) if pd.notna(r.target) else None),
        }
        for r in df_index.itertuples(index=False)
    ]

    rows: List[Dict[str, object]] = []

    def _worker(jd):
        p = Path(jd["abs_path"])  # absolute path
        feats = extract_features_for_path(p, cfg)
        return feats

    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        fut_map = {ex.submit(_worker, jd): jd for jd in jobs}
        for fut in tqdm(as_completed(fut_map), total=len(jobs), desc="Extracting"):
            jd = fut_map[fut]
            try:
                feats = fut.result()
            except Exception:
                feats = {}
            base = {
                "split": jd["split"],
                "file_id": jd["file_id"],
                "path": jd["abs_path"],
                "label": jd["label"],
                "target": jd["target"],
            }
            base.update(feats)
            rows.append(base)

    feat_df = pd.DataFrame(rows)
    cols_order = ["split", "file_id", "path", "label", "target"]
    other_cols = sorted([c for c in feat_df.columns if c not in cols_order])
    return feat_df[cols_order + other_cols]
