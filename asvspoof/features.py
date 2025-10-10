# asvspoof/features.py — STRICT, SECVENȚIAL, cu log pe fiecare ETAPĂ
from __future__ import annotations

# (1) Blochează thread-urile BLAS/OMP ÎNAINTE de a importa numpy/librosa
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from typing import Dict, List, Any
from time import monotonic

import numpy as np
import pandas as pd

from .config import ExtractConfig

def _pf(msg: str):
    print(msg, flush=True)

def _frame_params(sr: int, window_length_ms: float):
    n_fft = int(round(sr * window_length_ms / 1000.0))
    n_fft_pow2 = 1 << (n_fft - 1).bit_length()
    hop = max(1, n_fft_pow2 // 4)
    return n_fft_pow2, hop

def _load_audio_strict(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """
    .wav -> scipy.io.wavfile (fără soundfile/libsndfile)
    altceva (ex. .flac) -> librosa backend 'audioread'
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    ext = path.suffix.lower()
    import numpy.typing as npt

    def _to_float32(x: npt.NDArray) -> np.ndarray:
        if np.issubdtype(x.dtype, np.floating):
            y = x.astype(np.float32, copy=False)
        elif x.dtype == np.int16:
            y = (x.astype(np.float32) / 32768.0)
        elif x.dtype == np.int32:
            y = (x.astype(np.float32) / 2147483648.0)
        elif x.dtype == np.uint8:
            y = (x.astype(np.float32) - 128.0) / 128.0
        else:
            y = x.astype(np.float32)
            maxv = float(np.max(np.abs(y))) or 1.0
            y /= maxv
        return y

    if ext == ".wav":
        from scipy.io import wavfile
        sr, x = wavfile.read(str(path))  # x: np.int16/int32/float
        if x.size == 0:
            raise ValueError(f"Empty WAV: {path}")
        if x.ndim == 2:
            x = np.mean(x, axis=1)
        y = _to_float32(x)

        if int(sr) != int(target_sr):
            import librosa
            librosa.set_audio_backend("audioread")
            y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
            sr = int(target_sr)

        import librosa
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size == 0:
            raise ValueError(f"All-silence after trim: {path}")
        return y.astype(np.float32, copy=False), int(sr)

    else:
        import librosa
        try:
            librosa.set_audio_backend("audioread")
        except Exception:
            pass
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        if y.size == 0:
            raise ValueError(f"Empty audio: {path}")
        y, _ = librosa.effects.trim(y, top_db=30)
        if y.size == 0:
            raise ValueError(f"All-silence after trim: {path}")
        return y.astype(np.float32, copy=False), int(target_sr)

def extract_features_for_path(path: Path, cfg: ExtractConfig) -> Dict[str, float]:
    # import local ca să evităm efecte globale premature
    import librosa, pywt

    # --- LOAD ---
    _pf(f"    STAGE: load        START :: {path}")
    y, sr = _load_audio_strict(path, cfg.sampling_rate)
    _pf(f"    STAGE: load        DONE  :: len={len(y)} sr={sr}")

    # pregătire ferestre
    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    feats: Dict[str, float] = {}

    # --- ZCR/RMS ---
    _pf("    STAGE: zcr_rms     START")
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["rms_mean"] = float(np.mean(rms))
    _pf("    STAGE: zcr_rms     DONE")

    # --- Spectral basic ---
    _pf("    STAGE: spectral    START")
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
    feats["spec_centroid_mean"] = float(np.mean(spec_centroid))
    feats["spec_bw_mean"]       = float(np.mean(spec_bw))
    feats["spec_rolloff_mean"]  = float(np.mean(spec_rolloff))
    _pf("    STAGE: spectral    DONE")

    # --- Spectral contrast ---
    _pf("    STAGE: contrast    START")
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i, v in enumerate(np.mean(spec_contrast, axis=1), start=1):
        feats[f"spec_contrast_mean_{i:02d}"] = float(v)
    _pf("    STAGE: contrast    DONE")

    # --- Chroma ---
    _pf("    STAGE: chroma      START")
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i, v in enumerate(np.mean(chroma_stft, axis=1), start=1):
        feats[f"chroma_stft_mean_{i:02d}"] = float(v)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i, v in enumerate(np.mean(chroma_cqt, axis=1), start=1):
        feats[f"chroma_cqt_mean_{i:02d}"] = float(v)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i, v in enumerate(np.mean(chroma_cens, axis=1), start=1):
        feats[f"chroma_cens_mean_{i:02d}"] = float(v)
    _pf("    STAGE: chroma      DONE")

    # --- MFCC ---
    _pf("    STAGE: mfcc        START")
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop,
        n_mels=cfg.n_mels, fmax=cfg.fmax
    )
    for i, v in enumerate(np.mean(mfcc, axis=1), start=1):
        feats[f"mfcc_mean_{i:02d}"] = float(v)
    for i, v in enumerate(np.std(mfcc, axis=1), start=1):
        feats[f"mfcc_std_{i:02d}"] = float(v)
    _pf("    STAGE: mfcc        DONE")

    # --- Pitch (YIN) ---
    _pf("    STAGE: pitch_yin   START")
    f0 = librosa.yin(y, fmin=50.0, fmax=min(1000.0, sr / 2.0), sr=sr, frame_length=n_fft, hop_length=hop)
    f0 = np.where(np.isfinite(f0), f0, np.nan)
    if not np.any(np.isfinite(f0)):
        raise ValueError("Pitch extraction failed (all NaN)")
    feats["pitch_mean"] = float(np.nanmean(f0))
    feats["pitch_std"]  = float(np.nanstd(f0))
    _pf("    STAGE: pitch_yin   DONE")

    # --- Wavelets ---
    _pf("    STAGE: wavelets    START")
    import pywt
    coeffs  = pywt.wavedec(y, "db4", level=5)
    if not coeffs:
        raise ValueError("Wavelet decomposition failed")
    for i, c in enumerate(coeffs, start=1):
        abs_c = np.abs(c)
        feats[f"wavelet_mean_{i:02d}"] = float(np.mean(abs_c))
        feats[f"wavelet_std_{i:02d}"]  = float(np.std(abs_c))
    _pf("    STAGE: wavelets    DONE")

    return feats

def extract_all_features(df_index: pd.DataFrame, cfg: ExtractConfig, *, verbose: bool = True) -> pd.DataFrame:
    """
    STRICT + SECVENȚIAL:
      - Preflight MINIM (doar citire audio + RMS) ca să nu "înghețe" pe o etapă grea
      - Apoi parcurge secvențial; FAIL-FAST; log START/DONE + STAGE logs
    """
    # pregătește job-urile
    jobs: List[Dict[str, Any]] = [
        {
            "split": r.split,
            "file_id": r.file_id,
            "abs_path": r.abs_path,
            "label": (r.label if isinstance(r.label, str) else None),
            "target": (int(r.target) if pd.notna(r.target) else None),
        }
        for r in df_index.itertuples(index=False)
    ]
    if not jobs:
        return pd.DataFrame(columns=["split", "file_id", "path", "label", "target"])

    # --- Preflight MINIM (doar citire + RMS) ---
    first = jobs[0]
    p0 = Path(first["abs_path"])
    _pf(f"[*] Preflight minimal: load+RMS :: {p0}")
    y0, sr0 = _load_audio_strict(p0, cfg.sampling_rate)
    # ceva foarte scurt/robust, fără funcții grele
    rms0 = float(np.sqrt(np.mean(y0 ** 2)))
    _pf(f"[*] Preflight OK :: len={len(y0)} sr={sr0} rms~{rms0:.4f}")

    rows: List[Dict[str, object]] = []

    for i, jd in enumerate(jobs, start=1):
        p = Path(jd["abs_path"])
        if verbose:
            _pf(f"[{i}/{len(jobs)}] START {jd['split']} {jd['file_id']} :: {p}")
        t0 = monotonic()
        try:
            feats = extract_features_for_path(p, cfg)
        except Exception as e:
            _pf(f"[!] FAIL {jd['split']} {jd['file_id']} :: {p} :: {type(e).__name__}: {e}")
            raise
        dt = monotonic() - t0
        if verbose:
            _pf(f"[{i}/{len(jobs)}] DONE  {jd['split']} {jd['file_id']} :: {p} :: {dt:.3f}s")

        base = {
            "split": jd["split"],
            "file_id": jd["file_id"],
            "path": str(p),
            "label": jd["label"],
            "target": jd["target"],
        }
        base.update(feats)
        rows.append(base)

    feat_df = pd.DataFrame(rows)
    cols_order = ["split", "file_id", "path", "label", "target"]
    other_cols = sorted([c for c in feat_df.columns if c not in cols_order])
    return feat_df[cols_order + other_cols]
