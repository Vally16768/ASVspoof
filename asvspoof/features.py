# === features.py (STRICT: crash-resistant + live progress logs + decode timeout) ===
from __future__ import annotations

import os
import time
import subprocess
import shutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from .config import ExtractConfig  # sampling_rate, window_length_ms, n_mels, fmax, workers


# ----------------------------- Dependencies -----------------------------

def _require_deps():
    """
    Import hard dependencies once; raise immediately if missing.
    We decode with ffmpeg (CLI), then use librosa/pywt for features.
    """
    import librosa    # noqa: F401
    import pywt       # noqa: F401
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Please install ffmpeg (e.g. `sudo apt-get install ffmpeg`)."
        )


# ----------------------------- Utilities -------------------------------

def _frame_params(sr: int, window_length_ms: float) -> Tuple[int, int]:
    """Return (n_fft as next pow2 for stability, hop_length as n_fft//4)."""
    n_fft = int(round(sr * window_length_ms / 1000.0))
    n_fft_pow2 = 1 << (n_fft - 1).bit_length()
    hop = max(1, n_fft_pow2 // 4)
    return n_fft_pow2, hop


def _worker_init():
    """
    Initialize each worker process with sane thread caps for native libs.
    Prevents oversubscription/instability when running many workers.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# --------------------------- Per-file extractor -------------------------

def _load_audio_strict(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """
    STRICT, crash-resistant read using ffmpeg CLI:
      - Decodes *any* input via ffmpeg, converts to mono, resamples to target_sr,
        outputs raw float32 PCM on stdout, which we turn into np.float32 array.
      - Hard timeout to prevent hangs (env ASV_FFMPEG_TIMEOUT_SEC, default 20s).
      - No silent fallbacks; raises with a clear message if ffmpeg fails/times out.
    Requirements: ffmpeg must be installed and available on PATH.
    """
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    timeout_sec = float(os.getenv("ASV_FFMPEG_TIMEOUT_SEC", "20"))

    cmd = [
        "ffmpeg",
        "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(path),
        "-vn",                # ignore any video streams
        "-dn",                # ignore data streams
        "-ac", "1",           # mono
        "-ar", str(target_sr),
        "-f", "f32le",        # raw float32 little-endian to stdout
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as te:
        raise TimeoutError(
            f"ffmpeg decode timed out after {timeout_sec:.0f}s for: {path}"
        ) from te
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg.")  # pragma: no cover

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore")
        raise ValueError(f"ffmpeg failed to decode: {path}\n{err}")

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Empty audio after ffmpeg decode: {path}")

    # Trim leading/trailing silence; must remain non-empty.
    import librosa
    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        raise ValueError(f"All-silence audio after trim: {path}")

    return y, target_sr


def extract_features_for_path(path: Path, cfg: ExtractConfig) -> Dict[str, float]:
    """
    STRICT extractor:
      - Robust decode with ffmpeg (no native decoder segfaults in-process).
      - No silent fallbacks; any issue raises with explicit file path.
    """
    import librosa
    import pywt

    y, sr = _load_audio_strict(path, cfg.sampling_rate)
    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    feats: Dict[str, float] = {}

    # --- Core scalar features ---
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["rms_mean"] = float(np.mean(rms))

    # --- Spectral descriptors ---
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
    feats["spec_centroid_mean"] = float(np.mean(spec_centroid))
    feats["spec_bw_mean"]       = float(np.mean(spec_bw))
    feats["spec_rolloff_mean"]  = float(np.mean(spec_rolloff))

    # --- Spectral contrast ---
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i, v in enumerate(np.mean(spec_contrast, axis=1), start=1):
        feats[f"spec_contrast_mean_{i:02d}"] = float(v)

    # --- Chroma variants ---
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i, v in enumerate(np.mean(chroma_stft, axis=1), start=1):
        feats[f"chroma_stft_mean_{i:02d}"] = float(v)

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i, v in enumerate(np.mean(chroma_cqt, axis=1), start=1):
        feats[f"chroma_cqt_mean_{i:02d}"] = float(v)

    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i, v in enumerate(np.mean(chroma_cens, axis=1), start=1):
        feats[f"chroma_cens_mean_{i:02d}"] = float(v)

    # --- MFCCs ---
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop,
        n_mels=cfg.n_mels, fmax=cfg.fmax
    )
    for i, v in enumerate(np.mean(mfcc, axis=1), start=1):
        feats[f"mfcc_mean_{i:02d}"] = float(v)
    for i, v in enumerate(np.std(mfcc, axis=1), start=1):
        feats[f"mfcc_std_{i:02d}"] = float(v)

    # --- Pitch (YIN) ---
    f0 = librosa.yin(y, fmin=50.0, fmax=min(1000.0, sr / 2.0), sr=sr, frame_length=n_fft, hop_length=hop)
    f0 = np.where(np.isfinite(f0), f0, np.nan)
    if not np.any(np.isfinite(f0)):
        raise ValueError(f"Pitch extraction failed (all NaN): {path}")
    feats["pitch_mean"] = float(np.nanmean(f0))
    feats["pitch_std"]  = float(np.nanstd(f0))

    # --- Wavelets ---
    coeffs  = pywt.wavedec(y, "db4", level=5)
    if not coeffs:
        raise ValueError(f"Wavelet decomposition failed: {path}")
    w_means = [float(np.mean(np.abs(c))) for c in coeffs]
    w_stds  = [float(np.std(np.abs(c)))  for c in coeffs]
    for i, v in enumerate(w_means, start=1):
        feats[f"wavelet_mean_{i:02d}"] = v
    for i, v in enumerate(w_stds, start=1):
        feats[f"wavelet_std_{i:02d}"]  = v

    return feats


# ----------------------------- Pool worker -----------------------------

def _worker(jd: Dict[str, Any]) -> Dict[str, float]:
    """Worker: DO NOT swallow exceptions; propagate with path context."""
    path = Path(jd["abs_path"])
    cfg: ExtractConfig = jd["cfg"]
    return extract_features_for_path(path, cfg)


# -------------------------- Batch extraction ---------------------------

def extract_all_features(df_index: pd.DataFrame, cfg: ExtractConfig) -> pd.DataFrame:
    """
    Strict batch extraction + live logging.
    - Validates deps up front.
    - Preflights the first file sequentially (with timeout) to avoid pool-masked hangs.
    - Uses spawn-based pool with per-process thread caps.
    - Streams submissions and prints heartbeats if nothing completes for N seconds.
    - If any file fails, raises immediately with the offending path.
    """
    _require_deps()

    # Logging knobs
    LOG_EVERY = int(os.getenv("ASV_LOG_EVERY", "200"))
    HEARTBEAT_SEC = float(os.getenv("ASV_HEARTBEAT_SEC", "5"))
    MAX_INFLIGHT_FACTOR = int(os.getenv("ASV_MAX_INFLIGHT_FACTOR", "4"))

    # Build jobs list (expects absolute paths present in index)
    jobs: List[Dict[str, Any]] = [
        {
            "split": r.split,
            "file_id": r.file_id,
            "abs_path": r.abs_path,
            "label": (r.label if isinstance(r.label, str) else None),
            "target": (int(r.target) if pd.notna(r.target) else None),
            "cfg": cfg,
        }
        for r in df_index.itertuples(index=False)
    ]
    total = len(jobs)

    rows: List[Dict[str, object]] = []

    # Allow strict sequential mode for diagnosis if workers == 0
    if getattr(cfg, "workers", 1) == 0:
        print(f"[*] Sequential mode on {total} files", flush=True)
        for i, jd in enumerate(jobs, 1):
            feats = extract_features_for_path(Path(jd["abs_path"]), jd["cfg"])
            base = {
                "split": jd["split"],
                "file_id": jd["file_id"],
                "path": jd["abs_path"],
                "label": jd["label"],
                "target": jd["target"],
            }
            base.update(feats)
            rows.append(base)
            if i % LOG_EVERY == 0 or i == total:
                print(f"[seq] done {i}/{total}  ({100.0*i/total:.1f}%)  last={jd['file_id']}", flush=True)
    else:
        # --- Preflight: run the very first file in-process (with ffmpeg timeout)
        if jobs:
            first = jobs[0]
            print(f"[*] Preflight: {first['file_id']}", flush=True)
            _ = extract_features_for_path(Path(first["abs_path"]), first["cfg"])

        # Spawn-based pool to avoid fork+native-lib crashes
        ctx = mp.get_context("spawn")
        max_workers = max(1, int(cfg.workers))
        max_inflight = max_workers * max(1, MAX_INFLIGHT_FACTOR)

        print(f"[*] Starting pool: workers={max_workers}, max_inflight={max_inflight}", flush=True)

        submitted = 0
        completed = 0
        in_flight = {}
        start_t = time.perf_counter()

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_worker_init,
        ) as ex:
            # Prime the pump
            while submitted < total and len(in_flight) < max_inflight:
                jd = jobs[submitted]
                fut = ex.submit(_worker, jd)
                in_flight[fut] = jd
                submitted += 1

            # Main loop
            while in_flight:
                done, _ = wait(list(in_flight.keys()), timeout=HEARTBEAT_SEC, return_when=FIRST_COMPLETED)

                if not done:
                    # Heartbeat if nothing completed during the timeout
                    now = time.perf_counter()
                    rate = completed / max(1e-9, now - start_t)
                    print(f"[hb] submitted={submitted}/{total} inflight={len(in_flight)} "
                          f"done={completed} rate={rate:.1f}/s", flush=True)
                    continue

                # Consume completions
                for fut in done:
                    jd = in_flight.pop(fut)
                    try:
                        feats = fut.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"[FATAL] Feature extraction failed for file:\n"
                            f"        split={jd['split']} file_id={jd['file_id']}\n"
                            f"        path={jd['abs_path']}\n"
                            f"        reason={type(e).__name__}: {e}"
                        ) from e

                    base = {
                        "split": jd["split"],
                        "file_id": jd["file_id"],
                        "path": jd["abs_path"],
                        "label": jd["label"],
                        "target": jd["target"],
                    }
                    base.update(feats)
                    rows.append(base)

                    completed += 1
                    if completed % LOG_EVERY == 0 or completed == total:
                        now = time.perf_counter()
                        rate = completed / max(1e-9, now - start_t)
                        print(f"[ok] {completed}/{total} ({100.0*completed/total:.1f}%) "
                              f"last={jd['file_id']} inflight={len(in_flight)} rate={rate:.1f}/s",
                              flush=True)

                    # Refill to keep pipeline busy
                    while submitted < total and len(in_flight) < max_inflight:
                        jd2 = jobs[submitted]
                        fut2 = ex.submit(_worker, jd2)
                        in_flight[fut2] = jd2
                        submitted += 1

    feat_df = pd.DataFrame(rows)
    cols_order = ["split", "file_id", "path", "label", "target"]
    other_cols = sorted([c for c in feat_df.columns if c not in cols_order])
    return feat_df[cols_order + other_cols]
