from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Iterable
from time import monotonic

import joblib
import numpy as np
import pandas as pd
import librosa
import pywt
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ExtractConfig, FEATURES_LIST

META_COLS = ["split", "file_id", "path", "label", "target"]
_SSL_GROUPS = {"wav2vec", "wavlm"}
_SSL_BUNDLES: dict[str, tuple[Any, Any, Any, Any]] = {}
_STAT_NAME_CACHE: Dict[Tuple[str, int], Tuple[List[str], List[str]]] = {}
_SSL_RAW_NAME_CACHE: Dict[Tuple[str, int], List[str]] = {}
_SSL_PCA_NAME_CACHE: Dict[Tuple[str, int], List[str]] = {}


def _pf(msg: str) -> None:
    print(msg, flush=True)


def _stat_col_names(prefix: str, n: int) -> Tuple[List[str], List[str]]:
    key = (prefix, int(n))
    cached = _STAT_NAME_CACHE.get(key)
    if cached is not None:
        return cached
    means = [f"{prefix}_mean_{i:03d}" for i in range(1, n + 1)]
    stds = [f"{prefix}_std_{i:03d}" for i in range(1, n + 1)]
    _STAT_NAME_CACHE[key] = (means, stds)
    return means, stds


def _ssl_raw_col_names(group: str, n: int) -> List[str]:
    key = (group, int(n))
    cached = _SSL_RAW_NAME_CACHE.get(key)
    if cached is not None:
        return cached
    cols = [f"{group}_raw_{i:04d}" for i in range(1, n + 1)]
    _SSL_RAW_NAME_CACHE[key] = cols
    return cols


def _ssl_pca_col_names(group: str, n: int) -> List[str]:
    key = (group, int(n))
    cached = _SSL_PCA_NAME_CACHE.get(key)
    if cached is not None:
        return cached
    cols = [f"{group}_pca_{i:03d}" for i in range(1, n + 1)]
    _SSL_PCA_NAME_CACHE[key] = cols
    return cols


def normalize_groups(groups: Optional[Iterable[str]]) -> List[str]:
    if groups is None:
        return list(FEATURES_LIST)

    raw = [str(g).strip().lower() for g in groups if str(g).strip()]
    if not raw or "all" in raw:
        return list(FEATURES_LIST)

    unknown = sorted(set(raw) - set(FEATURES_LIST))
    if unknown:
        raise ValueError(f"Unknown feature groups requested: {unknown}")

    seen = set()
    out: List[str] = []
    for g in raw:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _frame_params(sr: int, window_length_ms: float) -> Tuple[int, int]:
    n_fft = int(round(sr * window_length_ms / 1000.0))
    n_fft = max(128, 1 << (n_fft - 1).bit_length())
    hop = max(1, n_fft // 4)
    return n_fft, hop


def _normalize_int_array_to_float32(x: np.ndarray) -> np.ndarray:
    info = np.iinfo(x.dtype)
    denom = float(max(abs(info.min), info.max))
    return x.astype(np.float32) / denom


def _load_audio_strict(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {path}")

    ext = path.suffix.lower()

    if ext == ".wav":
        sr, x = wavfile.read(str(path))
        if x.size == 0:
            raise ValueError(f"Empty WAV: {path}")

        if x.ndim == 2:
            x = np.mean(x, axis=1)

        if np.issubdtype(x.dtype, np.integer):
            y = _normalize_int_array_to_float32(x)
        elif np.issubdtype(x.dtype, np.floating):
            y = x.astype(np.float32, copy=False)
        else:
            y = x.astype(np.float32, copy=False)
            ma = float(np.max(np.abs(y))) or 1.0
            y /= ma

        if int(sr) != int(target_sr):
            y = librosa.resample(y, orig_sr=int(sr), target_sr=int(target_sr))
            sr = int(target_sr)
    else:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        if y.size == 0:
            raise ValueError(f"Empty audio: {path}")

    y, _ = librosa.effects.trim(y, top_db=30)
    if y.size == 0:
        raise ValueError(f"All-silence after trim: {path}")

    return y.astype(np.float32, copy=False), int(sr)


# ----------------------------
# CHROMA (NumPy)
# ----------------------------
def _chroma_numpy(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, window="hann", center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    k_bins = np.arange(1, S.shape[0], dtype=int)
    if k_bins.size == 0:
        return np.zeros((12, S.shape[1]), dtype=np.float32)

    freqs_k = freqs[k_bins]
    fC = 261.625565
    with np.errstate(divide="ignore"):
        pcs = np.round(12.0 * np.log2(np.maximum(freqs_k, 1e-12) / fC)).astype(int) % 12

    chroma = np.zeros((12, S.shape[1]), dtype=np.float32)
    for pc in range(12):
        mask = pcs == pc
        if np.any(mask):
            chroma[pc, :] = np.sum(S[k_bins[mask], :], axis=0).astype(np.float32)

    chroma /= (np.sum(chroma, axis=0, keepdims=True) + 1e-10)
    return chroma


# ----------------------------
# Pitch (autocorrelation)
# ----------------------------
def _pitch_autocorr(y: np.ndarray, sr: int, n_fft: int, hop: int) -> np.ndarray:
    fmin = max(50.0, sr / float(n_fft) * 1.1)
    fmax = min(800.0, sr / 4.0)

    lag_min = int(max(1, sr // fmax))
    lag_max = int(min(n_fft - 1, sr // fmin))

    if lag_max <= lag_min + 1:
        return np.full(1, np.nan, dtype=np.float32)

    win = np.hanning(n_fft).astype(np.float32)
    frames: List[float] = []
    for start in range(0, len(y) - n_fft + 1, hop):
        x = y[start:start + n_fft]
        x = (x - np.mean(x)) * win
        if not np.any(np.abs(x) > 0):
            frames.append(np.nan)
            continue

        r = np.correlate(x, x, mode="full")[n_fft - 1:]
        r0 = r[0] if r[0] > 0 else 1.0
        r /= r0

        seg = r[lag_min:lag_max]
        if seg.size == 0:
            frames.append(np.nan)
            continue

        lag = lag_min + int(np.argmax(seg))
        conf = r[lag]
        if conf < 0.1:
            frames.append(np.nan)
        else:
            frames.append(float(sr / lag))

    if not frames:
        return np.full(1, np.nan, dtype=np.float32)

    return np.array(frames, dtype=np.float32)


# ----------------------------
# LPCC/CQCC helpers (spafe)
# ----------------------------
def _load_spafe_functions() -> tuple[Any, Any]:
    lpcc_fn = None
    cqcc_fn = None

    lpcc_errors: List[str] = []
    for mod_name, fn_name in [
        ("spafe.features.lpc", "lpcc"),
        ("spafe.features.lpcc", "lpcc"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            lpcc_fn = getattr(mod, fn_name)
            break
        except Exception as e:
            lpcc_errors.append(f"{mod_name}.{fn_name}: {e}")

    cqcc_errors: List[str] = []
    for mod_name, fn_name in [
        ("spafe.features.cqcc", "cqcc"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            cqcc_fn = getattr(mod, fn_name)
            break
        except Exception as e:
            cqcc_errors.append(f"{mod_name}.{fn_name}: {e}")

    if lpcc_fn is None:
        raise ImportError("Could not import LPCC from spafe. Tried: " + " | ".join(lpcc_errors))
    if cqcc_fn is None:
        raise ImportError("Could not import CQCC from spafe. Tried: " + " | ".join(cqcc_errors))

    return lpcc_fn, cqcc_fn


def _call_spafe_feature(fn: Any, y: np.ndarray, sr: int, n_ceps: int, n_fft: int) -> np.ndarray:
    attempts = [
        lambda: fn(sig=y, fs=sr, num_ceps=n_ceps, nfft=n_fft),
        lambda: fn(sig=y, fs=sr, num_ceps=n_ceps),
        lambda: fn(y, sr, n_ceps),
    ]
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            out = attempt()
            arr = np.asarray(out, dtype=np.float32)
            if arr.size == 0:
                raise ValueError("spafe returned empty array")
            return arr
        except Exception as e:
            last_error = e
    raise RuntimeError(f"spafe feature extraction failed: {type(last_error).__name__}: {last_error}")


def _ensure_frames_x_coeffs(mat: np.ndarray, expected_coeffs: int) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D cepstral matrix, got shape={arr.shape}")

    if arr.shape[1] == expected_coeffs:
        return arr
    if arr.shape[0] == expected_coeffs:
        return arr.T

    if arr.shape[0] < arr.shape[1]:
        return arr.T
    return arr


def _cepstral_with_deltas(base_frames_coeffs: np.ndarray) -> np.ndarray:
    coeffs_frames = base_frames_coeffs.T
    d1 = librosa.feature.delta(coeffs_frames, order=1, axis=1)
    d2 = librosa.feature.delta(coeffs_frames, order=2, axis=1)
    all_coeffs = np.concatenate([coeffs_frames, d1, d2], axis=0)
    return all_coeffs.T


def _append_stats_features(feats: Dict[str, float], prefix: str, mat: np.ndarray) -> None:
    v_mean = np.mean(mat, axis=0)
    v_std = np.std(mat, axis=0)
    mean_cols, std_cols = _stat_col_names(prefix, v_mean.shape[0])
    for col, v in zip(mean_cols, v_mean):
        feats[col] = float(v)
    for col, v in zip(std_cols, v_std):
        feats[col] = float(v)


# ----------------------------
# SSL helpers (wav2vec / wavlm)
# ----------------------------
def _require_ssl_runtime(cfg: ExtractConfig) -> None:
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency 'torch' for wav2vec/WaveLM extraction") from e

    if cfg.ssl_require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU is mandatory for SSL feature extraction, but CUDA is not available.")


def _load_ssl_bundle(cfg: ExtractConfig, group: str) -> tuple[Any, Any, Any, Any]:
    import torch  # type: ignore
    from transformers import AutoModel, AutoFeatureExtractor  # type: ignore

    if group == "wav2vec":
        model_id = cfg.ssl_wav2vec_model_id
    elif group == "wavlm":
        model_id = cfg.ssl_wavlm_model_id
    else:
        raise KeyError(f"Unknown SSL group: {group}")

    key = f"{group}:{model_id}"
    cached = _SSL_BUNDLES.get(key)
    if cached is not None:
        return cached

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = cfg.ssl_hf_cache_dir or None

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
    model.to(device)
    model.eval()

    bundle = (feature_extractor, model, device, torch)
    _SSL_BUNDLES[key] = bundle
    return bundle


def _extract_ssl_raw_vector(y: np.ndarray, sr: int, cfg: ExtractConfig, group: str) -> np.ndarray:
    _require_ssl_runtime(cfg)
    feature_extractor, model, device, torch = _load_ssl_bundle(cfg, group)

    inputs = feature_extractor(
        y,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state

    arr = hidden.squeeze(0).detach().float().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    vec = np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)
    return vec.astype(np.float32, copy=False)


def load_pca_models(transforms_dir: Path, groups: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    groups = normalize_groups(groups)
    out: Dict[str, Any] = {}
    tdir = Path(transforms_dir)
    for g in groups:
        if g not in _SSL_GROUPS:
            continue
        p = tdir / f"{g}_pca.joblib"
        if p.exists():
            out[g] = joblib.load(p)
    return out


def _fit_and_apply_ssl_pca(feat_df: pd.DataFrame, cfg: ExtractConfig, groups: List[str]) -> pd.DataFrame:
    use_groups = [g for g in groups if g in _SSL_GROUPS]
    if not use_groups:
        return feat_df

    if "split" not in feat_df.columns:
        raise ValueError("Cannot fit PCA without split column")

    train_mask = feat_df["split"].astype(str) == "train"
    if not bool(train_mask.any()):
        raise ValueError("Cannot fit PCA for SSL features: no train split rows")

    out = feat_df.copy()
    transforms_dir = Path(cfg.out_dir) / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    for g in use_groups:
        raw_cols = sorted([c for c in out.columns if c.startswith(f"{g}_raw_")])
        if not raw_cols:
            continue

        X_train = out.loc[train_mask, raw_cols].to_numpy(dtype=np.float32, copy=False)
        if X_train.shape[0] < 2:
            raise ValueError(f"Need at least 2 train rows to fit PCA for {g}")

        n_comp = int(min(cfg.ssl_pca_components, X_train.shape[0], X_train.shape[1]))
        if n_comp < 1:
            raise ValueError(f"Invalid PCA component count for {g}: {n_comp}")

        pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            PCA(n_components=n_comp, random_state=cfg.random_state),
        )
        pipe.fit(X_train)

        X_all = pipe.transform(out[raw_cols].to_numpy(dtype=np.float32, copy=False)).astype(np.float32)
        pca_cols = _ssl_pca_col_names(g, X_all.shape[1])

        for i, col in enumerate(pca_cols):
            out[col] = X_all[:, i]

        out.drop(columns=raw_cols, inplace=True)
        joblib.dump(pipe, transforms_dir / f"{g}_pca.joblib")

    return out


def extract_features_for_path(
    path: Path,
    cfg: ExtractConfig,
    *,
    groups: Optional[Iterable[str]] = None,
    pca_models: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Extrage feature-uri pentru un fișier audio.
    - groups: subset de grupuri, sau None pentru toate.
    - pca_models: transformeri PCA deja antrenați pentru wav2vec/wavlm (inferență).
    """
    groups = normalize_groups(groups)
    feats: Dict[str, float] = {}

    _pf(f"    STAGE: load        START :: {path}")
    y, sr = _load_audio_strict(path, cfg.sampling_rate)
    _pf(f"    STAGE: load        DONE  :: len={len(y)} sr={sr}")

    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    if "zcr_rms" in groups:
        _pf("    STAGE: zcr_rms     START")
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["rms_mean"] = float(np.mean(rms))
        _pf("    STAGE: zcr_rms     DONE")

    if "spectral_basic" in groups:
        _pf("    STAGE: spectral    START")
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
        feats["spec_centroid_mean"] = float(np.mean(spec_centroid))
        feats["spec_bw_mean"] = float(np.mean(spec_bw))
        feats["spec_rolloff_mean"] = float(np.mean(spec_rolloff))
        _pf("    STAGE: spectral    DONE")

    if "spectral_contrast" in groups:
        _pf("    STAGE: contrast    START")
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        for i, v in enumerate(np.mean(spec_contrast, axis=1), start=1):
            feats[f"spec_contrast_mean_{i:02d}"] = float(v)
        _pf("    STAGE: contrast    DONE")

    if "chroma" in groups:
        _pf("    STAGE: chroma      START")
        chroma = _chroma_numpy(y, sr, n_fft, hop)
        for i, v in enumerate(np.mean(chroma, axis=1), start=1):
            feats[f"chroma_mean_{i:02d}"] = float(v)
        _pf("    STAGE: chroma      DONE")

    if "mfcc" in groups:
        _pf("    STAGE: mfcc        START")
        fmax_safe = float(min(cfg.fmax, (sr / 2.0) - 1.0))
        n_mels_safe = int(min(cfg.n_mels, max(8, n_fft // 4)))
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels_safe,
            fmax=fmax_safe,
        )
        feats.update({f"mfcc_mean_{i:02d}": float(v) for i, v in enumerate(np.mean(mfcc, axis=1), start=1)})
        feats.update({f"mfcc_std_{i:02d}": float(v) for i, v in enumerate(np.std(mfcc, axis=1), start=1)})
        _pf("    STAGE: mfcc        DONE")

    if "pitch" in groups:
        _pf("    STAGE: pitch_acf   START")
        f0_track = _pitch_autocorr(y, sr, n_fft, hop)
        if np.all(~np.isfinite(f0_track)):
            raise ValueError("Pitch extraction (autocorr) failed (all NaN)")
        feats["pitch_mean"] = float(np.nanmean(f0_track))
        feats["pitch_std"] = float(np.nanstd(f0_track))
        _pf("    STAGE: pitch_acf   DONE")

    if "wavelets" in groups:
        _pf("    STAGE: wavelets    START")
        coeffs = pywt.wavedec(y, "db4", level=5)
        if not coeffs:
            raise ValueError("Wavelet decomposition failed")
        for i, c in enumerate(coeffs, start=1):
            abs_c = np.abs(c)
            feats[f"wavelet_mean_{i:02d}"] = float(np.mean(abs_c))
            feats[f"wavelet_std_{i:02d}"] = float(np.std(abs_c))
        _pf("    STAGE: wavelets    DONE")

    if "lpcc" in groups or "cqcc" in groups:
        lpcc_fn, cqcc_fn = _load_spafe_functions()

        if "lpcc" in groups:
            _pf("    STAGE: lpcc        START")
            lpcc_mat = _call_spafe_feature(lpcc_fn, y, sr, cfg.lpcc_num_ceps, n_fft)
            lpcc_fc = _ensure_frames_x_coeffs(lpcc_mat, cfg.lpcc_num_ceps)
            lpcc_all = _cepstral_with_deltas(lpcc_fc)
            _append_stats_features(feats, "lpcc", lpcc_all)
            _pf("    STAGE: lpcc        DONE")

        if "cqcc" in groups:
            _pf("    STAGE: cqcc        START")
            cqcc_mat = _call_spafe_feature(cqcc_fn, y, sr, cfg.cqcc_num_ceps, n_fft)
            cqcc_fc = _ensure_frames_x_coeffs(cqcc_mat, cfg.cqcc_num_ceps)
            cqcc_all = _cepstral_with_deltas(cqcc_fc)
            _append_stats_features(feats, "cqcc", cqcc_all)
            _pf("    STAGE: cqcc        DONE")

    for g in ["wav2vec", "wavlm"]:
        if g not in groups:
            continue
        _pf(f"    STAGE: {g:<11s} START")
        raw_vec = _extract_ssl_raw_vector(y, sr, cfg, g)
        pca = (pca_models or {}).get(g)

        if pca is not None:
            vec = pca.transform(raw_vec.reshape(1, -1)).astype(np.float32, copy=False).ravel()
            pca_cols = _ssl_pca_col_names(g, vec.shape[0])
            for col, v in zip(pca_cols, vec):
                feats[col] = float(v)
        else:
            raw_cols = _ssl_raw_col_names(g, raw_vec.shape[0])
            for col, v in zip(raw_cols, raw_vec):
                feats[col] = float(v)

        _pf(f"    STAGE: {g:<11s} DONE")

    return feats


def _extract_ssl_raw_for_path(path: Path, cfg: ExtractConfig, group: str) -> np.ndarray:
    y, sr = _load_audio_strict(path, cfg.sampling_rate)
    return _extract_ssl_raw_vector(y, sr, cfg, group)


def _extract_single_ssl_group_with_pca(
    jobs: List[Dict[str, Any]],
    cfg: ExtractConfig,
    group: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    if group not in _SSL_GROUPS:
        raise ValueError(f"Expected SSL group, got: {group}")

    train_jobs = [j for j in jobs if str(j["split"]) == "train"]
    if len(train_jobs) < 2:
        raise ValueError(f"Need at least 2 train rows to fit PCA for {group}")

    _pf(f"[*] SSL({group}) pass-1/2: fit PCA on train split ({len(train_jobs)} rows)")
    X_train_rows: List[np.ndarray] = []
    for i, jd in enumerate(train_jobs, start=1):
        p = Path(jd["abs_path"])
        if verbose:
            _pf(f"[fit:{group} {i}/{len(train_jobs)}] {jd['file_id']} :: {p}")
        try:
            raw_vec = _extract_ssl_raw_for_path(p, cfg, group)
        except Exception as e:
            _pf(f"[!] FAIL fit:{group} {jd['file_id']} :: {p} :: {type(e).__name__}: {e}")
            raise
        X_train_rows.append(raw_vec)

    X_train = np.stack(X_train_rows, axis=0).astype(np.float32, copy=False)
    n_comp = int(min(cfg.ssl_pca_components, X_train.shape[0], X_train.shape[1]))
    if n_comp < 1:
        raise ValueError(f"Invalid PCA component count for {group}: {n_comp}")

    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        PCA(n_components=n_comp, random_state=cfg.random_state),
    )
    pipe.fit(X_train)

    transforms_dir = Path(cfg.out_dir) / "transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, transforms_dir / f"{group}_pca.joblib")
    pca_cols = _ssl_pca_col_names(group, n_comp)

    _pf(f"[*] SSL({group}) pass-2/2: transform all rows ({len(jobs)} rows)")
    rows: List[Dict[str, object]] = []
    for i, jd in enumerate(jobs, start=1):
        p = Path(jd["abs_path"])
        if verbose:
            _pf(f"[{i}/{len(jobs)}] START {jd['split']} {jd['file_id']} :: {p}")
        t0 = monotonic()
        try:
            raw_vec = _extract_ssl_raw_for_path(p, cfg, group)
            pca_vec = pipe.transform(raw_vec.reshape(1, -1)).astype(np.float32, copy=False).ravel()
        except Exception as e:
            _pf(f"[!] FAIL {jd['split']} {jd['file_id']} :: {p} :: {type(e).__name__}: {e}")
            raise

        dt = monotonic() - t0
        if verbose:
            _pf(f"[{i}/{len(jobs)}] DONE  {jd['split']} {jd['file_id']} :: {p} :: {dt:.3f}s")

        base: Dict[str, object] = {
            "split": jd["split"],
            "file_id": jd["file_id"],
            "path": str(p),
            "label": jd["label"],
            "target": jd["target"],
        }
        for col, v in zip(pca_cols, pca_vec):
            base[col] = float(v)
        rows.append(base)

    feat_df = pd.DataFrame(rows)
    other_cols = sorted([c for c in feat_df.columns if c not in META_COLS])
    return feat_df[META_COLS + other_cols]


def extract_all_features(
    df_index: pd.DataFrame,
    cfg: ExtractConfig,
    *,
    verbose: bool = True,
    groups: Optional[Iterable[str]] = None,
    fit_ssl_pca: bool = True,
) -> pd.DataFrame:
    """
    Extrage features secvențial pentru toate rândurile din index.
    Dacă groups include wav2vec/wavlm și fit_ssl_pca=True, aplică PCA train-only și salvează transformările în out_dir/transforms.
    """
    groups = normalize_groups(groups)

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
        return pd.DataFrame(columns=META_COLS)

    # Memory-safe path for SSL-only extraction with PCA:
    # fit on train rows, then transform per-row without materializing raw features for all rows.
    if fit_ssl_pca and len(groups) == 1 and groups[0] in _SSL_GROUPS:
        return _extract_single_ssl_group_with_pca(jobs, cfg, groups[0], verbose=verbose)

    first = jobs[0]
    p0 = Path(first["abs_path"])
    _pf(f"[*] Preflight minimal: load+RMS :: {p0}")
    y0, sr0 = _load_audio_strict(p0, cfg.sampling_rate)
    rms0 = float(np.sqrt(np.mean(y0 ** 2)))
    _pf(f"[*] Preflight OK :: len={len(y0)} sr={sr0} rms~{rms0:.4f}")

    rows: List[Dict[str, object]] = []

    for i, jd in enumerate(jobs, start=1):
        p = Path(jd["abs_path"])
        if verbose:
            _pf(f"[{i}/{len(jobs)}] START {jd['split']} {jd['file_id']} :: {p}")
        t0 = monotonic()
        try:
            feats = extract_features_for_path(p, cfg, groups=groups)
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

    if fit_ssl_pca and any(g in _SSL_GROUPS for g in groups):
        feat_df = _fit_and_apply_ssl_pca(feat_df, cfg, groups)

    other_cols = sorted([c for c in feat_df.columns if c not in META_COLS])
    return feat_df[META_COLS + other_cols]
