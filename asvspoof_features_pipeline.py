#!/usr/bin/env python3
"""
ASVspoof 2019 LA — feature extraction + split builder + combo materializer

What it does
------------
1) Reads ASVspoof 2019 LA protocol files and builds a table of audio files
   with labels (bonafide/spoof) where available.
2) Extracts a rich set of classic audio features for every file.
3) Saves a single Parquet with *all* features for all files.
4) Creates stratified Train/Validation/Test splits (by label) using the
   requested proportions (validation_size, test_size).
5) Materializes *all* non-empty feature-group combinations into compressed
   .npz matrices for each split (X, y, columns, combo_code), where the
   combo_code is the concatenation of letters from feature_name_mapping
   (e.g. "ABM" -> [mfcc_mean, mfcc_std, zcr_mean]).

Notes
-----
- This script computes each feature family once and stores them all together.
  The combinations step only slices the master table and writes matrices.
- Generating 2^15-1 = 32,767 combos can take *a lot* of disk space/time.
  The code streams them one-by-one and writes compressed .npz files to avoid
  loading everything at once in memory. You can also restrict combos via CLI.
- By default we split TRAIN+DEV (labeled) into train/val/test using the given
  proportions. The official EVAL split is unlabeled; we still compute features
  and keep it in the Parquet as split='eval', but we don't create combo matrices
  for eval since it lacks labels.

Usage examples
--------------
# 1) Extract features (multi-process) and build splits + parquet
python asvspoof_features_pipeline.py extract \
  --data-root database/data/asvspoof2019 --workers 8

# 2) Materialize ALL combos into X/y NPZs (train/val/test)
python asvspoof_features_pipeline.py combos \
  --data-root database/data/asvspoof2019 --out-dir database/data/asvspoof2019/index \
  --all

# 3) Materialize only specific combos (e.g. AB, ABL, M, AEMNO)
python asvspoof_features_pipeline.py combos \
  --codes AB ABL M AEMNO

Dependencies
------------
- numpy, pandas, soundfile, librosa, pywt, scikit-learn, pyarrow (for parquet)
- Optional: tqdm (for nicer progress bars)

"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import re
import numpy as np
import pandas as pd

# Heavy libs are imported lazily inside functions when needed to speed CLI help

# ----------------------- Config from user request ----------------------------
FEATURES_LIST = [
    'mfcc_mean',
    'mfcc_std',
    'spec_centroid_mean',
    'spec_bw_mean',
    'spec_contrast_mean',
    'spec_rolloff_mean',
    'rms_mean',
    'zcr_mean',
    'chroma_cens_mean',
    'chroma_cqt_mean',
    'wavelet_mean',
    'chroma_stft_mean',
    'wavelet_std',
    'pitch_mean',
    'pitch_std',
]

FEATURE_NAME_MAPPING = {
    'mfcc_mean': 'A',
    'mfcc_std': 'B',
    'spec_centroid_mean': 'C',
    'spec_bw_mean': 'D',
    'spec_contrast_mean': 'E',
    'spec_rolloff_mean': 'F',
    'rms_mean': 'G',
    'chroma_cens_mean': 'H',
    'chroma_cqt_mean': 'I',
    'wavelet_mean': 'J',
    'chroma_stft_mean': 'K',
    'wavelet_std': 'L',
    'zcr_mean': 'M',
    'pitch_mean': 'N',
    'pitch_std': 'O',
}
FEATURE_NAME_REVERSE_MAPPING = {v: k for k, v in FEATURE_NAME_MAPPING.items()}

# Default hyper-parameters (can be overridden by CLI)
DEFAULTS = dict(
    n_mels=128,
    n_frames=1024,
    test_size=0.10,
    random_state=42,
    epochs=400,
    batch_size=8,
    sampling_rate=44100,
    fmax=22050,
    window_length_ms=15,
    validation_size=0.15,
)

# ------------------------------- Data Classes -------------------------------
@dataclass
class ExtractConfig:
    data_root: Path
    out_dir: Path
    n_mels: int = DEFAULTS['n_mels']
    n_frames: int = DEFAULTS['n_frames']  # not directly used, kept for context
    sampling_rate: int = DEFAULTS['sampling_rate']
    fmax: int = DEFAULTS['fmax']
    window_length_ms: float = DEFAULTS['window_length_ms']
    workers: int = max(1, os.cpu_count() or 1)

@dataclass
class SplitConfig:
    validation_size: float = DEFAULTS['validation_size']
    test_size: float = DEFAULTS['test_size']
    random_state: int = DEFAULTS['random_state']

# ------------------------------ Path Utilities ------------------------------

def protocol_paths(root: Path) -> Dict[str, Path]:
    cm_dir = root / 'ASVspoof2019_LA_cm_protocols'
    return {
        'train': cm_dir / 'ASVspoof2019.LA.cm.train.trn.txt',
        'dev':   cm_dir / 'ASVspoof2019.LA.cm.dev.trl.txt',
        'eval':  cm_dir / 'ASVspoof2019.LA.cm.eval.trl.txt',  # labels unknown
    }


def audio_dir_for_split(root: Path, split: str) -> Path:
    if split == 'train':
        return root / 'ASVspoof2019_LA_train' / 'flac'
    if split == 'dev':
        return root / 'ASVspoof2019_LA_dev' / 'flac'
    if split == 'eval':
        return root / 'ASVspoof2019_LA_eval' / 'flac'
    raise ValueError(f'Unknown split {split}')


# --------------------------- Dataset Index Builder --------------------------

def read_cm_protocol(path: Path, split: str) -> pd.DataFrame:
    """Parse a CM protocol file -> DataFrame with file_id and (optional) label.

    Robustly finds the *audio* file id token like LA_T_1234567 / LA_D_XXXX.
    """
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=['split', 'file_id', 'label'])

    pat_file = re.compile(r'^LA_[TDE]_[0-9]+$')
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Pick the token that looks like an audio id
            file_id = None
            for tok in parts:
                if pat_file.match(tok):
                    file_id = tok
                    break
            # Label is the last token for train/dev
            label = None
            if parts:
                last = parts[-1].lower()
                if last in {'bonafide', 'spoof'}:
                    label = last
            if file_id is not None:
                rows.append({'split': split, 'file_id': file_id, 'label': label})
    df = pd.DataFrame(rows)
    return df


def build_index_table(root: Path) -> pd.DataFrame:
    paths = protocol_paths(root)
    dfs = []
    for split in ['train', 'dev', 'eval']:
        dfs.append(read_cm_protocol(paths[split], split))
    df = pd.concat(dfs, ignore_index=True)

    # Attach absolute audio path
    def make_path(row):
        return audio_dir_for_split(root, row['split']) / f"{row['file_id']}.flac"

    df['path'] = df.apply(make_path, axis=1).astype('string')
    # Ensure label is lower-case or NaN
    df['label'] = df['label'].str.lower()
    # Numeric target: bonafide=1, spoof=0, NaN for eval
    df['target'] = df['label'].map({'bonafide': 1, 'spoof': 0})

    return df


# ---------------------------- Feature Extraction ----------------------------

def _lazy_import_features_libs():
    import soundfile as sf
    import librosa
    import pywt
    return sf, librosa, pywt


def _frame_params(sr: int, window_length_ms: float) -> Tuple[int, int]:
    n_fft = int(round(sr * window_length_ms / 1000.0))
    # use power-of-two n_fft for speed
    n_fft_pow2 = 1 << (n_fft - 1).bit_length()
    hop_length = max(1, n_fft_pow2 // 4)
    return n_fft_pow2, hop_length


def extract_features_for_path(path: Path, cfg: ExtractConfig) -> Dict[str, float]:
    """Compute all requested feature families for one audio file.

    Returns a flat dict: {column_name: value, ...}
    Column naming convention: {group}_{k} with 2-digit index when k >= 1.
    e.g. mfcc_mean_01 ... mfcc_mean_13, spec_contrast_mean_01 ... _07, etc.
    """
    sf, librosa, pywt = _lazy_import_features_libs()

    # Robust IO
    try:
        y, sr = sf.read(str(path), dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
    except Exception:
        # Try librosa.load as fallback
        y, sr = librosa.load(str(path), sr=None, mono=True)

    # Resample
    if sr != cfg.sampling_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sampling_rate)
        sr = cfg.sampling_rate

    # Trim leading/trailing silence (light)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) == 0:
        y = np.zeros(sr // 2, dtype=np.float32)

    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    feats: Dict[str, float] = {}

    # ---- ZCR ----
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    feats['zcr_mean'] = float(np.mean(zcr))

    # ---- RMS ----
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    feats['rms_mean'] = float(np.mean(rms))

    # ---- Spectral features ----
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    feats['spec_centroid_mean'] = float(np.mean(spec_centroid))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    feats['spec_bw_mean'] = float(np.mean(spec_bw))

    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)
    feats['spec_rolloff_mean'] = float(np.mean(spec_rolloff))

    # Spectral contrast returns (n_bands+1, frames). Default n_bands=6 -> 7 rows
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    sc_means = np.mean(spec_contrast, axis=1)
    for i, v in enumerate(sc_means, start=1):
        feats[f'spec_contrast_mean_{i:02d}'] = float(v)

    # ---- Chroma features ----
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    cs_means = np.mean(chroma_stft, axis=1)
    for i, v in enumerate(cs_means, start=1):
        feats[f'chroma_stft_mean_{i:02d}'] = float(v)

    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    ccqt_means = np.mean(chroma_cqt, axis=1)
    for i, v in enumerate(ccqt_means, start=1):
        feats[f'chroma_cqt_mean_{i:02d}'] = float(v)

    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    ccens_means = np.mean(chroma_cens, axis=1)
    for i, v in enumerate(ccens_means, start=1):
        feats[f'chroma_cens_mean_{i:02d}'] = float(v)

    # ---- MFCC ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop, n_mels=DEFAULTS['n_mels'], fmax=DEFAULTS['fmax'])
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)
    for i, v in enumerate(mfcc_means, start=1):
        feats[f'mfcc_mean_{i:02d}'] = float(v)
    for i, v in enumerate(mfcc_stds, start=1):
        feats[f'mfcc_std_{i:02d}'] = float(v)

    # ---- Pitch (YIN) ----
    try:
        f0 = librosa.yin(y, fmin=50.0, fmax=min(1000.0, sr/2.0), sr=sr, frame_length=n_fft, hop_length=hop)
        f0 = np.where(np.isfinite(f0), f0, np.nan)
        feats['pitch_mean'] = float(np.nanmean(f0)) if np.any(np.isfinite(f0)) else 0.0
        feats['pitch_std'] = float(np.nanstd(f0)) if np.any(np.isfinite(f0)) else 0.0
    except Exception:
        feats['pitch_mean'] = 0.0
        feats['pitch_std'] = 0.0

    # ---- Wavelet (DWT) ----
    try:
        # 5-level DWT over db4; returns [cA5, cD5, cD4, cD3, cD2, cD1]
        coeffs = pywt.wavedec(y, 'db4', level=5)
        w_means = [float(np.mean(np.abs(c))) for c in coeffs]
        w_stds  = [float(np.std(np.abs(c))) for c in coeffs]
        for i, v in enumerate(w_means, start=1):
            feats[f'wavelet_mean_{i:02d}'] = v
        for i, v in enumerate(w_stds, start=1):
            feats[f'wavelet_std_{i:02d}'] = v
    except Exception:
        # Fallback to zeros matching typical 6 subbands
        for i in range(1, 7):
            feats[f'wavelet_mean_{i:02d}'] = 0.0
            feats[f'wavelet_std_{i:02d}'] = 0.0

    return feats


def extract_all_features(df_index: pd.DataFrame, cfg: ExtractConfig) -> pd.DataFrame:
    """Iterate over index and extract features into a single DataFrame.

    The result contains columns: [split, file_id, path, label, target, ...features]
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x  # minimal fallback

    rows = []

    def _worker(row_dict: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
        path = Path(row_dict['path'])
        feats = extract_features_for_path(path, cfg)
        return row_dict['file_id'], feats

    # Minimal dicts to ship across processes
    jobs = [
        {
            'split': r.split,
            'file_id': r.file_id,
            'path': r.path,
            'label': (r.label if isinstance(r.label, str) else None),
            'target': (int(r.target) if not pd.isna(r.target) else None),
        }
        for r in df_index.itertuples(index=False)
    ]

    # Process pool
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(_worker, jd): jd for jd in jobs}
        for fut in tqdm(as_completed(futures), total=len(jobs), desc='Extracting'):
            jd = futures[fut]
            try:
                file_id, feats = fut.result()
            except Exception as e:  # pragma: no cover
                # Skip file but keep metadata, filled with NaNs
                feats = {}
            base = {
                'split': jd['split'],
                'file_id': jd['file_id'],
                'path': jd['path'],
                'label': jd['label'],
                'target': jd['target'],
            }
            base.update(feats)
            rows.append(base)

    feat_df = pd.DataFrame(rows)
    # Sort columns for stability
    cols_order = ['split', 'file_id', 'path', 'label', 'target']
    other_cols = sorted([c for c in feat_df.columns if c not in cols_order])
    feat_df = feat_df[cols_order + other_cols]
    return feat_df


# ----------------------------- Split Management -----------------------------

def build_splits(feat_df: pd.DataFrame, split_cfg: SplitConfig) -> pd.Series:
    """Create a new column 'cv_split' in {train,val,test,eval}.

    - Labeled rows (target in {0,1}): stratified split into train/val/test.
    - Unlabeled eval rows keep cv_split='eval'.
    """
    from sklearn.model_selection import train_test_split

    df_lab = feat_df[feat_df['target'].isin([0, 1])].copy()
    df_eval = feat_df[~feat_df['target'].isin([0, 1])].copy()

    # First, hold out test
    df_trainval, df_test = train_test_split(
        df_lab,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
        stratify=df_lab['target'],
    )

    # Then carve validation out of trainval
    val_size_rel = split_cfg.validation_size / (1.0 - split_cfg.test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size_rel,
        random_state=split_cfg.random_state,
        stratify=df_trainval['target'],
    )

    cv_split = pd.Series(index=feat_df.index, dtype='string')
    cv_split.loc[df_train.index] = 'train'
    cv_split.loc[df_val.index] = 'val'
    cv_split.loc[df_test.index] = 'test'
    cv_split.loc[df_eval.index] = 'eval'
    return cv_split


# ------------------------- Combination (subset) logic ------------------------

def group_columns_from_df(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Discover column name groups -> list of columns for that feature group.

    We use prefixes that match our group names. For scalar groups (e.g., zcr_mean),
    the column is exactly that name. For vector groups, columns start with
    '{group}_' (e.g., 'mfcc_mean_01').
    """
    groups: Dict[str, List[str]] = {g: [] for g in FEATURES_LIST}
    for col in df.columns:
        if col in {'split', 'file_id', 'path', 'label', 'target', 'cv_split'}:
            continue
        for g in FEATURES_LIST:
            if col == g or col.startswith(g + '_'):
                groups[g].append(col)
                break
    # Ensure stable order
    for g in groups:
        groups[g] = sorted(groups[g])
    return groups


def all_combo_codes() -> List[str]:
    letters = [FEATURE_NAME_MAPPING[g] for g in FEATURES_LIST]
    # All non-empty subsets
    codes: List[str] = []
    for r in range(1, len(letters) + 1):
        for combo in itertools.combinations(letters, r):
            codes.append(''.join(sorted(combo)))
    return codes


def normalize_codes_to_sorted_unique(codes: Iterable[str]) -> List[str]:
    norm = set(''.join(sorted(c.strip().upper())) for c in codes if c.strip())
    return sorted(norm)


def columns_for_code(code: str, group_cols: Dict[str, List[str]]) -> List[str]:
    cols: List[str] = []
    for ch in code:
        g = FEATURE_NAME_REVERSE_MAPPING[ch]
        cols.extend(group_cols.get(g, []))
    return cols


def write_npz(out_path: Path, X: np.ndarray, y: Optional[np.ndarray], columns: List[str], combo_code: str):
    np.savez_compressed(
        out_path,
        X=X,
        y=y if y is not None else np.array([]),
        columns=np.array(columns),
        combo_code=np.array(combo_code),
    )


def materialize_combos(
    feat_df: pd.DataFrame,
    out_dir: Path,
    codes: List[str],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / 'combos_meta.json'

    group_cols = group_columns_from_df(feat_df)

    # Save a handy meta
    meta = {
        'features_list': FEATURES_LIST,
        'feature_name_mapping': FEATURE_NAME_MAPPING,
        'feature_name_reverse_mapping': FEATURE_NAME_REVERSE_MAPPING,
        'groups_to_columns': group_cols,
        'num_rows': int(len(feat_df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Filter labeled rows
    df_lab = feat_df[feat_df['cv_split'].isin(['train', 'val', 'test'])].copy()

    # Pre-slice by split
    split_idx = {s: df_lab[df_lab['cv_split'] == s].index for s in ['train', 'val', 'test']}

    # We'll keep a view of the data matrix to speed slicing
    base_cols = sorted([c for c in df_lab.columns if c not in {'split', 'file_id', 'path', 'label', 'target', 'cv_split'}])
    M = df_lab[base_cols]
    y_all = df_lab['target'].astype('int16').to_numpy()

    # Build a small reverse map from column name -> position in M
    col_to_pos = {c: i for i, c in enumerate(base_cols)}

    # Helpers
    def slice_for(code: str) -> Tuple[List[str], np.ndarray]:
        cols = columns_for_code(code, group_cols)
        pos = [col_to_pos[c] for c in cols]
        return cols, pos

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = lambda x, **k: x

    # Iterate codes
    for code in tqdm(codes, desc='Combos'):
        cols, pos = slice_for(code)
        if not cols:
            continue

        for split in ['train', 'val', 'test']:
            idx = split_idx[split]
            X = M.iloc[idx, pos].to_numpy(dtype=np.float32, copy=False)
            y = y_all[idx]
            split_dir = out_dir / 'combos' / split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_path = split_dir / f'{code}.npz'
            write_npz(out_path, X, y, cols, code)


# ----------------------------------- CLI ------------------------------------

def cmd_extract(args):
    cfg = ExtractConfig(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir or (Path(args.data_root) / 'index')),
        n_mels=args.n_mels,
        n_frames=args.n_frames,
        sampling_rate=args.sampling_rate,
        fmax=args.fmax,
        window_length_ms=args.window_length_ms,
        workers=args.workers,
    )
    split_cfg = SplitConfig(
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print('[1/4] Building dataset index...')
    df_index = build_index_table(cfg.data_root)

    # Simple checks
    for sp in ['train', 'dev', 'eval']:
        total = int((df_index['split'] == sp).sum())
        exists = int(df_index[df_index['split'] == sp]['path'].map(lambda p: Path(p).exists()).sum())
        print(f'  {sp}: {exists}/{total} audio files found on disk')

    # Keep only rows whose audio actually exist on disk (avoids noisy errors)
    df_index = df_index[df_index['path'].map(lambda p: Path(p).exists())].reset_index(drop=True)

    print('[2/4] Extracting features...')
    feat_df = extract_all_features(df_index, cfg)

    print('[3/4] Creating CV splits...')
    feat_df['cv_split'] = build_splits(feat_df, split_cfg)

    print('[4/4] Writing Parquet + CSV...')
    parquet_path = cfg.out_dir / 'features_all.parquet'
    csv_path = cfg.out_dir / 'features_all.csv'

    # Save
    feat_df.to_parquet(parquet_path, index=False)
    feat_df.to_csv(csv_path, index=False)

    # Also save a minimal meta
    meta = {
        'extract_config': asdict(cfg),
        'split_config': asdict(split_cfg),
        'features_list': FEATURES_LIST,
        'feature_name_mapping': FEATURE_NAME_MAPPING,
    }
    (cfg.out_dir / 'features_meta.json').write_text(json.dumps(meta, indent=2, default=str))

    print(f'[✓] Done. Saved: {parquet_path} and {csv_path}')


def cmd_combos(args):
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir or (data_root / 'index'))
    parquet_path = out_dir / 'features_all.parquet'
    if not parquet_path.exists():
        raise SystemExit(f"Parquet not found: {parquet_path}. Run 'extract' first.")

    print('[1/3] Loading features parquet...')
    feat_df = pd.read_parquet(parquet_path)

    # Build/validate requested codes
    if args.all:
        codes = all_combo_codes()
    else:
        codes = normalize_codes_to_sorted_unique(args.codes)
        if not codes:
            raise SystemExit('No combo codes supplied. Use --all or --codes ...')

    print(f'[2/3] Preparing to materialize {len(codes)} combinations...')
    materialize_combos(feat_df, out_dir, codes)

    print('[3/3] Done.')


def cmd_list(args):
    print('Feature groups and their letters:')
    for g in FEATURES_LIST:
        print(f"  {FEATURE_NAME_MAPPING[g]}: {g}")

    print('\nAll non-empty combo count:', 2 ** len(FEATURES_LIST) - 1)


# ---------------------------------- Main ------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser(description='ASVspoof LA features + combos')
    sub = p.add_subparsers(dest='cmd', required=True)

    # extract
    pe = sub.add_parser('extract', help='Extract features and build splits')
    pe.add_argument('--data-root', type=str, default='database/data/asvspoof2019')
    pe.add_argument('--out-dir', type=str, default=None)
    pe.add_argument('--n-mels', type=int, default=DEFAULTS['n_mels'])
    pe.add_argument('--n-frames', type=int, default=DEFAULTS['n_frames'])
    pe.add_argument('--sampling-rate', type=int, default=DEFAULTS['sampling_rate'])
    pe.add_argument('--fmax', type=int, default=DEFAULTS['fmax'])
    pe.add_argument('--window-length-ms', type=float, default=DEFAULTS['window_length_ms'])
    pe.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1))
    pe.add_argument('--validation-size', type=float, default=DEFAULTS['validation_size'])
    pe.add_argument('--test-size', type=float, default=DEFAULTS['test_size'])
    pe.add_argument('--random-state', type=int, default=DEFAULTS['random_state'])
    pe.set_defaults(func=cmd_extract)

    # combos
    pc = sub.add_parser('combos', help='Materialize feature combos into NPZs')
    pc.add_argument('--data-root', type=str, default='database/data/asvspoof2019')
    pc.add_argument('--out-dir', type=str, default=None)
    pc.add_argument('--all', action='store_true', help='Generate ALL non-empty combos')
    pc.add_argument('--codes', nargs='*', default=[], help='Specific combo codes, e.g. AB, AEM, M, ABLK')
    pc.set_defaults(func=cmd_combos)

    # list
    pl = sub.add_parser('list', help='Show mapping and combo count')
    pl.set_defaults(func=cmd_list)

    return p


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
