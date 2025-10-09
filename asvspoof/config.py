#config.py
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List

# Feature groups and mapping letters (stable, used by combos)
FEATURES_LIST: List[str] = [
    "mfcc_mean",
    "mfcc_std",
    "spec_centroid_mean",
    "spec_bw_mean",
    "spec_contrast_mean",
    "spec_rolloff_mean",
    "rms_mean",
    "zcr_mean",
    "chroma_cens_mean",
    "chroma_cqt_mean",
    "wavelet_mean",
    "chroma_stft_mean",
    "wavelet_std",
    "pitch_mean",
    "pitch_std",
]

FEATURE_NAME_MAPPING: Dict[str, str] = {
    "mfcc_mean": "A",
    "mfcc_std": "B",
    "spec_centroid_mean": "C",
    "spec_bw_mean": "D",
    "spec_contrast_mean": "E",
    "spec_rolloff_mean": "F",
    "rms_mean": "G",
    "chroma_cens_mean": "H",
    "chroma_cqt_mean": "I",
    "wavelet_mean": "J",
    "chroma_stft_mean": "K",
    "wavelet_std": "L",
    "zcr_mean": "M",
    "pitch_mean": "N",
    "pitch_std": "O",
}
FEATURE_NAME_REVERSE_MAPPING: Dict[str, str] = {
    v: k for k, v in FEATURE_NAME_MAPPING.items()
}

DEFAULTS = dict(
    n_mels=128,
    n_frames=1024,          # kept for context/compat
    test_size=0.10,
    random_state=42,
    epochs=400,
    batch_size=8,
    sampling_rate=44100,
    fmax=22050,
    window_length_ms=15.0,
    validation_size=0.15,
)

@dataclass
class ExtractConfig:
    """Configuration for feature extraction and dataset paths."""
    data_root: Path
    out_dir: Path
    n_mels: int = DEFAULTS["n_mels"]
    n_frames: int = DEFAULTS["n_frames"]
    sampling_rate: int = DEFAULTS["sampling_rate"]
    fmax: int = DEFAULTS["fmax"]
    window_length_ms: float = DEFAULTS["window_length_ms"]
    workers: int = max(1, os.cpu_count() or 1)

@dataclass
class SplitConfig:
    """Configuration for splitting labeled data into train/val/test."""
    validation_size: float = DEFAULTS["validation_size"]
    test_size: float = DEFAULTS["test_size"]
    random_state: int = DEFAULTS["random_state"]
