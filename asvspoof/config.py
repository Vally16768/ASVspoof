# === config.py (thin shim around constants) ===
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import constants as C

try:
    FEATURES_LIST: List[str] = list(C.combo_features_list)
except AttributeError:
    # fallback minimal; va fi suficient pt. pornire
    FEATURES_LIST = [
        'zcr_mean','rms_mean','spec_centroid_mean','spec_bw_mean','spec_rolloff_mean',
        'pitch_mean','pitch_std'
    ]

try:
    FEATURE_NAME_MAPPING: Dict[str, str] = dict(C.combo_feature_name_mapping)
except AttributeError:
    FEATURE_NAME_MAPPING = {k: k for k in FEATURES_LIST}

try:
    FEATURE_NAME_REVERSE_MAPPING: Dict[str, str] = dict(C.combo_feature_name_reverse_mapping)
except AttributeError:
    FEATURE_NAME_REVERSE_MAPPING = {v: k for k, v in FEATURE_NAME_MAPPING.items()}

@dataclass
class ExtractConfig:
    data_root: Path
    out_dir: Path  # typically <data_root>/<index_folder_name>
    sampling_rate: int = C.sampling_rate
    n_mels: int = C.n_mels
    fmax: int = C.fmax
    window_length_ms: float = C.window_length_ms
    workers: int = C.workers
