# === config.py (thin shim around constants) ===
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from . import constants as C

FEATURES_LIST: List[str] = list(C.combo_features_list)
FEATURE_NAME_MAPPING: Dict[str, str] = dict(C.combo_feature_name_mapping)
FEATURE_NAME_REVERSE_MAPPING: Dict[str, str] = dict(C.combo_feature_name_reverse_mapping)

@dataclass
class ExtractConfig:
    data_root: Path
    out_dir: Path  # typically <data_root>/<index_folder_name>
    sampling_rate: int = C.sampling_rate
    n_mels: int = C.n_mels
    fmax: int = C.fmax
    window_length_ms: float = C.window_length_ms
    workers: int = C.workers
