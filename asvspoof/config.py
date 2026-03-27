#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

# Importă configurația unică a repo-ului
import constants as C

# ---------------------------
# Helpers pentru citirea C.*
# ---------------------------
def _get(name: str, default):
    return getattr(C, name, default)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}

def _get_path_from_env_or_constants() -> Path:
    """
    Respectă ASVSPOOF_ROOT dacă este setat; altfel folosește constants.directory.
    """
    env = os.getenv("ASVSPOOF_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(_get("directory", "database/data/asvspoof2019")).expanduser().resolve()


# ---------------------------
# Dataset paths & folders
# ---------------------------
DATA_ROOT: Path = _get_path_from_env_or_constants()

INDEX_FOLDER_NAME: str   = _get("index_folder_name", "index")
RESULTS_FOLDER_NAME: str = _get("results_folder", "results")
TEMP_FOLDER_NAME: str    = _get("temp_data_folder_name", "temp_data")
MODELS_FOLDER_NAME: str  = _get("models_folder", "models")

# Fișiere artifacte comune (passthrough)
SAVE_EVAL_JSON: str          = _get("save_evaluation_model_results_file_name", "eval_results.json")
SAVE_BEST_COMBOS_TXT: str    = _get("save_the_best_combination_file_name", "combinations_ordered_by_accuracy.txt")
SAVE_COMBOS_TXT: str         = _get("save_combinations_file_name", "combinations_accuracy.txt")
TRAIN_LOG_CSV: str           = _get("train_log_filename", "train_log.csv")
BEST_MODEL_FILENAME: str     = _get("best_model_filename", "best_model.keras")
FINAL_MODEL_FILENAME: str    = _get("final_model_filename", "final_model.keras")
ACCURACY_TXT_FILENAME: str   = _get("accuracy_txt_filename", "accuracy.txt")
CLASS_REPORT_TXT: str        = _get("classification_report_filename", "classification_report.txt")
PREDICTIONS_CSV: str         = _get("predictions_csv_filename", "predictions.csv")
CONF_MAT_PNG: str            = _get("confusion_matrix_png_filename", "confusion_matrix.png")
COMBO_PRIMARY_METRIC: str    = str(_get("combo_primary_metric", "min_tDCF"))
COMBO_TIE_BREAK_METRIC: str  = str(_get("combo_tie_break_metric", "accuracy"))

# Subdirectoare ASVspoof 2019 LA (relative la DATA_ROOT)
LA_TRAIN_FLAC_SUBDIR: str = _get("la_train_flac_subdir", "ASVspoof2019_LA_train/flac")
LA_DEV_FLAC_SUBDIR: str   = _get("la_dev_flac_subdir",   "ASVspoof2019_LA_dev/flac")
LA_EVAL_FLAC_SUBDIR: str  = _get("la_eval_flac_subdir",  "ASVspoof2019_LA_eval/flac")
LA_PROTOCOLS_SUBDIR: str  = _get("la_protocols_subdir",  "ASVspoof2019_LA_cm_protocols")

# Protocoale
LA_TRAIN_TRN: str = _get("la_train_trn_filename", "ASVspoof2019.LA.cm.train.trn.txt")
LA_DEV_TRL: str   = _get("la_dev_trl_filename",   "ASVspoof2019.LA.cm.dev.trl.txt")
LA_EVAL_TRL: str  = _get("la_eval_trl_filename",  "ASVspoof2019.LA.cm.eval.trl.txt")

# Căi derivate utile
LA_TRAIN_FLAC_DIR: Path = DATA_ROOT / LA_TRAIN_FLAC_SUBDIR
LA_DEV_FLAC_DIR:   Path = DATA_ROOT / LA_DEV_FLAC_SUBDIR
LA_EVAL_FLAC_DIR:  Path = DATA_ROOT / LA_EVAL_FLAC_SUBDIR
LA_PROTOCOLS_DIR:  Path = DATA_ROOT / LA_PROTOCOLS_SUBDIR

INDEX_DIR: Path   = DATA_ROOT / INDEX_FOLDER_NAME
RESULTS_DIR: Path = DATA_ROOT / RESULTS_FOLDER_NAME
TEMP_DIR: Path    = DATA_ROOT / TEMP_FOLDER_NAME
MODELS_DIR: Path  = DATA_ROOT / MODELS_FOLDER_NAME


# ---------------------------
# Audio / Feature parameters (din constants.py cu fallback-uri)
# ---------------------------
SAMPLING_RATE: int        = int(_get("sampling_rate", 16000))
WINDOW_LENGTH_MS: float   = float(_get("window_length_ms", 25.0))
FMAX: float               = float(_get("fmax", 8000.0))
N_MELS: int               = int(_get("n_mels", 128))
RANDOM_STATE: int         = int(_get("random_state", 42))

# Extended feature params
LPCC_NUM_CEPS: int        = int(_get("lpcc_num_ceps", 30))
CQCC_NUM_CEPS: int        = int(_get("cqcc_num_ceps", 30))
SSL_WAV2VEC_MODEL_ID: str = str(_get("ssl_wav2vec_model_id", "facebook/wav2vec2-xls-r-300m"))
SSL_WAVLM_MODEL_ID: str   = str(_get("ssl_wavlm_model_id", "microsoft/wavlm-large"))
SSL_PCA_COMPONENTS: int   = int(_get("ssl_pca_components", 128))
SSL_REQUIRE_GPU: bool     = _as_bool(_get("ssl_require_gpu", True))
SSL_HF_CACHE_DIR: str     = str(_get("ssl_hf_cache_dir", "") or "")

# ---------------------------
# Grupurile macro folosite pentru combinații
# ---------------------------
FEATURES_LIST = [
    "zcr_rms",
    "spectral_basic",
    "spectral_contrast",
    "chroma",
    "mfcc",
    "pitch",
    "wavelets",
    "lpcc",
    "cqcc",
    "wav2vec",
    "wavlm",
]

# Nume „frumoase” pentru CLI / rapoarte
FEATURE_NAME_MAPPING = {
    "zcr_rms": "ZeroCross/RMS",
    "spectral_basic": "Centroid/Bandwidth/Rolloff",
    "spectral_contrast": "Spectral Contrast",
    "chroma": "Chroma (STFT/CQT/CENS)",
    "mfcc": "MFCC(13) mean/std",
    "pitch": "YIN pitch mean/std",
    "wavelets": "DWT(db4) stats",
    "lpcc": "LPCC(30)+delta+delta2 mean/std",
    "cqcc": "CQCC(30)+delta+delta2 mean/std",
    "wav2vec": "wav2vec2 mean/std + PCA",
    "wavlm": "WaveLM mean/std + PCA",
}

# Prefix aliases: ce prefixe de coloane intră în fiecare grup
# (acoperă exact ce produce features.py)
GROUP_ALIASES = {
    "zcr_rms": ["zcr", "rms"],
    "spectral_basic": ["spec_centroid", "spec_bw", "spec_rolloff"],
    "spectral_contrast": ["spec_contrast"],
    "chroma": ["chroma"],
    "mfcc": ["mfcc"],
    "pitch": ["pitch"],
    "wavelets": ["wavelet", "wavelets", "dwt", "wt"],
    "lpcc": ["lpcc"],
    "cqcc": ["cqcc"],
    "wav2vec": ["wav2vec", "w2v"],
    "wavlm": ["wavlm"],
}

# Maparea stabilă literă->grup pentru fișiere .npz (opțional, altfel se auto-atribuie)
FEATURE_NAME_REVERSE_MAPPING: dict[str, str] = {
    "A": "mfcc",
    "B": "lpcc",
    "C": "cqcc",
    "D": "wav2vec",
    "E": "wavlm",
    "H": "chroma",
    "K": "zcr_rms",
    "L": "spectral_basic",
    "M": "spectral_contrast",
    "N": "pitch",
    "O": "wavelets",
}

# ---------------------------
# ExtractConfig — STRICT secvențial 
# ---------------------------
@dataclass(frozen=True)
class ExtractConfig:
    data_root: Path = DATA_ROOT
    out_dir: Path   = INDEX_DIR

    # audio/feature params
    sampling_rate: int      = SAMPLING_RATE
    window_length_ms: float = WINDOW_LENGTH_MS
    n_mels: int             = N_MELS
    fmax: float             = FMAX

    # extended feature params
    random_state: int       = RANDOM_STATE
    lpcc_num_ceps: int      = LPCC_NUM_CEPS
    cqcc_num_ceps: int      = CQCC_NUM_CEPS
    ssl_wav2vec_model_id: str = SSL_WAV2VEC_MODEL_ID
    ssl_wavlm_model_id: str   = SSL_WAVLM_MODEL_ID
    ssl_pca_components: int = SSL_PCA_COMPONENTS
    ssl_require_gpu: bool   = SSL_REQUIRE_GPU
    ssl_hf_cache_dir: str   = SSL_HF_CACHE_DIR

# ------- Compat aliases pentru cod vechi -------
FEATURE_NAME_MAPPING_LETTERS: dict[str, str] = getattr(C, "feature_name_mapping", {})
