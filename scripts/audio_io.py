#!/usr/bin/env python3
from pathlib import Path
import numpy as np

# detectează bz2 (unii Python n-au _bz2 compilat)
try:
    import bz2  # noqa
    _HAS_BZ2 = True
except Exception:
    _HAS_BZ2 = False

def load_audio_any(path: Path, target_sr: int) -> np.ndarray:
    """
    Încearcă soundfile; dacă _bz2 lipsește sau dă eroare,
    cade pe librosa (audioread/ffmpeg). Returnează mono float32 la target_sr.
    """
    if _HAS_BZ2:
        try:
            import soundfile as sf
            y, sr = sf.read(str(path), always_2d=False, dtype="float32")
            if y.ndim > 1: y = y.mean(axis=1)
            if sr != target_sr:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            return y.astype("float32")
        except Exception:
            pass
    import librosa
    y, _sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype("float32")
