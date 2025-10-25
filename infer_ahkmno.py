# infer_ahkmno.py
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

# asvspoof stack
from asvspoof.features import (
    _load_audio_strict,
    _frame_params,
    _chroma_numpy,
    extract_features_for_path,
)
try:
    from asvspoof.config import ExtractConfig
except Exception:
    # Fallback minimal dacă nu avem config-ul tău: doar câmpurile necesare inferenței
    class ExtractConfig:
        def __init__(self, sampling_rate=16000, window_length_ms=10, fmax=8000, n_mels=128):
            self.sampling_rate = sampling_rate
            self.window_length_ms = window_length_ms
            self.fmax = fmax
            self.n_mels = n_mels

import librosa

# --------------------------
# Helperi: softmax & scaler
# --------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def try_load_scaler(path: Path):
    try:
        import joblib
        if path.exists():
            return joblib.load(path)
    except Exception:
        return None
    return None

# ------------------------------------------
# Extractori care folosesc features.py core
# ------------------------------------------
def extract_AHKMNO(y: np.ndarray, sr: int, cfg: ExtractConfig) -> np.ndarray:
    """
    A: MFCC (mean+std)            -> from features.py (13 coef)
    H: CHROMA (mean)              -> from features.py (12 clase)
    K: Spectral contrast (mean)   -> from features.py
    M: Mel-spectrogram (mean+std) -> computed here, same params as cfg
    N: Tonnetz (mean+std)         -> computed here
    O: ZCR (mean)                 -> from features.py
    """
    # Ferestrare identică cu features.py
    n_fft, hop = _frame_params(sr, cfg.window_length_ms)

    # 1) Folosim pipeline-ul oficial ca să obținem ce e deja acolo (A,H,K + O + bazice)
    # Construim un DataFrame "index" de un singur fișier pentru a apela extract_features_for_path elegant.
    import pandas as pd
    tmp_df = pd.DataFrame([{
        "split": "infer",
        "file_id": "audio",
        "abs_path": "__MEMORY__",  # placeholder
        "label": None,
        "target": None,
    }])

    # Dar extract_features_for_path cere Path; îl apelăm direct în loc de extract_all_features.
    # Refolosim exact logica internă pentru A,H,K,O printr-un apel controlat:
    # -> copiem bucăți necesare aici fără să modificăm features.py

    # === A: MFCC (mean+std pe 13) ===
    fmax_safe   = float(min(cfg.fmax, (sr / 2.0) - 1.0))
    n_mels_safe = int(min(cfg.n_mels, max(8, n_fft // 4)))
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop,
        n_mels=n_mels_safe, fmax=fmax_safe
    )
    A_mean = np.mean(mfcc, axis=1)
    A_std  = np.std(mfcc,  axis=1)
    A_vec  = np.concatenate([A_mean, A_std], axis=0).astype(np.float32)  # (26,)

    # === H: CHROMA (NumPy din features.py) -> mean pe 12 clase ===
    chroma = _chroma_numpy(y, sr, n_fft, hop)  # (12, T)
    H_vec = np.mean(chroma, axis=1).astype(np.float32)  # (12,)

    # === K: Spectral contrast (mean pe benzi) ===
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    K_vec = np.mean(spec_contrast, axis=1).astype(np.float32)  # tipic (7,)

    # === M: Mel-spectrogram bands (mean+std per band) ===
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels_safe, n_fft=n_fft, hop_length=hop
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    M_mean = np.mean(mel_db, axis=1)
    M_std  = np.std(mel_db,  axis=1)
    M_vec  = np.concatenate([M_mean, M_std], axis=0).astype(np.float32)  # (2 * n_mels_safe,)

    # === N: Tonnetz (mean+std pe 6 dim) ===
    y_harm = librosa.effects.harmonic(y)
    chroma_cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chroma_cqt, sr=sr)  # (6, T)
    N_mean = np.mean(tonnetz, axis=1)
    N_std  = np.std(tonnetz,  axis=1)
    N_vec  = np.concatenate([N_mean, N_std], axis=0).astype(np.float32)  # (12,)

    # === O: Zero Crossing Rate (mean) ===
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    O_vec = np.array([float(np.mean(zcr))], dtype=np.float32)  # (1,)

    # Concatenare în ordinea specificată A H K M N O
    return np.concatenate([A_vec, H_vec, K_vec, M_vec, N_vec, O_vec], axis=0)

# --------------------------
# Aliniere la inputul model
# --------------------------
def align_to_model_input(x: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    in_shape = model.inputs[0].shape
    feat_dim = in_shape[-1] if in_shape and in_shape[-1] is not None else None
    x = x.flatten().astype(np.float32)
    if feat_dim is None:
        return x[None, :]
    need = int(feat_dim)
    if x.size > need:
        x = x[:need]
    elif x.size < need:
        x = np.pad(x, (0, need - x.size))
    return x[None, :]

# -------------
# Main routine
# -------------
def main():
    p = argparse.ArgumentParser(description="Inferență pe model .keras cu feature combo AHKMNO (compatibil features.py).")
    p.add_argument("--model",  type=str, default="final_model/best_model.keras")
    p.add_argument("--audio",  type=str, default="final_model/fake.wav")
    p.add_argument("--labels", type=str, default="")  # opțional
    p.add_argument("--sr",     type=int, default=1600*10)  # 16000
    p.add_argument("--scaler", type=str, default="final_model/scaler.pkl")
    args = p.parse_args()

    model_path  = Path(args.model)
    audio_path  = Path(args.audio)
    labels_path = Path(args.labels) if args.labels else None
    scaler_path = Path(args.scaler) if args.scaler else None

    if not model_path.exists():
        raise FileNotFoundError(f"Modelul nu a fost găsit: {model_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Fișierul audio nu a fost găsit: {audio_path}")

    # Config identic cu pipeline-ul tău
    cfg = ExtractConfig(
        sampling_rate=args.sr,
        window_length_ms=10,
        fmax=8000,
        n_mels=128,
    )

    # Load audio cu rutina strictă din features.py (mono + resample + trim)
    y, sr = _load_audio_strict(audio_path, cfg.sampling_rate)

    # Extrage AHKMNO cu aceiași parametri de ferestrare ai pipeline-ului
    x = extract_AHKMNO(y, sr, cfg)  # vector 1D

    # (opțional) scaler sklearn dacă există
    scaler = try_load_scaler(scaler_path) if scaler_path else None
    if scaler is not None:
        x = scaler.transform(x.reshape(1, -1)).flatten()

    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)

    # Potrivire la inputul cerut de model
    X = align_to_model_input(x, model)

    # Inferență
    logits = model.predict(X, verbose=0)
    logits = np.array(logits)
    if logits.ndim == 1:
        logits = logits[None, :]
    probs = softmax(logits)

    # Afișare rezultate
    if labels_path and labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        if len(labels) == probs.shape[1]:
            pairs = sorted(zip(labels, probs[0].tolist()), key=lambda t: t[1], reverse=True)
            print("\n=== Rezultat inferență (etichetă : p) ===")
            for lab, p_ in pairs:
                print(f"{lab:>20s} : {p_:.4f}")
            print(f"\nPredicție: {pairs[0][0]} (p={pairs[0][1]:.4f})")
        else:
            print("[!] labels.txt dimension mismatch; afișez indici.")
            for i, p_ in enumerate(probs[0]):
                print(f"{i:>3d} : {p_:.4f}")
            top = int(np.argmax(probs[0]))
            print(f"\nPredicție: clasa index {top} (p={probs[0, top]:.4f})")
    else:
        print("\n=== Rezultat inferență (index : p) ===")
        for i, p_ in enumerate(probs[0]):
            print(f"{i:>3d} : {p_:.4f}")
        top = int(np.argmax(probs[0]))
        print(f"\nPredicție: clasa index {top} (p={probs[0, top]:.4f})")

    # Log util
    try:
        need = model.inputs[0].shape[-1]
        print(f"\n[i] Dim featuri AHKMNO extrase: {x.size}")
        print(f"[i] Dim cerută de model (ultimul dim): {int(need) if need is not None else 'necunoscut'}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
