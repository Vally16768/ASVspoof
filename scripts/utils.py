import numpy as np
import librosa

def load_audio_mono(path, target_sr=16000):
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def pad_or_trim(y, sr, target_seconds=4.0):
    target_len = int(target_seconds * sr)
    if len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y[:target_len]

def extract_basic_features(y, sr, n_mfcc=20):
    # MFCC + delta + delta-delta (statistici)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(F):
        return np.concatenate([F.mean(axis=1), F.std(axis=1), F.min(axis=1), F.max(axis=1)])

    mfcc_stats = stats(mfcc)
    d1_stats = stats(d1)
    d2_stats = stats(d2)

    # Alte trăsături simple
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sro = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]

    zcr_stats = np.array([zcr.mean(), zcr.std(), zcr.min(), zcr.max()])
    sc_stats  = np.array([sc.mean(),  sc.std(),  sc.min(),  sc.max()])
    sro_stats = np.array([sro.mean(), sro.std(), sro.min(), sro.max()])

    feat = np.concatenate([mfcc_stats, d1_stats, d2_stats, zcr_stats, sc_stats, sro_stats])
    return feat

def label_to_int(lbl: str):
    m = {"bonafide": 0, "spoof": 1}
    return m[lbl.strip().lower()]
