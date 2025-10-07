import numpy as np
import librosa

def _safe_feature(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32, copy=False)

def extract_features(
    file_path: str,
    target_sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mfcc: int = 40,
) -> np.ndarray:
    """
    Întoarce un array 2D de formă (T, F) cu feature-uri aliniate pe timp:
      [MFCC(40) | delta MFCC(40) | centroid(1) | bandwidth(1) | contrast(7)
       | rolloff(1) | flatness(1) | chroma_stft(12) | rms(1) | zcr(1)]
    Total F = 40 + 40 + 1 + 1 + 7 + 1 + 1 + 12 + 1 + 1 = 105
    """
    # încărcare audio (mono), resample la target_sr
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)

    # caracteristici frame-wise (shape: (feat_dim, T))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

    # aliniere pe același număr de frame-uri (T = min T_i)
    T = min(
        mfcc.shape[1], mfcc_delta.shape[1], centroid.shape[1], bandwidth.shape[1],
        contrast.shape[1], rolloff.shape[1], flatness.shape[1], chroma.shape[1],
        rms.shape[1], zcr.shape[1]
    )

    feats = [
        mfcc[:, :T], mfcc_delta[:, :T],
        centroid[:, :T], bandwidth[:, :T], contrast[:, :T],
        rolloff[:, :T], flatness[:, :T], chroma[:, :T],
        rms[:, :T], zcr[:, :T]
    ]

    # concat pe axa feature -> (sum_feat_dim, T), apoi transpose la (T, F)
    F = np.concatenate([_safe_feature(f) for f in feats], axis=0).T
    return F  # (T, 105)
