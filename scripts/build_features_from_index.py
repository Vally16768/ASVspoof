#!/usr/bin/env python3
import argparse, csv, subprocess, shutil
from pathlib import Path
import numpy as np, pandas as pd

# ---- setări MFCC (simple, fără librosa) ----
def stft_frames(y, sr, win_len=0.025, hop_len=0.010):
    Nw = int(win_len * sr)
    Nh = int(hop_len * sr)
    if len(y) < Nw:  # pad minim
        y = np.pad(y, (0, Nw - len(y)))
    n_frames = 1 + (len(y) - Nw) // Nh if len(y) >= Nw else 1
    win = np.hanning(Nw).astype(np.float32)
    frames = np.stack([y[i*Nh:i*Nh+Nw] * win for i in range(n_frames)], axis=0)
    return frames

def power_spectrum(frames, n_fft):
    # zero-pad la n_fft, apoi FFT
    pad = n_fft - frames.shape[1]
    if pad > 0:
        frames = np.pad(frames, ((0,0),(0,pad)))
    X = np.fft.rfft(frames, n=n_fft, axis=1)
    return (np.abs(X) ** 2).astype(np.float32)

def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    # implementare compactă de fbank
    fmax = fmax or (sr/2)
    def hz_to_mel(f): return 2595.0 * np.log10(1.0 + f/700.0)
    def mel_to_hz(m): return 700.0 * (10**(m/2595.0) - 1.0)

    m_min, m_max = hz_to_mel(fmin), hz_to_mel(fmax)
    m_points = np.linspace(m_min, m_max, n_mels+2)
    f_points = mel_to_hz(m_points)
    bins = np.floor((n_fft+1) * f_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft//2 + 1), dtype=np.float32)
    for m in range(1, n_mels+1):
        f_m_minus, f_m, f_m_plus = bins[m-1], bins[m], bins[m+1]
        if f_m_minus == f_m: f_m -= 1
        if f_m == f_m_plus: f_m_plus += 1
        for k in range(f_m_minus, f_m):
            if 0 <= k < fb.shape[1]:
                fb[m-1, k] = (k - f_m_minus) / max(1, (f_m - f_m_minus))
        for k in range(f_m, f_m_plus):
            if 0 <= k < fb.shape[1]:
                fb[m-1, k] = (f_m_plus - k) / max(1, (f_m_plus - f_m))
    return fb

def dct_type2(M, N):
    # DCT-II simplă (matrix form) pentru MFCC
    k = np.arange(N)
    n = np.arange(M)[:, None]
    dct = np.cos(np.pi / N * (k + 0.5) * n)
    dct[0] *= 1/np.sqrt(2.0)
    return dct * np.sqrt(2.0 / N)

def mfcc_from_y(y, sr, n_mfcc=40, n_mels=64, n_fft=None, win_len=0.025, hop_len=0.010):
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(win_len * sr)))
    frames = stft_frames(y, sr, win_len, hop_len)        # (T, Nw)
    S = power_spectrum(frames, n_fft)                    # (T, n_fft//2+1)
    fb = mel_filterbank(sr, n_fft, n_mels)               # (n_mels, n_bins)
    E = np.maximum(1e-10, S @ fb.T)                      # (T, n_mels)
    logE = np.log(E)
    D = dct_type2(n_mfcc, n_mels)                        # (n_mfcc, n_mels)
    C = (D @ logE.T).T                                   # (T, n_mfcc)
    return C.astype(np.float32)                          # frames x n_mfcc

def agg_stats(mat):  # (T, D) -> (4*D,)
    # statistici simple pe timp: mean/std/q25/q75
    if mat.ndim == 1:
        mat = mat[:, None]
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    q25  = np.nanquantile(mat, 0.25, axis=0)
    q75  = np.nanquantile(mat, 0.75, axis=0)
    return np.concatenate([mean, std, q25, q75], axis=0).astype(np.float32)

def ffmpeg_decode_mono_f32(path: Path, sr: int) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg nu este instalat. `sudo apt-get install -y ffmpeg`")
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore"))
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    return y

def build_A_feature(y, sr, n_mfcc=40):
    C = mfcc_from_y(y, sr, n_mfcc=n_mfcc)  # (T, n_mfcc)
    vec = agg_stats(C)                      # (4*n_mfcc,)
    names = [f"mfcc_{stat}{i}" for stat in ("_mean_","_std_","_q25_","_q75_") for i in range(C.shape[1])]
    return vec, names

def read_index(csv_path: Path):
    df = pd.read_csv(csv_path)
    col = "path" if "path" in df.columns else ("relpath" if "relpath" in df.columns else None)
    if col is None or "label" not in df.columns:
        raise SystemExit(f"[!] {csv_path} trebuie să aibă coloane path/relpath și label")
    return df[[col, "label"]].rename(columns={col: "relpath"})

def process_split(root: Path, split_csv: Path, out_parquet: Path, sr: int, combo: str):
    df = read_index(split_csv)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # doar combinația A (MFCC)
    if combo.upper() != "A":
        raise SystemExit("[!] În acest extractor rapid suport doar combinația 'A' (MFCC).")

    # obține dimensiunea exactă a vectorului pt. fallback NAN
    dummy = np.zeros(int(sr * 1.0), dtype=np.float32)  # 1s tăcere
    _, feat_names = build_A_feature(dummy, sr)
    F = []
    for i, row in df.iterrows():
        wav = (root / row.relpath).resolve()
        try:
            y = ffmpeg_decode_mono_f32(wav, sr)
            vec, _ = build_A_feature(y, sr)
            F.append(vec)
        except Exception as e:
            print(f"[!] {wav}: {e}")
            F.append(np.full((len(feat_names),), np.nan, dtype=np.float32))
    X = np.vstack(F)
    out_df = pd.DataFrame(X, columns=feat_names)
    out_df["label"] = (df["label"].astype(str).str.lower().isin(["bonafide","real","1","genuine"])).astype(int)
    out_df["relpath"] = df["relpath"]
    out_df.to_parquet(out_parquet, index=False)
    print(f"[✓] {out_parquet}  ({len(out_df)} rânduri, {X.shape[1]} coloane)")

def main():
    ap = argparse.ArgumentParser(description="Build tabular MFCC features (combinația A) fără librosa/ bz2")
    ap.add_argument("--data-root", required=True, help="ex: /.../asvspoof2019")
    ap.add_argument("--combo", default="A")
    ap.add_argument("--sr", type=int, default=16000)
    args = ap.parse_args()

    root = Path(args.data_root)
    idx = root / "index"
    out_dir = Path("features_tabular") / args.combo.upper()

    train_csv = idx / "train.csv"
    val_csv   = idx / "val.csv"
    test_csv  = idx / "test.csv"
    if not train_csv.exists():
        raise SystemExit(f"[!] Lipsesc CSV-urile în {idx} (nu găsesc train.csv).")

    process_split(root, train_csv, out_dir/"train.parquet", args.sr, args.combo)
    if val_csv.exists():  process_split(root, val_csv,  out_dir/"val.parquet",  args.sr, args.combo)
    if test_csv.exists(): process_split(root, test_csv, out_dir/"test.parquet", args.sr, args.combo)

if __name__ == "__main__":
    main()
