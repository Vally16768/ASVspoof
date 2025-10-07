import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
import soundfile as sf
import librosa

# Dimensiune feature finală: 105 (40 MFCC + 40 delta + 1+1+7+1+1+12+1+1)
def extract_one(path, sr=16000, n_fft=1024, hop=256, n_mfcc=40):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # mono
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    d1   = librosa.feature.delta(mfcc, order=1)
    cen  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    con  = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rol  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    flt  = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)
    chr  = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rms  = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)
    zcr  = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=hop)
    T = min(mfcc.shape[1], d1.shape[1], cen.shape[1], bw.shape[1], con.shape[1],
            rol.shape[1], flt.shape[1], chr.shape[1], rms.shape[1], zcr.shape[1])
    feats = np.concatenate([
        mfcc[:,:T], d1[:,:T], cen[:,:T], bw[:,:T], con[:,:T], rol[:,:T],
        flt[:,:T], chr[:,:T], rms[:,:T], zcr[:,:T]
    ], axis=0).T
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return feats  # (T, 105)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ASVSPOOF_ROOT","database/data/asvspoof2019"))
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="features/seq")
    ap.add_argument("--max_files", type=int, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    split = df["split"].iloc[0]
    out_split = outdir / split
    out_split.mkdir(parents=True, exist_ok=True)
    manifest = []

    it = df.itertuples(index=False)
    if args.max_files: it = list(it)[:args.max_files]

    for row in tqdm(it, total=(args.max_files or len(df))):
        utt_id, rel, sp, label, label_id = row
        if sp == "eval" or label_id == "" or (isinstance(label_id, float) and np.isnan(label_id)):
            # ignora eval sau linii fără etichete
            continue
        src = Path(args.root) / rel
        if not src.exists():
            continue
        X = extract_one(str(src))
        y = int(label_id)
        out_npz = out_split / f"{utt_id}.npz"
        np.savez_compressed(out_npz, X=X, y=y, utt_id=utt_id, label=label)
        manifest.append({"utt_id": utt_id, "frames": int(X.shape[0]), "feat_dim": int(X.shape[1]), "y": y})
    with open(outdir / f"{split}.jsonl", "w") as g:
        for m in manifest: g.write(json.dumps(m)+"\n")
    print(f"[OK] {split}: salvat {len(manifest)} fișiere în {out_split}")

if __name__=="__main__":
    main()
