import argparse, json, os
from pathlib import Path
import numpy as np
from glob import glob
from tqdm import tqdm

def load_npz_list(dir_glob):
    files = sorted(glob(dir_glob))
    Xs, ys = [], []
    for f in files:
        d = np.load(f)
        Xs.append(d["X"])
        ys.append(int(d["y"]))
    return Xs, np.array(ys, dtype="int64"), files

def pad_trunc(xs, T):
    out=[]
    F = xs[0].shape[1]
    for a in xs:
        if a.shape[0] >= T:
            out.append(a[:T])
        else:
            pad = np.zeros((T - a.shape[0], F), dtype=a.dtype)
            out.append(np.vstack([a, pad]))
    return np.stack(out, axis=0)  # (N, T, F)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_root", default="features/seq")
    ap.add_argument("--splits", nargs="+", default=["train","dev"])
    ap.add_argument("--outdir", default="dataset/extracted_features")
    ap.add_argument("--frames", type=int, default=400, help="lungime fixă pe timp (padding/trunc)")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    for sp in args.splits:
        Xs, y, files = load_npz_list(os.path.join(args.features_root, sp, "*.npz"))
        if not Xs:
            print(f"[WARN] Nu am găsit features pentru split={sp}")
            continue
        F = Xs[0].shape[1]
        X = pad_trunc(Xs, T=args.frames)
        np.save(out / f"X_{'val' if sp=='dev' else sp}.npy", X)
        np.save(out / f"y_{'val' if sp=='dev' else sp}.npy", y)
        print(f"[OK] {sp}: X shape {X.shape}  y shape {y.shape}  F={F}")

if __name__=="__main__":
    main()
