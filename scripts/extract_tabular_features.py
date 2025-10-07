import os, argparse, numpy as np, joblib, pandas as pd
from glob import glob

def agg_stats(mat):
    # mat: (T, F) sau (F, T) - noi salvăm (T, F)
    # extragem câteva statistici robuste pe fiecare coloană de features
    stats = {}
    X = mat
    stats["mean"] = np.nanmean(X, axis=0)
    stats["std"]  = np.nanstd(X, axis=0)
    stats["p25"]  = np.nanpercentile(X, 25, axis=0)
    stats["p75"]  = np.nanpercentile(X, 75, axis=0)
    # concatenăm
    return np.concatenate([stats["mean"], stats["std"], stats["p25"], stats["p75"]], axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted_dir", default="dataset/extracted_features", help="folder cu X_*.npy, y_*.npy")
    ap.add_argument("--split", choices=["train","val","test"], required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    X = np.load(os.path.join(args.extracted_dir, f"X_{args.split}.npy"))
    y = np.load(os.path.join(args.extracted_dir, f"y_{args.split}.npy"))
    rows = []
    for i in range(X.shape[0]):
        feats = agg_stats(X[i])  # (4 * n_features)
        rows.append(np.concatenate([feats, [y[i]]]))
    cols = [f"f{j}" for j in range(len(rows[0])-1)] + ["label"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv} with shape {df.shape}")
if __name__ == "__main__":
    main()
