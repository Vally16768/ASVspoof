import os, argparse, numpy as np, pandas as pd
def agg_stats(mat):
    return np.concatenate([
        np.nanmean(mat, axis=0),
        np.nanstd(mat, axis=0),
        np.nanpercentile(mat, 25, axis=0),
        np.nanpercentile(mat, 75, axis=0),
    ], axis=0)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--extracted_dir", default="dataset/extracted_features")
    ap.add_argument("--split", choices=["train","val","test"], required=True)
    ap.add_argument("--out_csv", required=True)
    a=ap.parse_args()
    X=np.load(os.path.join(a.extracted_dir, f"X_{a.split}.npy"))
    y=np.load(os.path.join(a.extracted_dir, f"y_{a.split}.npy"))
    rows=[]
    for i in range(X.shape[0]):
        feats=agg_stats(X[i])
        rows.append(np.concatenate([feats,[y[i]]]))
    cols=[f"f{j}" for j in range(len(rows[0])-1)]+["label"]
    df=pd.DataFrame(rows, columns=cols)
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    df.to_csv(a.out_csv, index=False)
    print(f"[OK] {a.out_csv} -> {df.shape}")
if __name__=="__main__": main()
