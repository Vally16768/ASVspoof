import argparse
from pathlib import Path
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import dump

def split_Xy(df):
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=200)
    args = ap.parse_args()

    train_df = pd.read_parquet(args.train)
    dev_df   = pd.read_parquet(args.dev)

    Xtr, ytr = split_Xy(train_df)
    Xdv, ydv = split_Xy(dev_df)

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.C, max_iter=args.max_iter, class_weight="balanced", solver="lbfgs")
    )
    pipe.fit(Xtr, ytr)

    # scor rapid pe dev
    proba = pipe.predict_proba(Xdv)[:, 1]
    auc = roc_auc_score(ydv, proba)
    print(f"ROC-AUC dev: {auc:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, args.out)
    print(f"Model salvat la: {args.out}")

if __name__ == "__main__":
    main()
