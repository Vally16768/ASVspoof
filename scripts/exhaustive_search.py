import argparse, itertools, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os, sys

def kfold_auc(X, y, n_splits=5, seed=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs=[]
    for tr, va in kf.split(X,y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear"))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_txt", default=os.path.join("temp_data","combinations_accuracy.txt"))
    ap.add_argument("--max_features", type=int, default=None, help="limitează dimensiunea subsetului (ex. 12)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df["label"].values
    X = df.drop(columns=["label"]).values
    d = X.shape[1]
    idx = list(range(d))
    os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)

    with open(args.out_txt, "w") as f:
        upper = (args.max_features or d)
        for r in range(1, upper+1):
            for comb in itertools.combinations(idx, r):
                auc = kfold_auc(X[:, comb], y)
                f.write(f"accuracy: {auc:.2%}\n")
                f.write(f"combination: {','.join(map(str, comb))}\n")
    print("[OK] scris:", args.out_txt)
if __name__ == "__main__":
    main()
