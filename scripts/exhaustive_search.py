import itertools, argparse, numpy as np, pandas as pd, os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def kfold_auc(X, y, n_splits=5, seed=42):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs=[]
    for tr, va in kf.split(X,y):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, n_jobs=None))
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y[va], p))
    return float(np.mean(aucs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)           # train_tabular.csv
    ap.add_argument("--max_features", type=int, default=None) # dacă vrei să limitezi mărimea subsetului
    ap.add_argument("--out_txt", default=os.path.join("temp_data", "combinations_accuracy.txt"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    y = df["label"].values
    X = df.drop(columns=["label"]).values
    d = X.shape[1]
    idx = list(range(d))
    os.makedirs("temp_data", exist_ok=True)

    with open(args.out_txt,"w") as f:
        for r in range(1, (args.max_features or d)+1):
            for comb in itertools.combinations(idx, r):
                auc = kfold_auc(X[:, comb], y)
                f.write(f"accuracy: {auc:.2%}\n")
                f.write(f"combination: {','.join(map(str, comb))}\n")
    print("Done:", args.out_txt)
if __name__ == "__main__":
    main()
