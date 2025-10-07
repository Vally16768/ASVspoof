import argparse,itertools,os,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def kfold_auc(X,y,n_splits=5,seed=42):
    kf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    aucs=[]
    for tr,va in kf.split(X,y):
        clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear"))
        clf.fit(X[tr],y[tr])
        p=clf.predict_proba(X[va])[:,1]
        aucs.append(roc_auc_score(y[va],p))
    return float(np.mean(aucs))
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_txt", default=os.path.join("temp_data","combinations_accuracy.txt"))
    ap.add_argument("--max_features", type=int, default=None)
    a=ap.parse_args()
    df=pd.read_csv(a.csv); y=df["label"].values; X=df.drop(columns=["label"]).values
    d=X.shape[1]
    os.makedirs(os.path.dirname(a.out_txt), exist_ok=True)
    with open(a.out_txt,"w") as f:
        upper=(a.max_features or d)
        for r in range(1, upper+1):
            for comb in itertools.combinations(range(d), r):
                auc=kfold_auc(X[:,comb],y)
                f.write(f"accuracy: {auc:.2%}\n")
                f.write(f"combination: {','.join(map(str,comb))}\n")
    print("[OK] scris:", a.out_txt)
if __name__=="__main__": main()
