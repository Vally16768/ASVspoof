import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import roc_auc_score, roc_curve, det_curve

def eer_from_scores(y_true, scores):
    # FPR vs TPR -> EER = punctul unde FNR = FPR
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    # găsim punctul cel mai apropiat de egalitate
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fnr[idx] + fpr[idx]) / 2.0
    return float(eer)

def save_plot_roc(y_true, scores, out_png):
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def save_plot_det(y_true, scores, out_png):
    import matplotlib.pyplot as plt
    fpr, fnr, _ = det_curve(y_true, scores)
    plt.figure()
    plt.plot(fpr, fnr)
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.title("DET Curve")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help=".parquet cu features")
    ap.add_argument("--model", required=True, help=".joblib")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--test", action="store_true", help="mod test (fără etichete)")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    pipe = load(args.model)

    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values
    scores = pipe.predict_proba(X)[:, 1] if hasattr(pipe[-1], "predict_proba") else pipe.decision_function(X)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if args.test:
        out_csv = Path(args.outdir) / "predictions_test.csv"
        pd.DataFrame({"rel_path": df["rel_path"], "score_spoof": scores}).to_csv(out_csv, index=False)
        print(f"Scris {out_csv}")
        return

    # dev cu etichete
    y = df["label"].values
    auc = roc_auc_score(y, scores)
    eer = eer_from_scores(y, scores)

    metrics = {"roc_auc": float(auc), "eer": float(eer), "n": int(len(y))}
    with open(Path(args.outdir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics:", metrics)

    # salvăm scoruri și predicții binare cu prag optim (Equal Error Rate)
    # prag EER ~ punctul în care FPR ≈ FNR
    # găsim pragul corespunzător indexului ales în funcția EER (recalculăm rapid)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    thr_eer = float(thr[idx])
    pred = (scores >= thr_eer).astype(int)

    out_pred = Path(args.outdir) / "predictions_dev.csv"
    pd.DataFrame({
        "rel_path": df["rel_path"],
        "label": y,
        "score_spoof": scores,
        "pred_spoof": pred
    }).to_csv(out_pred, index=False)

    save_plot_roc(y, scores, Path(args.outdir) / "roc_curve.png")
    save_plot_det(y, scores, Path(args.outdir) / "det_curve.png")

    print(f"Salvat: metrics.json, predictions_dev.csv, roc_curve.png, det_curve.png în {args.outdir}")

if __name__ == "__main__":
    main()
