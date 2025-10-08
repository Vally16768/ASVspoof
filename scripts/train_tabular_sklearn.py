#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def read_parquet_or_none(p: Path):
    return pd.read_parquet(p) if p.exists() else None

def plot_confusion(cm: np.ndarray, classes, out_path: Path):
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45); plt.yticks(ticks, classes)
    thr = cm.max()/2 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,format(cm[i,j],"d"),ha="center",
                     color="white" if cm[i,j]>thr else "black")
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Pred")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Train scikit-learn MLP on tabular MFCC features")
    ap.add_argument("--combo", default="A")
    ap.add_argument("--results-name", default=None, help="ex: A_mfcc40_tabular")
    ap.add_argument("--hidden", default="256,128,64")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res_name = args.results_name or f"{args.combo}_mfcc_tabular"
    out_dir = Path("results")/res_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fdir = Path("features_tabular")/args.combo.upper()
    train = read_parquet_or_none(fdir/"train.parquet")
    val   = read_parquet_or_none(fdir/"val.parquet")
    test  = read_parquet_or_none(fdir/"test.parquet")
    if train is None:
        raise SystemExit(f"[!] Nu găsesc features în {fdir}. Rulează întâi build_features_from_index.")

    Xtr = train.drop(columns=["label","relpath"]).values
    ytr = train["label"].values

    if val is not None:
        Xva = val.drop(columns=["label","relpath"]).values
        yva = val["label"].values
    else:
        Xva = np.empty((0, Xtr.shape[1])); yva = np.empty((0,))

    if test is not None:
        Xte = test.drop(columns=["label","relpath"]).values
        yte = test["label"].values
    else:
        Xte = np.empty((0, Xtr.shape[1])); yte = np.empty((0,))

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        batch_size=args.batch_size,
        max_iter=args.max_iter,
        random_state=args.seed,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        verbose=False
    )
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)
    pipe.fit(Xtr, ytr)

    # loss curve
    loss = pipe.named_steps["mlpclassifier"].loss_curve_
    plt.figure(figsize=(6,4))
    plt.plot(loss)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (MLP)")
    plt.tight_layout(); plt.savefig(out_dir/"loss.png", dpi=140); plt.close()

    # accuracy bar (train și best val)
    train_acc = pipe.score(Xtr, ytr)
    best_val  = getattr(pipe.named_steps["mlpclassifier"], "best_validation_score_", None)
    plt.figure(figsize=(5,3))
    bars = [train_acc] + ([best_val] if best_val is not None else [])
    labels = ["train"] + (["val(best)"] if best_val is not None else [])
    plt.bar(labels, bars); plt.ylim(0,1); plt.title("Accuracy")
    plt.tight_layout(); plt.savefig(out_dir/"accuracy.png", dpi=140); plt.close()

    # test
    if len(yte):
        ypred = pipe.predict(Xte)
        acc = accuracy_score(yte, ypred)
        (out_dir/"accuracy.txt").write_text(f"Test accuracy: {acc:.4f}\nSamples: {len(yte)}\n")
        (out_dir/"classification_report.txt").write_text(classification_report(yte, ypred, target_names=["spoof","bonafide"]))
        cm = confusion_matrix(yte, ypred, labels=[0,1])
        plot_confusion(cm, ["spoof","bonafide"], out_dir/"confusion_matrix.png")
        pd.DataFrame({"relpath": test["relpath"].values, "label": yte, "pred": ypred}).to_csv(out_dir/"predictions.csv", index=False)

    print(f"[✓] Rezultate în: {out_dir}")

if __name__ == "__main__":
    main()
