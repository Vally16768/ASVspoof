from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(hist.get("accuracy", []), label="train")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png", dpi=140); plt.close()

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(hist.get("loss", []), label="train")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=140); plt.close()

def plot_confusion(cm: np.ndarray, classes, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45); plt.yticks(ticks, classes)
    thr = cm.max()/2 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center",
                     color="white" if cm[i, j] > thr else "black")
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Pred")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
