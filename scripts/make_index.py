import argparse, csv, os, sys
from pathlib import Path

PROT = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
}
SUBDIR = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev":   "ASVspoof2019_LA_dev/flac",
    "eval":  "ASVspoof2019_LA_eval/flac",
}

def parse_protocols(root: Path, split: str):
    prot = root / "ASVspoof2019_LA_cm_protocols" / PROT[split]
    if not prot.exists():
        raise FileNotFoundError(prot)
    rows = []
    with prot.open() as f:
        for line in f:
            parts = line.strip().split()
            if not parts: 
                continue
            utt_id = parts[0]
            label = parts[-1].lower() if split in ("train","dev") else ""
            rows.append((utt_id, label))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",  default=os.environ.get("ASVSPOOF_ROOT", "database/data/asvspoof2019"))
    ap.add_argument("--out",   default="database/data/asvspoof2019/index")
    ap.add_argument("--splits", nargs="+", default=["train","dev"], choices=["train","dev","eval"])
    args = ap.parse_args()

    root = Path(args.root)
    out  = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for sp in args.splits:
        items = parse_protocols(root, sp)
        sub   = SUBDIR[sp]
        csv_path = out / ("train.csv" if sp=="train" else ("val.csv" if sp=="dev" else "test.csv"))
        with open(csv_path, "w", newline="") as g:
            w = csv.writer(g)
            w.writerow(["utt_id","relpath","split","label","label_id"])
            for utt_id, label in items:
                rel = f"{sub}/{utt_id}.flac"
                wav = root / rel
                if not wav.exists():
                    continue
                label_id = {"bonafide":0, "spoof":1}.get(label, "")
                w.writerow([utt_id, rel, sp, label, label_id])
        print(f"[OK] {csv_path}  ({len(items)} linii în protocol; există pe disc: {sum(1 for _ in open(csv_path)) - 1})")
    print("[DONE] Indexuri create în", out)

if __name__=="__main__":
    sys.exit(main())
