#make_index_from_protocols.py
#!/usr/bin/env python3
import csv, argparse
from pathlib import Path

def parse_trl_line(line: str):
    # Format uzual: <spk> <file_id> <sys> <attk> <key>
    parts = line.strip().split()
    if not parts or parts[0].startswith("#"): return None
    fid = parts[1] if len(parts) > 1 else None
    key = parts[-1].lower() if parts else None
    if fid and key in {"bonafide","spoof"}:
        return fid, key
    return None

def read_trl(p: Path):
    items = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            pr = parse_trl_line(ln)
            if pr: items.append(pr)
    return items

def write_csv(out_csv: Path, rows):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path","label"])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Generează index/{train,val,test}.csv din protocoale LA")
    ap.add_argument("--data-root", required=True, help="Rădăcina datasetului (ex: data/asvspoof2019)")
    ap.add_argument("--protocols-root", default=None,
                    help="Folderul cu ASVspoof2019_LA_cm_protocols; dacă lipsește, se caută în mai multe locuri")
    args = ap.parse_args()
    root = Path(args.data_root)

    # Unde sunt protocoalele
    candidates = [
        Path(args.protocols_root) if args.protocols_root else None,
        root/"ASVspoof2019_LA_cm_protocols",
        root.parent/"asvspoof2019_labelled/ASVspoof2019_LA_cm_protocols",
    ]
    protos = next((c for c in candidates if c and c.exists()), None)
    if not protos:
        raise SystemExit("[!] Nu am găsit folderul ASVspoof2019_LA_cm_protocols")

    train_trn = protos/"ASVspoof2019.LA.cm.train.trn.txt"
    dev_trl   = protos/"ASVspoof2019.LA.cm.dev.trl.txt"
    eval_trl  = protos/"ASVspoof2019.LA.cm.eval.trl.txt"
    for p in [train_trn, dev_trl, eval_trl]:
        if not p.exists():
            raise SystemExit(f"[!] Lipsește: {p}")

    # Construim căi spre .flac
    def map_rows(pairs, split):
        if split=="train":
            base = root/"ASVspoof2019_LA_train/flac"
        elif split=="dev":
            base = root/"ASVspoof2019_LA_dev/flac"
        else:
            base = root/"ASVspoof2019_LA_eval/flac"
        rows=[]
        for fid, key in pairs:
            rows.append([str((base/f"{fid}.flac").relative_to(root)), key])
        return rows

    rows_tr  = map_rows(read_trl(train_trn), "train")
    rows_dev = map_rows(read_trl(dev_trl),   "dev")
    rows_te  = map_rows(read_trl(eval_trl),  "eval")

    idx = root/"index"
    write_csv(idx/"train.csv", rows_tr)
    write_csv(idx/"val.csv",   rows_dev)   # dev -> val
    write_csv(idx/"test.csv",  rows_te)    # eval -> test

    print(f"[✓] Scris: {idx/'train.csv'}, {idx/'val.csv'}, {idx/'test.csv'}")

if __name__ == "__main__":
    main()
