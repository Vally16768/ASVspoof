#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path

def parse_protocol_line(line: str):
    # speaker_id  file_id  sys/path  attack/sys  key(bonafide/spoof)
    parts = line.strip().split()
    if not parts or parts[0].startswith("#"): return None
    if len(parts) < 2: return None
    file_id = parts[1]
    label = parts[-1].lower() if parts[-1].lower() in {"bonafide","spoof"} else ""
    return file_id, label

def read_list(p: Path):
    items=[]
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            pr = parse_protocol_line(ln)
            if pr: items.append(pr)
    return items

def fid_to_path(fid: str, train_flac: Path, dev_flac: Path, eval_flac: Path):
    # mapează după prefix (LA_T_, LA_D_, LA_E_)
    if fid.startswith("LA_T_"): base = train_flac
    elif fid.startswith("LA_D_"): base = dev_flac
    elif fid.startswith("LA_E_"): base = eval_flac
    else: base = train_flac
    p = base / f"{fid}.flac"
    if p.exists(): return p
    # fallback
    for b in (train_flac, dev_flac, eval_flac):
        q = b / f"{fid}.flac"
        if q.exists(): return q
    return None

def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        for r in rows: w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="database/data/asvspoof2019")
    ap.add_argument("--out",  default="database/data/asvspoof2019/index")
    ap.add_argument("--path_mode", choices=["rel","abs"], default="rel",
                    help="cale în CSV: relativă la --root sau absolută")
    ap.add_argument("--with-eval", action="store_true",
                    help="scrie și eval.list (fără etichete)")
    args = ap.parse_args()

    root = Path(args.root)
    proto = root / "ASVspoof2019_LA_cm_protocols"
    train_flac = root / "ASVspoof2019_LA_train" / "flac"
    dev_flac   = root / "ASVspoof2019_LA_dev"   / "flac"
    eval_flac  = root / "ASVspoof2019_LA_eval"  / "flac"

    trn_proto = proto / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto = proto / "ASVspoof2019.LA.cm.dev.trl.txt"
    for p in [trn_proto, dev_proto, train_flac, dev_flac]:
        if not p.exists(): sys.exit(f"[!] Lipsește: {p}")

    train_items = read_list(trn_proto)   # (fid,label)
    dev_items   = read_list(dev_proto)

    def rows_from(items):
        rows=[]
        for fid, lab in items:
            p = fid_to_path(fid, train_flac, dev_flac, eval_flac)
            if not p: continue
            path = str(p if args.path_mode=="abs" else p.relative_to(root))
            rows.append([path, lab])
        return rows

    train_rows = rows_from(train_items)
    val_rows   = rows_from(dev_items)

    out_dir = Path(args.out)
    write_csv(train_rows, out_dir / "train.csv")
    write_csv(val_rows,   out_dir / "val.csv")

    print(f"[*] train.csv: {len(train_rows)}  | val.csv: {len(val_rows)}")
    # opțional: listă de fișiere pentru eval (fără etichete)
    if args.with_eval and eval_flac.exists():
        eval_list = sorted(str((p if args.path_mode=="abs" else p.relative_to(root)))
                           for p in eval_flac.glob("*.flac"))
        (out_dir / "eval.list").write_text("\n".join(eval_list))
        print(f"[*] eval.list: {len(eval_list)} (fără etichete)")

if __name__ == "__main__":
    main()
