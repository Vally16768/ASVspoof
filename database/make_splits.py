#!/usr/bin/env python3
import argparse, random, shutil
from pathlib import Path

def parse_protocol_line(line: str):
    """
    Formatele CM au de obicei 5 coloane:
    speaker_id, file_id, path/sys, attack_id_or_sys, key(bonafide/spoof)
    -> ne întoarcem (file_id, key) din coloana 2 și ultima.
    """
    parts = line.strip().split()
    if not parts or parts[0].startswith('#'):
        return None
    if len(parts) < 2:
        return None
    file_id = parts[1]                # <-- AICI era problema: trebuie coloana 2
    label   = parts[-1].lower()
    return file_id, label

def read_list(protocol_path: Path):
    items = []
    with protocol_path.open('r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            parsed = parse_protocol_line(ln)
            if parsed:
                items.append(parsed)
    return items

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, do_copy: bool):
    if dst.exists():
        return
    if do_copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)  # fallback pe Windows

def find_audio(base_dir: Path, file_id: str):
    """Caută rapid fișierul (.flac implicit) fără a depinde strict de extensie."""
    cand = base_dir / f"{file_id}.flac"
    if cand.exists():
        return cand
    # fallback: caută prin glob dacă vreodată apare altă extensie
    hits = list(base_dir.glob(f"{file_id}.*"))
    return hits[0] if hits else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="database/data/asvspoof2019",
                    help="Folderul care conține ASVspoof2019_LA_* și ASVspoof2019_LA_cm_protocols")
    ap.add_argument("--out", default="database/data/asvspoof2019/split",
                    help="Unde să creeze split-urile (train/val/test)")
    ap.add_argument("--val-size", type=float, default=0.5,
                    help="Proporția din DEV care devine VAL (restul devine TEST)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copiază fișierele în loc de symlink")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    proto_dir = root / "ASVspoof2019_LA_cm_protocols"
    train_flac = root / "ASVspoof2019_LA_train" / "flac"
    dev_flac   = root / "ASVspoof2019_LA_dev" / "flac"

    train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto   = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"

    if not all(p.exists() for p in [train_proto, dev_proto, train_flac, dev_flac]):
        raise SystemExit("Nu găsesc protocoalele/FLAC-urile. Ai rulat download_asvspoof_la.sh și e calea corectă?")

    train_items = read_list(train_proto)
    dev_items   = read_list(dev_proto)

    # separă dev pe etichete și împarte în val/test
    dev_bona = [fid for (fid, lab) in dev_items if lab == "bonafide"]
    dev_spoof= [fid for (fid, lab) in dev_items if lab == "spoof"]

    rnd = random.Random(args.seed)
    rnd.shuffle(dev_bona)
    rnd.shuffle(dev_spoof)

    n_val_b = int(len(dev_bona) * args.val_size)
    n_val_s = int(len(dev_spoof) * args.val_size)

    val_ids   = set(dev_bona[:n_val_b] + dev_spoof[:n_val_s])
    test_ids  = set(dev_bona[n_val_b:] + dev_spoof[n_val_s:])

    # directoare
    for split in ["train","val","test"]:
        for lab in ["bonafide","spoof"]:
            ensure_dir(out_root / split / lab)

    # umple TRAIN din train/
    miss_train = 0
    for fid, lab in train_items:
        src = find_audio(train_flac, fid)
        if not src:
            miss_train += 1
            continue
        dst = out_root / "train" / lab / f"{src.stem}{src.suffix}"
        link_or_copy(src, dst, args.copy)

    # umple VAL / TEST din dev/
    miss_dev = 0
    for fid, lab in dev_items:
        split = "val" if fid in val_ids else ("test" if fid in test_ids else None)
        if not split:
            continue
        src = find_audio(dev_flac, fid)
        if not src:
            miss_dev += 1
            continue
        dst = out_root / split / lab / f"{src.stem}{src.suffix}"
        link_or_copy(src, dst, args.copy)

    # raportează câte fișiere sunt în fiecare + câte lipsesc
    def count_files(p): return sum(1 for _ in p.rglob("*.flac"))
    print("[*] Done.")
    print(f"[i] Missing (train/dev) matches: {miss_train} / {miss_dev}")
    print("Counts:")
    for split in ["train","val","test"]:
        for lab in ["bonafide","spoof"]:
            p = out_root / split / lab
            print(f"  {split}/{lab}: {count_files(p)}")
