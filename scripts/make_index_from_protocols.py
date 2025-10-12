#!/usr/bin/env python3
"""
Build ASVspoof 2019 LA indices (Option A):
  - train.csv : labeled   (LA_train minus a speaker-grouped validation cut)
  - val.csv   : labeled   (10% from LA_train, stratified by label, grouped by speaker)
  - test.csv  : labeled   (LA_dev — official dev set)
  - eval.list : unlabeled (LA_eval paths)

All paths/names come from constants.py.
"""

from __future__ import annotations
import csv
import os
import sys
import random
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set

# ---------- constants.py ----------
sys.path.insert(0, os.getcwd())
try:
    import importlib
    C = importlib.import_module("constants")
except Exception as e:
    raise SystemExit(f"[!] Could not import constants.py from CWD: {e}")

DATA_ROOT  = Path(getattr(C, "directory", "dataset")).resolve()
INDEX_NAME = getattr(C, "index_folder_name", "index")
VAL_SIZE   = float(getattr(C, "validation_size", 0.10))
SEED       = int(getattr(C, "random_state", 42))

LA_TRAIN_FLAC_SUBDIR = getattr(C, "la_train_flac_subdir", "ASVspoof2019_LA_train/flac")
LA_DEV_FLAC_SUBDIR   = getattr(C, "la_dev_flac_subdir",   "ASVspoof2019_LA_dev/flac")
LA_EVAL_FLAC_SUBDIR  = getattr(C, "la_eval_flac_subdir",  "ASVspoof2019_LA_eval/flac")
LA_PROTOCOLS_SUBDIR  = getattr(C, "la_protocols_subdir",  "ASVspoof2019_LA_cm_protocols")

LA_TRAIN_TRN_FILENAME = getattr(C, "la_train_trn_filename", "ASVspoof2019.LA.cm.train.trn.txt")
LA_DEV_TRL_FILENAME   = getattr(C, "la_dev_trl_filename",   "ASVspoof2019.LA.cm.dev.trl.txt")
LA_EVAL_TRL_FILENAME  = getattr(C, "la_eval_trl_filename",  "ASVspoof2019.LA.cm.eval.trl.txt")

# ---------- Parsere protocol ----------
def parse_trl_line_keep_spk(line: str) -> Tuple[str, str, str] | None:
    """
    Format LA: <spk> <file_id> <sys> <attk> <key>
    Returnăm (spk, file_id, key) cu key în {bonafide, spoof}.
    """
    parts = line.strip().split()
    if not parts or parts[0].startswith("#"):
        return None
    spk = parts[0] if len(parts) > 0 else None
    fid = parts[1] if len(parts) > 1 else None
    key = parts[-1].lower() if parts else None
    if spk and fid and key in {"bonafide", "spoof"}:
        return spk, fid, key
    return None

def read_trl_labeled_triples(p: Path) -> List[Tuple[str, str, str]]:
    if not p.exists():
        raise FileNotFoundError(p)
    items: List[Tuple[str, str, str]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            pr = parse_trl_line_keep_spk(ln)
            if pr:
                items.append(pr)
    return items

def read_any_fids(p: Path) -> List[str]:
    """Extrage file_id chiar dacă lipsesc etichetele (util pentru eval)."""
    if not p.exists():
        return []
    fids: List[str] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.strip().split()
            if parts and not parts[0].startswith("#") and len(parts) > 1:
                fids.append(parts[1])
    return fids

# ---------- IO helpers ----------
def write_csv(out_csv: Path, rows: Iterable[Tuple[str, str]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        for r in rows:
            w.writerow(r)

def write_list(out_txt: Path, paths: Iterable[str]) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

def as_rows_pairs(root: Path, base_rel: str, triples: Iterable[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
    """Mapează (spk,fid,label) -> (relpath,label)."""
    base = root / base_rel
    out: List[Tuple[str, str]] = []
    for _, fid, key in triples:
        rel = (base / f"{fid}.wav").relative_to(root)
        out.append((str(rel), key))
    return out

# ---------- Split: stratificat pe LABEL, grupat pe SPEAKER ----------
def stratified_group_split_by_speaker(
    triples: List[Tuple[str, str, str]],  # (spk, fid, label)
    val_size: float,
    seed: int,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Împarte LA_train în (train, val) alegând SPEAKERI pentru validare,
    separat pe fiecare label (bonafide/spoof).
    """
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0,1)")

    # grupăm speakerii pe label
    speakers_by_label: Dict[str, Set[str]] = {}
    for spk, _, lab in triples:
        speakers_by_label.setdefault(lab, set()).add(spk)

    rng = random.Random(seed)
    val_speakers: Set[str] = set()
    for lab, spk_set in speakers_by_label.items():
        spks = sorted(spk_set)
        n_take = max(1, round(val_size * len(spks)))
        n_take = min(n_take, len(spks) - 1) if len(spks) > 1 else 1
        chosen = set(rng.sample(spks, n_take))
        val_speakers |= chosen

    train_triples, val_triples = [], []
    for spk, fid, lab in triples:
        (val_triples if spk in val_speakers else train_triples).append((spk, fid, lab))
    return train_triples, val_triples

# ---------- Main ----------
def main() -> None:
    # Verificări structură — la fel ca înainte
    must_exist = [
        DATA_ROOT / LA_TRAIN_FLAC_SUBDIR,
        DATA_ROOT / LA_DEV_FLAC_SUBDIR,
        DATA_ROOT / LA_PROTOCOLS_SUBDIR,
    ]
    missing = [str(p) for p in must_exist if not p.exists()]
    if missing:
        msg = [
            f"[!] Data root appears incorrect: {DATA_ROOT}",
            "    The following required paths are missing:",
            *[f"      - {m}" for m in missing],
            "",
            "    Fix one of these:",
            "      1) Set an existing path in constants.directory, OR",
            "      2) Adjust the *_subdir constants (la_*_flac_subdir / la_protocols_subdir).",
        ]
        raise SystemExit("\n".join(msg))

    protos = DATA_ROOT / LA_PROTOCOLS_SUBDIR
    train_trn = protos / LA_TRAIN_TRN_FILENAME
    dev_trl   = protos / LA_DEV_TRL_FILENAME
    eval_trl  = protos / LA_EVAL_TRL_FILENAME

    if not train_trn.exists() or not dev_trl.exists():
        raise SystemExit(f"[!] Missing protocol(s): {train_trn if not train_trn.exists() else ''} {dev_trl if not dev_trl.exists() else ''}")

    # 1) LA_train: citim (spk,fid,label)
    triples_train_full = read_trl_labeled_triples(train_trn)

    # 2) Split train/val pe SPEAKER (stratificat pe label)
    triples_train, triples_val = stratified_group_split_by_speaker(
        triples_train_full, val_size=VAL_SIZE, seed=SEED
    )

    # 3) LA_dev devine TEST (perechi path,label)
    dev_pairs = as_rows_pairs(DATA_ROOT, LA_DEV_FLAC_SUBDIR, read_trl_labeled_triples(dev_trl))

    # 4) EVAL list (doar path-uri)
    eval_paths: List[str] = []
    base_eval = DATA_ROOT / LA_EVAL_FLAC_SUBDIR
    if eval_trl.exists():
        fids = read_any_fids(eval_trl)
        eval_paths = [str((base_eval / f"{fid}.wav").relative_to(DATA_ROOT)) for fid in fids]
    else:
        if not base_eval.exists():
            raise SystemExit(f"[!] Eval FLAC dir not found: {base_eval}")
        eval_paths = [str(p.relative_to(DATA_ROOT)) for p in sorted(base_eval.glob("*.wav"))]

    # 5) Scriem index-urile
    idx = DATA_ROOT / INDEX_NAME
    write_csv(idx / "train.csv", as_rows_pairs(DATA_ROOT, LA_TRAIN_FLAC_SUBDIR, triples_train))
    write_csv(idx / "val.csv",   as_rows_pairs(DATA_ROOT, LA_TRAIN_FLAC_SUBDIR, triples_val))
    write_csv(idx / "test.csv",  dev_pairs)
    write_list(idx / "eval.list", eval_paths)

    # Summary util
    def count_lbl(rows: List[Tuple[str, str]]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for _, lab in rows:
            c[lab] = c.get(lab, 0) + 1
        return c

    tr_pairs = as_rows_pairs(DATA_ROOT, LA_TRAIN_FLAC_SUBDIR, triples_train)
    va_pairs = as_rows_pairs(DATA_ROOT, LA_TRAIN_FLAC_SUBDIR, triples_val)
    print("[✓] Wrote indices to:", idx)
    print(f"    train.csv : {len(tr_pairs)}  by label {count_lbl(tr_pairs)}")
    print(f"    val.csv   : {len(va_pairs)}  by label {count_lbl(va_pairs)}")
    print(f"    test.csv  : {len(dev_pairs)} by label {count_lbl(dev_pairs)}")
    print(f"    eval.list : {len(eval_paths)}  (paths only)")
    print(f"    val_size  : {VAL_SIZE}  seed={SEED}")

if __name__ == "__main__":
    main()
