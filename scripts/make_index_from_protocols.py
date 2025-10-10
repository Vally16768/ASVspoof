#!/usr/bin/env python3
"""
Build ASVspoof 2019 LA indices using ONLY constants.py.

Outputs (under <constants.directory>/<constants.index_folder_name>):
  - train.csv : labeled   (train minus a stratified held-out test)
  - val.csv   : labeled   (dev -> validation)
  - test.csv  : labeled   (held-out, stratified from train by label; size = constants.test_size)
  - eval.list : unlabeled (paths only; blind evaluation set)
"""

from __future__ import annotations
import csv
import os
import sys
import random
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set

# ------------------------------------------------------------------------------
# Config from constants.py (the script itself does NOT read env vars directly)
# ------------------------------------------------------------------------------
sys.path.insert(0, os.getcwd())
try:
    import importlib
    constants = importlib.import_module("constants")
except Exception as e:
    raise SystemExit(f"[!] Could not import constants.py from CWD: {e}")

# Required
DATA_ROOT = Path(getattr(constants, "directory", "dataset")).resolve()
INDEX_DIRNAME = getattr(constants, "index_folder_name", "index")
TEST_SIZE = float(getattr(constants, "test_size", 0.20))
SEED = int(getattr(constants, "random_state", 1337))

# Optional (add these in constants.py if you want custom names)
LA_TRAIN_FLAC_SUBDIR = getattr(constants, "la_train_flac_subdir", "ASVspoof2019_LA_train/flac")
LA_DEV_FLAC_SUBDIR   = getattr(constants, "la_dev_flac_subdir",   "ASVspoof2019_LA_dev/flac")
LA_EVAL_FLAC_SUBDIR  = getattr(constants, "la_eval_flac_subdir",  "ASVspoof2019_LA_eval/flac")
LA_PROTOCOLS_SUBDIR  = getattr(constants, "la_protocols_subdir",  "ASVspoof2019_LA_cm_protocols")

LA_TRAIN_TRN_FILENAME = getattr(
    constants, "la_train_trn_filename", "ASVspoof2019.LA.cm.train.trn.txt"
)
LA_DEV_TRL_FILENAME = getattr(
    constants, "la_dev_trl_filename", "ASVspoof2019.LA.cm.dev.trl.txt"
)
LA_EVAL_TRL_FILENAME = getattr(
    constants, "la_eval_trl_filename", "ASVspoof2019.LA.cm.eval.trl.txt"
)

# ------------------------------------------------------------------------------
# Helpers: protocol parsing
# ------------------------------------------------------------------------------
def parse_trl_line(line: str) -> Tuple[str, str] | None:
    """
    Standard LA cm .trn/.trl format:
      <spk> <file_id> <sys> <attk> <key>
    Keep (file_id, key) where key in {bonafide, spoof}.
    """
    parts = line.strip().split()
    if not parts or parts[0].startswith("#"):
        return None
    fid = parts[1] if len(parts) > 1 else None
    key = parts[-1].lower() if parts else None
    if fid and key in {"bonafide", "spoof"}:
        return fid, key
    return None


def read_trl_labeled(p: Path) -> List[Tuple[str, str]]:
    if not p.exists():
        raise FileNotFoundError(p)
    items: List[Tuple[str, str]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            pr = parse_trl_line(ln)
            if pr:
                items.append(pr)
    return items


def read_any_fids(p: Path) -> List[str]:
    """
    Relaxed parser to extract file IDs even if labels are absent
    (useful for eval lists that sometimes ship without keys).
    """
    if not p.exists():
        return []
    fids: List[str] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.strip().split()
            if parts and not parts[0].startswith("#") and len(parts) > 1:
                fids.append(parts[1])
    return fids


# ------------------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
# Core
# ------------------------------------------------------------------------------
def map_rows(root: Path, base_rel: str, pairs: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    base = root / base_rel
    rows: List[Tuple[str, str]] = []
    for fid, key in pairs:
        rel = (base / f"{fid}.wav").relative_to(root)
        rows.append((str(rel), key))
    return rows


def stratified_split(
    rows: List[Tuple[str, str]],
    test_size: float,
    seed: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Stratified split by the 'label' (bonafide/spoof). Returns (train_rows, test_rows).
    """
    by_label: Dict[str, List[int]] = {}
    for i, (_, label) in enumerate(rows):
        by_label.setdefault(label, []).append(i)

    rng = random.Random(seed)
    test_indices: Set[int] = set()

    for label, idxs in by_label.items():
        if not idxs:
            continue
        n_total = len(idxs)
        if n_total == 1:
            chosen = {idxs[0]}  # singleton -> test
        else:
            n_test = max(1, int(round(test_size * n_total)))
            n_test = min(n_test, n_total - 1)  # keep at least 1 for train
            chosen = set(rng.sample(idxs, n_test))
        test_indices |= chosen

    test_rows = [rows[i] for i in sorted(test_indices)]
    train_rows = [rows[i] for i in range(len(rows)) if i not in test_indices]
    return train_rows, test_rows


def main() -> None:
    # Validate constants
    if not (0.0 < TEST_SIZE < 1.0):
        raise SystemExit(f"[!] Invalid constants.test_size={TEST_SIZE}. Expected (0,1).")

    # Check dataset structure
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
            "      2) Keep your constants.directory but adjust the *_subdir constants, e.g.:",
            "           la_train_flac_subdir = 'ASVspoof2019_LA_train/flac'",
            "           la_dev_flac_subdir   = 'ASVspoof2019_LA_dev/flac'",
            "           la_eval_flac_subdir  = 'ASVspoof2019_LA_eval/flac'",
            "           la_protocols_subdir  = 'ASVspoof2019_LA_cm_protocols'",
        ]
        raise SystemExit("\n".join(msg))

    protos = DATA_ROOT / LA_PROTOCOLS_SUBDIR
    train_trn = protos / LA_TRAIN_TRN_FILENAME
    dev_trl   = protos / LA_DEV_TRL_FILENAME
    eval_trl  = protos / LA_EVAL_TRL_FILENAME

    for p in [train_trn, dev_trl]:
        if not p.exists():
            raise SystemExit(f"[!] Missing protocol file: {p}")

    # Labeled rows (train/dev)
    rows_train_full = map_rows(DATA_ROOT, LA_TRAIN_FLAC_SUBDIR, read_trl_labeled(train_trn))
    rows_val        = map_rows(DATA_ROOT, LA_DEV_FLAC_SUBDIR,   read_trl_labeled(dev_trl))

    # Stratified internal test from (original) train
    rows_train, rows_test = stratified_split(rows_train_full, test_size=TEST_SIZE, seed=SEED)

    # Eval list (unlabeled)
    eval_paths: List[str] = []
    base_eval = DATA_ROOT / LA_EVAL_FLAC_SUBDIR
    if eval_trl.exists():
        fids = read_any_fids(eval_trl)
        eval_paths = [str((base_eval / f"{fid}.wav").relative_to(DATA_ROOT)) for fid in fids]
    else:
        if not base_eval.exists():
            raise SystemExit(f"[!] Eval FLAC dir not found: {base_eval}")
        eval_paths = [str(p.relative_to(DATA_ROOT)) for p in sorted(base_eval.glob("*.wav"))]

    # Write outputs under <data-root>/<INDEX_DIRNAME>
    idx = DATA_ROOT / INDEX_DIRNAME
    write_csv(idx / "train.csv", rows_train)
    write_csv(idx / "val.csv",   rows_val)
    write_csv(idx / "test.csv",  rows_test)
    write_list(idx / "eval.list", eval_paths)

    # Summary
    def count_by_label(rows: List[Tuple[str, str]]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for _, lab in rows:
            c[lab] = c.get(lab, 0) + 1
        return c

    print("[✓] Wrote indices to:", idx)
    print(f"    train.csv : {len(rows_train)}  by label {count_by_label(rows_train)}")
    print(f"    val.csv   : {len(rows_val)}    by label {count_by_label(rows_val)}")
    print(f"    test.csv  : {len(rows_test)}   by label {count_by_label(rows_test)}")
    print(f"    eval.list : {len(eval_paths)}  (paths only)")
    print(f"    test_size : {TEST_SIZE}  seed={SEED}")


if __name__ == "__main__":
    main()
