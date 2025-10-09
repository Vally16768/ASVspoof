#!/usr/bin/env python3
"""
Build ASVspoof 2019 LA indices.

- train.csv : labeled   (train minus a stratified held-out test)
- val.csv   : labeled   (dev -> validation)
- test.csv  : labeled   (held-out, stratified from train by label; size = constants.test_size)
- eval.list : unlabeled (paths only; blind evaluation set)

The script prefers the official LA cm protocol files to enumerate files,
but will gracefully fall back to scanning the eval/flac directory for eval.list.

Requires:
  constants.py at project root with: test_size = 0.20
"""

from __future__ import annotations
import argparse
import csv
import os
import random
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set

# ---- Config from constants.py (with robust fallback) -------------------------
try:
    from constants import test_size as TEST_SIZE  # noqa: N812
except Exception:
    # Allows override via env TEST_SIZE, otherwise defaults to 0.20
    TEST_SIZE = float(os.getenv("TEST_SIZE", "0.20"))

DEFAULT_SEED = 1337


# ---- Helpers: protocol parsing ------------------------------------------------
def parse_trl_line(line: str) -> Tuple[str, str] | None:
    """
    Standard LA cm .trn/.trl format:
      <spk> <file_id> <sys> <attk> <key>
    We only need (file_id, key) where key in {bonafide, spoof}.
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
    fids: List[str] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            parts = ln.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) > 1:
                fids.append(parts[1])
    return fids


# ---- IO helpers ---------------------------------------------------------------
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


# ---- Core logic ---------------------------------------------------------------
def map_rows(root: Path, pairs: Iterable[Tuple[str, str]], split: str) -> List[Tuple[str, str]]:
    if split == "train":
        base = root / "ASVspoof2019_LA_train" / "flac"
    elif split == "dev":
        base = root / "ASVspoof2019_LA_dev" / "flac"
    elif split == "eval":
        base = root / "ASVspoof2019_LA_eval" / "flac"
    else:
        raise ValueError(f"Unknown split: {split}")

    rows: List[Tuple[str, str]] = []
    for fid, key in pairs:
        rel = (base / f"{fid}.flac").relative_to(root)
        rows.append((str(rel), key))
    return rows


def stratified_split(
    rows: List[Tuple[str, str]],
    test_size: float,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Stratified split by the 'label' (2 classes: bonafide/spoof).
    Returns (train_rows, test_rows).
    """
    by_label: Dict[str, List[int]] = {"bonafide": [], "spoof": []}
    for i, (_, label) in enumerate(rows):
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(i)

    rng = random.Random(seed)
    test_indices: Set[int] = set()

    for label, idxs in by_label.items():
        if not idxs:
            continue
        n_total = len(idxs)
        n_test = max(1, int(round(test_size * n_total))) if n_total > 1 else 1
        n_test = min(n_test, n_total - 1) if n_total > 1 else n_test  # keep at least 1 for train
        chosen = set(rng.sample(idxs, n_test))
        test_indices |= chosen

    test_rows = [rows[i] for i in sorted(test_indices)]
    train_rows = [rows[i] for i in range(len(rows)) if i not in test_indices]
    return train_rows, test_rows


def main():
    ap = argparse.ArgumentParser(description="Generate index/{train,val,test}.csv and index/eval.list for ASVspoof 2019 LA")
    ap.add_argument("--data-root", required=True, help="Dataset root, e.g. data/asvspoof2019 or database/data/asvspoof2019")
    ap.add_argument("--protocols-root", default=None,
                    help="Folder with ASVspoof2019_LA_cm_protocols (optional; auto-detected otherwise)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for the stratified split (default: {DEFAULT_SEED})")
    ap.add_argument("--test-size", type=float, default=None,
                    help="Override constants.test_size at runtime (e.g., 0.2).")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    if not root.exists():
        raise SystemExit(f"[!] Data root not found: {root}")

    # Resolve protocol folder
    candidates = [
        Path(args.protocols_root) if args.protocols_root else None,
        root / "ASVspoof2019_LA_cm_protocols",
        root.parent / "asvspoof2019_labelled" / "ASVspoof2019_LA_cm_protocols",
    ]
    protos = next((c for c in candidates if c and c.exists()), None)
    if not protos:
        raise SystemExit("[!] Could not find ASVspoof2019_LA_cm_protocols")

    train_trn = protos / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_trl   = protos / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_trl  = protos / "ASVspoof2019.LA.cm.eval.trl.txt"

    for p in [train_trn, dev_trl]:
        if not p.exists():
            raise SystemExit(f"[!] Missing: {p}")

    # Labeled rows (train/dev)
    rows_train_full = map_rows(root, read_trl_labeled(train_trn), "train")
    rows_val        = map_rows(root, read_trl_labeled(dev_trl), "dev")

    # Test size selection (constants.py → CLI override → env/default)
    test_size = args.test_size if args.test_size is not None else TEST_SIZE
    if not (0.0 < test_size < 1.0):
        raise SystemExit(f"[!] Invalid test_size={test_size}. Expected (0,1).")

    rows_train, rows_test = stratified_split(rows_train_full, test_size=test_size, seed=args.seed)

    # Eval list (unlabeled)
    eval_paths: List[str] = []
    if eval_trl.exists():
        # Prefer protocol list if available
        fids = read_any_fids(eval_trl)
        base = root / "ASVspoof2019_LA_eval" / "flac"
        eval_paths = [str((base / f"{fid}.flac").relative_to(root)) for fid in fids]
    else:
        # Fallback: scan directory
        base = root / "ASVspoof2019_LA_eval" / "flac"
        if not base.exists():
            raise SystemExit(f"[!] Eval FLAC dir not found: {base}")
        eval_paths = [str(p.relative_to(root)) for p in sorted(base.glob("*.flac"))]

    # Write outputs
    idx = root / "index"
    write_csv(idx / "train.csv", rows_train)
    write_csv(idx / "val.csv",   rows_val)
    write_csv(idx / "test.csv",  rows_test)
    write_list(idx / "eval.list", eval_paths)

    # Pretty summary
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
    print(f"    test_size : {test_size}  seed={args.seed}")


if __name__ == "__main__":
    main()
