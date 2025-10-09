#!/usr/bin/env python3
"""
Train all available combinations by delegating to train_cnn1d.py.
- Discovers combos (NPZ stems under index/combos/train, or lines in temp_data/combinations_ordered_by_accuracy.txt).
- Calls train_cnn1d.py for each combo (so that data loading, testing and metrics saving are centralized).
- Aggregates results (combo, accuracy, results_dir, error) into a CSV.

Usage examples:
  python train_all_combos.py --data-root /path/to/asvspoof2019
  python train_all_combos.py --epochs 50 --batch-size 128
  python train_all_combos.py --combos-file temp_data/combinations_ordered_by_accuracy.txt
"""

import argparse
import sys
import subprocess
import re
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Locate repo root & import constants ----------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE  # this script is expected at repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from constants import (
    # dataset & folders
    directory as DEFAULT_DATA_ROOT,
    index_folder_name as INDEX_DIRNAME,
    results_folder as RESULTS_ROOT,
    temp_data_folder_name as TEMP_DIRNAME,
    # defaults for model/training
    sampling_rate as DEFAULT_SR,
    cnn1d_n_mfcc as DEFAULT_N_MFCC,
    cnn1d_duration_seconds as DEFAULT_DURATION,
    cnn1d_batch_size as DEFAULT_BATCH_SIZE,
    cnn1d_epochs as DEFAULT_EPOCHS,
    random_state as DEFAULT_SEED,
    # filenames
    accuracy_txt_filename as ACC_TXT_NAME,
    final_model_filename as FINAL_MODEL_NAME,
    best_model_filename as BEST_MODEL_NAME,
)

# ---------- Combo discovery ----------
def read_combos_from_npz(index_dir: Path) -> list[str]:
    """Discover combos from NPZ files: <index>/combos/train/*.npz (use file stem as combo code)."""
    combos_dir = index_dir / "combos" / "train"
    if not combos_dir.exists():
        return []
    return sorted(p.stem for p in combos_dir.glob("*.npz"))

def sanitize_combo_name(name: str) -> str:
    """Turn display strings like 'A+B+C' into a safe directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

def read_combos_from_txt(txt_path: Path) -> list[str]:
    """
    Read combos from a text file (first token per line is treated as the combo id).
    Accepts forms like:
      A
      A+B
      A+B+C  accuracy=0.92
    """
    if not txt_path.exists():
        return []
    combos = []
    for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        token = ln.split()[0]
        combos.append(token)
    # Keep original tokens for display; sanitize only when we pass as folder name
    return sorted(set(combos))

def find_available_combos(data_root: Path) -> list[str]:
    """Try NPZ stems first, then the temp text file. Fallback to a single default combo."""
    index_dir = data_root / INDEX_DIRNAME
    combos = read_combos_from_npz(index_dir)
    if combos:
        return combos
    # try temp txt
    txt = REPO_ROOT / TEMP_DIRNAME / "combinations_ordered_by_accuracy.txt"
    combos = read_combos_from_txt(txt)
    if combos:
        return combos
    # final fallback: just one reasonable default
    from constants import cnn1d_default_combo_name as DEFAULT_COMBO
    return [DEFAULT_COMBO]

# ---------- Train one combo by calling train_cnn1d.py ----------
def run_single_combo(
    combo_display: str,
    data_root: Path,
    index_dir: Path | None,
    sr: int,
    n_mfcc: int,
    duration: float,
    batch_size: int,
    epochs: int,
    seed: int,
) -> tuple[float | None, Path, str | None]:
    """
    Calls train_cnn1d.py with the provided args.
    Returns: (accuracy or None, results_dir, error or None)
    """
    safe_name = sanitize_combo_name(combo_display)
    train_script = REPO_ROOT / "scripts" / "train_cnn1d.py"

    cmd = [
        sys.executable, str(train_script),
        "--data-root", str(data_root),
        "--combo-name", safe_name,
        "--sr", str(sr),
        "--n-mfcc", str(n_mfcc),
        "--duration", str(duration),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    if index_dir is not None:
        cmd += ["--index-dir", str(index_dir)]

    try:
        # Stream output to console so user sees progress from the child script
        ret = subprocess.run(cmd, check=False)
        # Parse accuracy from results/<combo>/accuracy.txt
        results_dir = REPO_ROOT / RESULTS_ROOT / safe_name
        acc_file = results_dir / ACC_TXT_NAME
        acc = None
        if acc_file.exists():
            m = re.search(r"Test\s+accuracy:\s*([0-9.]+)", acc_file.read_text(encoding="utf-8", errors="ignore"))
            if m:
                acc = float(m.group(1))
        # If the run failed, leave acc as None and capture a lightweight message
        err = None if ret.returncode == 0 else f"train_cnn1d returned {ret.returncode}"
        return acc, results_dir, err
    except Exception as e:
        return None, REPO_ROOT / RESULTS_ROOT / safe_name, str(e)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Train 1D-CNN over all discovered combinations (sequential).")
    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                    help="Dataset root (defaults to constants.directory)")
    ap.add_argument("--index-dir", type=str, default=None,
                    help=f"Split CSV folder (default: <data-root>/{INDEX_DIRNAME})")
    ap.add_argument("--combos-file", type=str, default=None,
                    help="Optional file with combos (one per line; first token used). Overrides discovery.")
    ap.add_argument("--out-csv", type=str, default=str(REPO_ROOT / RESULTS_ROOT / "combos_accuracy.csv"),
                    help="Where to write the summary CSV.")
    # training defaults (mirroring train_cnn1d)
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--n-mfcc", type=int, default=DEFAULT_N_MFCC)
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    index_dir = Path(args.index_dir).resolve() if args.index_dir else (data_root / INDEX_DIRNAME)
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) discover combos
    if args.combos_file:
        combos = read_combos_from_txt(Path(args.combos_file))
        if not combos:
            print(f"[!] No combos found in {args.combos_file}. Falling back to discovery.", flush=True)
            combos = find_available_combos(data_root)
    else:
        combos = find_available_combos(data_root)

    # Show list & count
    print("\n=== Discovered combinations ===")
    for c in combos:
        print(" -", c)
    print(f"Total: {len(combos)}\n")

    # 2) iterate & train
    rows = []
    for i, combo in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] Training combo: {combo}")
        acc, results_dir, err = run_single_combo(
            combo_display=combo,
            data_root=data_root,
            index_dir=index_dir if index_dir.exists() else None,
            sr=args.sr,
            n_mfcc=args.n_mfcc,
            duration=args.duration,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
        )
        row = {
            "combo": combo,
            "results_dir": str(results_dir),
            "accuracy": (np.nan if acc is None else acc),
            "best_model": str(results_dir / BEST_MODEL_NAME),
            "final_model": str(results_dir / FINAL_MODEL_NAME),
            "error": ("" if err is None else err),
        }
        rows.append(row)
        # Write incrementally so we persist progress
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        status = "OK" if err is None else f"ERR: {err}"
        print(f" -> {status}; accuracy={acc if acc is not None else 'NA'}; saved to {out_csv}\n")

    print(f"[✓] Done. Summary at: {out_csv}")

if __name__ == "__main__":
    main()
