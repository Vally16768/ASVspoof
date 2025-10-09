#!/usr/bin/env python3
"""
Train all available combinations by delegating to train_cnn1d.py.

What it does
------------
1) Discovers combos:
   - NPZ stems under <data_root>/<index_folder_name>/combos/train/*.npz
   - OR lines in <repo>/<temp_data_folder_name>/<save_the_best_combination_file_name>
   - Fallback: constants.cnn1d_default_combo_name

2) For each combo it calls train_cnn1d.py (which loads data, trains, evaluates,
   and saves per-run artifacts in <results_folder>/<combo_name>/).

3) Aggregates (combo, accuracy, results_dir, best_model, final_model, error)
   into a single CSV in <results_folder>/<save_combinations_file_name or .csv>.

Notes
-----
- No training hyperparameters are taken from CLI; train_cnn1d.py reads *all*
  hyperparameters from constants.py. Here we only allow optional overrides for:
  --data-root, --index-dir, --combos-file, --out-csv.
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
    directory as CFG_DATA_ROOT,
    index_folder_name as INDEX_DIRNAME,
    results_folder as RESULTS_ROOT,
    temp_data_folder_name as TEMP_DIRNAME,
    # filenames
    accuracy_txt_filename as ACC_TXT_NAME,
    final_model_filename as FINAL_MODEL_NAME,
    best_model_filename as BEST_MODEL_NAME,
    save_the_best_combination_file_name as BEST_COMBOS_TXT,
    save_combinations_file_name as SAVE_COMBOS_NAME,
    # default combo tag
    cnn1d_default_combo_name as DEFAULT_COMBO,
)

# ---------- Utilities ----------
def sanitize_combo_name(name: str) -> str:
    """Turn display strings like 'A+B+C' into a safe directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

def _train_script_path() -> Path:
    """Find train_cnn1d.py either at repo root or under scripts/."""
    c1 = REPO_ROOT / "train_cnn1d.py"
    if c1.exists():
        return c1
    c2 = REPO_ROOT / "scripts" / "train_cnn1d.py"
    if c2.exists():
        return c2
    raise SystemExit("[!] Could not locate train_cnn1d.py at repo root or scripts/")

# ---------- Combo discovery ----------
def read_combos_from_npz(index_dir: Path) -> list[str]:
    """Discover combos from NPZ files: <index>/combos/train/*.npz (use file stem as combo code)."""
    combos_dir = index_dir / "combos" / "train"
    if not combos_dir.exists():
        return []
    return sorted(p.stem for p in combos_dir.glob("*.npz"))

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
    return sorted(set(combos))

def find_available_combos(data_root: Path) -> list[str]:
    """Try NPZ stems first, then the temp text file. Fallback to DEFAULT_COMBO."""
    index_dir = data_root / INDEX_DIRNAME
    combos = read_combos_from_npz(index_dir)
    if combos:
        return combos
    txt = REPO_ROOT / TEMP_DIRNAME / BEST_COMBOS_TXT
    combos = read_combos_from_txt(txt)
    if combos:
        return combos
    return [DEFAULT_COMBO]

# ---------- Train one combo by calling train_cnn1d.py ----------
def run_single_combo(
    combo_display: str,
    data_root: Path,
    index_dir: Path | None,
) -> tuple[float | None, Path, str | None]:
    """
    Calls train_cnn1d.py with minimal args (constants drive the rest).
    Returns: (accuracy or None, results_dir, error or None)
    """
    safe_name = sanitize_combo_name(combo_display)
    train_script = _train_script_path()

    cmd = [
        sys.executable, str(train_script),
        "--data-root", str(data_root),
        "--combo-name", safe_name,
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
        err = None if ret.returncode == 0 else f"train_cnn1d returned {ret.returncode}"
        return acc, results_dir, err
    except Exception as e:
        return None, REPO_ROOT / RESULTS_ROOT / safe_name, str(e)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Train 1D-CNN over all discovered combinations (sequential). "
                    "Hyperparameters are taken from constants.py."
    )
    # No hyperparameter flags here. We only allow optional overrides for paths/files.
    ap.add_argument("--data-root", type=str, required=False,
                    help="Override dataset root (optional). Default is constants.directory.")
    ap.add_argument("--index-dir", type=str, required=False,
                    help=f"Override split CSV folder (optional). Default is <data-root>/{INDEX_DIRNAME}.")
    ap.add_argument("--combos-file", type=str, required=False,
                    help=f"Optional file with combos (one per line; first token used). "
                         f"Default is <repo>/{TEMP_DIRNAME}/{BEST_COMBOS_TXT}.")
    ap.add_argument("--out-csv", type=str, required=False,
                    help=f"Summary CSV path. Default is <repo>/{RESULTS_ROOT}/"
                         f"{SAVE_COMBOS_NAME if SAVE_COMBOS_NAME.lower().endswith('.csv') else SAVE_COMBOS_NAME.rsplit('.',1)[0] + '.csv'}")
    args = ap.parse_args()

    # Resolve roots from constants (allow optional overrides)
    data_root = Path(args.data_root).resolve() if args.data_root else Path(CFG_DATA_ROOT).resolve()

    # Accept common alternative layouts as a convenience (same logic as train_cnn1d)
    alt1 = (REPO_ROOT / "data/asvspoof2019").resolve()
    alt2 = (REPO_ROOT / "database/data/asvspoof2019").resolve()
    if not (data_root / INDEX_DIRNAME).exists():
        if (alt1 / INDEX_DIRNAME).exists():
            data_root = alt1
        elif (alt2 / INDEX_DIRNAME).exists():
            data_root = alt2

    index_dir = Path(args.index_dir).resolve() if args.index_dir else (data_root / INDEX_DIRNAME)

    # Determine output CSV path from constants (allow optional override)
    out_name = SAVE_COMBOS_NAME
    if not out_name.lower().endswith(".csv"):
        out_name = out_name.rsplit(".", 1)[0] + ".csv"
    out_csv = Path(args.out_csv).resolve() if args.out_csv else (REPO_ROOT / RESULTS_ROOT / out_name)
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
