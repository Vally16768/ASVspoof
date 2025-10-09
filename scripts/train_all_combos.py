#!/usr/bin/env python3
"""
Train all available combinations by delegating to train_cnn1d.py.

- Reads every setting (paths, filenames) from constants.py.
- Discovers combos from <data_root>/<index_folder_name>/combos/train/*.npz,
  or from <repo>/<temp_data_folder_name>/<save_the_best_combination_file_name>,
  else falls back to constants.cnn1d_default_combo_name.
- Verifies NPZs exist for train/val/test before launching.
- Calls:  python train_cnn1d.py --code <CODE>
- Writes a summary CSV to <repo>/<results_folder>/<save_combinations_file_name|.csv>
"""

from __future__ import annotations
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- Repo root & sys.path ----------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------- Constants ----------
from constants import (
    directory as CFG_DATA_ROOT,
    index_folder_name as INDEX_DIRNAME,
    results_folder as RESULTS_ROOT,
    temp_data_folder_name as TEMP_DIRNAME,
    accuracy_txt_filename as ACC_TXT_NAME,
    final_model_filename as FINAL_MODEL_NAME,
    best_model_filename as BEST_MODEL_NAME,
    save_the_best_combination_file_name as BEST_COMBOS_TXT,
    save_combinations_file_name as SAVE_COMBOS_NAME,
    cnn1d_default_combo_name as DEFAULT_COMBO,
)

# ---------- Combos normalizer ----------
try:
    from combos import normalize_codes_to_sorted_unique
except Exception:
    # If used inside a package layout
    from .combos import normalize_codes_to_sorted_unique  # type: ignore

# ---------- Helpers ----------
def _train_script_path() -> Path:
    c1 = REPO_ROOT / "train_cnn1d.py"
    if c1.exists():
        return c1
    c2 = REPO_ROOT / "scripts" / "train_cnn1d.py"
    if c2.exists():
        return c2
    raise SystemExit("[!] Could not locate train_cnn1d.py at repo root or scripts/")

def _npz_path(index_dir: Path, split: str, code: str) -> Path:
    return index_dir / "combos" / split / f"{code}.npz"

def npz_triple_exists(index_dir: Path, code: str) -> bool:
    return all(_npz_path(index_dir, sp, code).exists() for sp in ("train", "val", "test"))

def read_combos_from_npz(index_dir: Path) -> list[str]:
    combos_dir = index_dir / "combos" / "train"
    if not combos_dir.exists():
        return []
    stems = sorted(p.stem for p in combos_dir.glob("*.npz"))
    return normalize_codes_to_sorted_unique(stems)

def read_combos_from_txt(txt_path: Path) -> list[str]:
    if not txt_path.exists():
        return []
    raw = []
    for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        raw.append(s.split()[0])  # first token (accepts "A+B", "AB", etc.)
    return normalize_codes_to_sorted_unique(raw)

def find_available_combos(data_root: Path) -> list[str]:
    index_dir = data_root / INDEX_DIRNAME
    combos = read_combos_from_npz(index_dir)
    if combos:
        return combos
    txt = REPO_ROOT / TEMP_DIRNAME / BEST_COMBOS_TXT
    combos = read_combos_from_txt(txt)
    if combos:
        return combos
    return normalize_codes_to_sorted_unique([DEFAULT_COMBO])

def results_dir_for(code: str) -> Path:
    # Must mirror train_cnn1d.py (which saves to results_folder / f"combo_{code}")
    return REPO_ROOT / RESULTS_ROOT / f"combo_{code}"

# ---------- Main ----------
def main():
    # Everything from constants.py
    data_root = Path(CFG_DATA_ROOT).resolve()
    index_dir = data_root / INDEX_DIRNAME

    # Discover combos
    combos = find_available_combos(data_root)

    # Prepare summary CSV path
    out_name = SAVE_COMBOS_NAME
    if not out_name.lower().endswith(".csv"):
        out_name = out_name.rsplit(".", 1)[0] + ".csv"
    out_csv = (REPO_ROOT / RESULTS_ROOT / out_name).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Show list
    print("\n=== Discovered combinations (normalized codes) ===")
    for c in combos:
        print(" -", c)
    print(f"Total: {len(combos)}\n")

    train_script = _train_script_path()
    rows = []

    for i, code in enumerate(combos, 1):
        if not npz_triple_exists(index_dir, code):
            print(f"[{i}/{len(combos)}] SKIP {code} — missing NPZ(s) under {index_dir}/combos/{{train,val,test}}")
            continue

        print(f"[{i}/{len(combos)}] Training combo: {code}")
        cmd = [sys.executable, str(train_script), "--code", code]  # <-- ONLY argument required
        ret = subprocess.run(cmd, check=False)
        err = None if ret.returncode == 0 else f"train_cnn1d returned {ret.returncode}"

        # Collect results
        rdir = results_dir_for(code)
        acc = np.nan
        acc_file = rdir / ACC_TXT_NAME
        if acc_file.exists():
            import re as _re
            m = _re.search(r"Test\s+accuracy:\s*([0-9.]+)", acc_file.read_text(encoding="utf-8", errors="ignore"))
            if m:
                acc = float(m.group(1))

        rows.append({
            "combo": code,
            "results_dir": str(rdir),
            "accuracy": acc,
            "best_model": str(rdir / BEST_MODEL_NAME),
            "final_model": str(rdir / FINAL_MODEL_NAME),
            "error": ("" if err is None else err),
        })

        # Persist progressive summary
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        status = "OK" if err is None else f"ERR: {err}"
        print(f" -> {status}; accuracy={('NA' if np.isnan(acc) else acc):s}; saved to {out_csv}\n")

    print(f"[✓] Done. Summary at: {out_csv}")

if __name__ == "__main__":
    main()
