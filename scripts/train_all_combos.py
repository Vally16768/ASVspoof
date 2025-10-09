#!/usr/bin/env python3
"""
Train all available combinations by delegating to train_cnn1d.py.

Improvements vs. previous version
---------------------------------
- Normalizes combo strings (e.g., "A+B+C" -> "ABC") using combos.normalize_codes_to_sorted_unique.
- Ensures NPZs exist for train/val/test before launching a run.
- Passes BOTH --combo-code (letters) and --combo-name (safe tag) to train_cnn1d.py.
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
REPO_ROOT = HERE
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

# Use your combos helpers for robust code handling
try:
    from combos import normalize_codes_to_sorted_unique  # local import when run from repo root
except Exception:
    from .combos import normalize_codes_to_sorted_unique  # package-style import

# ---------- Utilities ----------
def sanitize_combo_name(name: str) -> str:
    """Turn display strings like 'A+B+C' or 'ABC' into a safe directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

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

# ---------- Combo discovery ----------
def read_combos_from_npz(index_dir: Path) -> list[str]:
    """Discover combos from NPZ files: <index>/combos/train/*.npz (use stem as code)."""
    combos_dir = index_dir / "combos" / "train"
    if not combos_dir.exists():
        return []
    stems = sorted(p.stem for p in combos_dir.glob("*.npz"))
    # Ensure only letter codes (stems should already be letters like AB, AEM)
    return normalize_codes_to_sorted_unique(stems)

def read_combos_from_txt(txt_path: Path) -> list[str]:
    """
    Read combos from a text file. Accepts lines like:
      A
      A+B
      A+B+C  accuracy=0.92
      aem
    We normalize to sorted uppercase letter codes (e.g., 'ABC', 'AEM').
    """
    if not txt_path.exists():
        return []
    raw = []
    for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        token = ln.split()[0]  # first token only
        raw.append(token)
    return normalize_codes_to_sorted_unique(raw)

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
    return normalize_codes_to_sorted_unique([DEFAULT_COMBO])

# ---------- Train one combo by calling train_cnn1d.py ----------
def run_single_combo(
    combo_code: str,          # e.g., 'AB', 'AEM', 'M'
    combo_display: str,       # pretty/original form for logs (e.g., 'A+B' or 'AB')
    data_root: Path,
    index_dir: Path | None,
) -> tuple[float | None, Path, str | None]:
    """
    Calls train_cnn1d.py with minimal args (constants drive the rest).
    Returns: (accuracy or None, results_dir, error or None)
    """
    safe_name = sanitize_combo_name(combo_display)  # results/<safe_name>
    train_script = _train_script_path()

    cmd = [
        sys.executable, str(train_script),
        "--data-root", str(data_root),
        "--combo-code", combo_code,     # <-- the IMPORTANT part the trainer needs
        "--combo-name", safe_name,      # <-- tag for results directory
    ]
    if index_dir is not None and index_dir.exists():
        cmd += ["--index-dir", str(index_dir)]

    try:
        ret = subprocess.run(cmd, check=False)
        results_dir = REPO_ROOT / RESULTS_ROOT / safe_name
        acc_file = results_dir / ACC_TXT_NAME
        acc = None
        if acc_file.exists():
            import re as _re
            m = _re.search(r"Test\s+accuracy:\s*([0-9.]+)", acc_file.read_text(encoding="utf-8", errors="ignore"))
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
    ap.add_argument("--data-root", type=str, required=False,
                    help="Override dataset root (optional). Default is constants.directory.")
    ap.add_argument("--index-dir", type=str, required=False,
                    help=f"Override split/combos folder. Default is <data-root>/{INDEX_DIRNAME}.")
    ap.add_argument("--combos-file", type=str, required=False,
                    help=f"Optional file with combos (one per line; accepts 'A+B', 'AB', etc.). "
                         f"Default is <repo>/{TEMP_DIRNAME}/{BEST_COMBOS_TXT}.")
    ap.add_argument("--out-csv", type=str, required=False,
                    help=f"Summary CSV path. Default is <repo>/{RESULTS_ROOT}/"
                         f"{SAVE_COMBOS_NAME if SAVE_COMBOS_NAME.lower().endswith('.csv') else SAVE_COMBOS_NAME.rsplit('.',1)[0] + '.csv'}")
    args = ap.parse_args()

    # Resolve roots (allow optional overrides)
    data_root = Path(args.data_root).resolve() if args.data_root else Path(CFG_DATA_ROOT).resolve()

    # Accept common alternative layouts (same behavior as cli.py extract) :contentReference[oaicite:3]{index=3}
    alt1 = (REPO_ROOT / "data/asvspoof2019").resolve()
    alt2 = (REPO_ROOT / "database/data/asvspoof2019").resolve()
    if not (data_root / INDEX_DIRNAME).exists():
        if (alt1 / INDEX_DIRNAME).exists():
            data_root = alt1
        elif (alt2 / INDEX_DIRNAME).exists():
            data_root = alt2

    index_dir = Path(args.index_dir).resolve() if args.index_dir else (data_root / INDEX_DIRNAME)

    # Determine output CSV path
    out_name = SAVE_COMBOS_NAME
    if not out_name.lower().endswith(".csv"):
        out_name = out_name.rsplit(".", 1)[0] + ".csv"
    out_csv = Path(args.out_csv).resolve() if args.out_csv else (REPO_ROOT / RESULTS_ROOT / out_name)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) discover combos
    combos = read_combos_from_txt(Path(args.combos_file)) if args.combos_file else find_available_combos(data_root)

    # Show list & count
    print("\n=== Discovered combinations (normalized codes) ===")
    for c in combos:
        print(" -", c)
    print(f"Total: {len(combos)}\n")

    # 2) iterate & train
    rows = []
    for i, code in enumerate(combos, 1):
        # Prefer the normalized code as both the loader key and a readable name
        display = code  # or keep original token if you want; sanitized below

        # Verify NPZs exist for this code before launching the run :contentReference[oaicite:4]{index=4}
        if not npz_triple_exists(index_dir, code):
            print(f"[{i}/{len(combos)}] SKIP {code} — missing NPZ(s) under {index_dir}/combos/{{train,val,test}}")
            continue

        print(f"[{i}/{len(combos)}] Training combo: {code}")
        acc, results_dir, err = run_single_combo(
            combo_code=code,
            combo_display=display,
            data_root=data_root,
            index_dir=index_dir if index_dir.exists() else None,
        )
        row = {
            "combo": code,
            "results_dir": str(results_dir),
            "accuracy": (np.nan if acc is None else acc),
            "best_model": str(results_dir / BEST_MODEL_NAME),
            "final_model": str(results_dir / FINAL_MODEL_NAME),
            "error": ("" if err is None else err),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        status = "OK" if err is None else f"ERR: {err}"
        print(f" -> {status}; accuracy={acc if acc is not None else 'NA'}; saved to {out_csv}\n")

    print(f"[✓] Done. Summary at: {out_csv}")

if __name__ == "__main__":
    main()
