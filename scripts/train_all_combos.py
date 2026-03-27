#!/usr/bin/env python3
"""
Train combinations by delegating to train_cnn1d.py and build ranked summary.

- Supports explicit --codes / --codes-file to restrict runs.
- Reads metrics from metrics.json + tdcf_metrics.json.
- Writes summary CSV to results/<save_combinations_file_name>.csv.
- Writes ordered combo list to temp_data/<save_the_best_combination_file_name>.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

# ---------- Locate repo root & add to sys.path ----------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "asvspoof") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "asvspoof"))

# ---------- Constants import (robust) ----------
from constants import (  # type: ignore
    directory as CFG_DATA_ROOT,
    index_folder_name as INDEX_DIRNAME,
    results_folder as RESULTS_ROOT,
    temp_data_folder_name as TEMP_DIRNAME,
    final_model_filename as FINAL_MODEL_NAME,
    best_model_filename as BEST_MODEL_NAME,
    save_the_best_combination_file_name as BEST_COMBOS_TXT,
    save_combinations_file_name as SAVE_COMBOS_NAME,
    cnn1d_default_combo_name as DEFAULT_COMBO,
)

try:
    from constants import combo_primary_metric as DEFAULT_PRIMARY_METRIC  # type: ignore
except Exception:
    DEFAULT_PRIMARY_METRIC = "min_tDCF"

try:
    from constants import combo_tie_break_metric as DEFAULT_TIE_BREAK_METRIC  # type: ignore
except Exception:
    DEFAULT_TIE_BREAK_METRIC = "accuracy"

# ---------- Combos normalizer ----------
try:
    from asvspoof.combos import normalize_codes_to_sorted_unique
except Exception:
    from combos import normalize_codes_to_sorted_unique  # type: ignore

ALLOWED_METRICS = ("accuracy", "min_tDCF")


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


def read_combos_from_npz(index_dir: Path) -> List[str]:
    combos_dir = index_dir / "combos" / "train"
    if not combos_dir.exists():
        return []
    stems = sorted(p.stem for p in combos_dir.glob("*.npz"))
    return normalize_codes_to_sorted_unique(stems)


def read_combos_from_txt(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        return []
    raw: List[str] = []
    for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        raw.append(s.split()[0])
    return normalize_codes_to_sorted_unique(raw)


def find_available_combos(data_root: Path) -> List[str]:
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
    # Must mirror train_cnn1d.py (results/<CODE>)
    return REPO_ROOT / RESULTS_ROOT / code


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def collect_metrics(rdir: Path) -> Dict[str, float]:
    m = _load_json(rdir / "metrics.json")
    t = _load_json(rdir / "tdcf_metrics.json")
    return {
        "accuracy": float(m.get("accuracy", np.nan)),
        "balanced_accuracy": float(m.get("balanced_accuracy", np.nan)),
        "eer": float(m.get("eer", np.nan)),
        "min_tDCF": float(t.get("min_tDCF", np.nan)),
    }


def _sort_summary(df: pd.DataFrame, primary_metric: str, tie_break_metric: str) -> pd.DataFrame:
    if df.empty:
        return df

    for col in ["accuracy", "balanced_accuracy", "eer", "min_tDCF"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = [primary_metric]
    ascending = [primary_metric == "min_tDCF"]
    if tie_break_metric and tie_break_metric != primary_metric:
        sort_cols.append(tie_break_metric)
        ascending.append(tie_break_metric == "min_tDCF")

    return df.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def _write_best_list(df_sorted: pd.DataFrame, out_txt: Path) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for _, row in df_sorted.iterrows():
        combo = str(row.get("combo", "")).strip()
        if combo:
            lines.append(combo)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train combos and rank them by selected metric.")
    ap.add_argument("--codes", nargs="*", default=[], help="Explicit combo codes (e.g., AHKLMNO ABHKLMNO)")
    ap.add_argument("--codes-file", type=str, default="", help="Text file with combo codes")
    ap.add_argument("--primary-metric", choices=ALLOWED_METRICS, default=DEFAULT_PRIMARY_METRIC)
    ap.add_argument("--tie-break-metric", choices=ALLOWED_METRICS, default=DEFAULT_TIE_BREAK_METRIC)
    ap.add_argument("--no-write-best-list", action="store_true", help="Do not write ordered combo list txt")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(CFG_DATA_ROOT).resolve()
    index_dir = data_root / INDEX_DIRNAME

    if args.codes:
        combos = normalize_codes_to_sorted_unique(args.codes)
    elif args.codes_file:
        combos = read_combos_from_txt(Path(args.codes_file))
    else:
        combos = find_available_combos(data_root)

    if not combos:
        raise SystemExit("[!] No valid combo codes found.")

    out_name = SAVE_COMBOS_NAME
    if not out_name.lower().endswith(".csv"):
        out_name = out_name.rsplit(".", 1)[0] + ".csv"
    out_csv = (REPO_ROOT / RESULTS_ROOT / out_name).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ordered_txt = (REPO_ROOT / TEMP_DIRNAME / BEST_COMBOS_TXT).resolve()

    print("\n=== Combinations (normalized) ===")
    for c in combos:
        print(" -", c)
    print("Total:", len(combos), "\n")

    train_script = _train_script_path()
    rows: List[Dict[str, Any]] = []

    for i, code in enumerate(combos, 1):
        if not npz_triple_exists(index_dir, code):
            print(f"[{i}/{len(combos)}] SKIP {code} — missing NPZ(s) under {index_dir}/combos/{{train,val,test}}")
            rows.append({
                "combo": code,
                "results_dir": str(results_dir_for(code)),
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "eer": np.nan,
                "min_tDCF": np.nan,
                "best_model": str(results_dir_for(code) / BEST_MODEL_NAME),
                "final_model": str(results_dir_for(code) / FINAL_MODEL_NAME),
                "error": "missing_npz",
            })
            continue

        print(f"[{i}/{len(combos)}] Training combo: {code}")
        ret = subprocess.run([sys.executable, str(train_script), "--code", code], check=False)
        err = None if ret.returncode == 0 else f"train_cnn1d returned {ret.returncode}"

        rdir = results_dir_for(code)
        m = collect_metrics(rdir)

        rows.append({
            "combo": code,
            "results_dir": str(rdir),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "eer": m["eer"],
            "min_tDCF": m["min_tDCF"],
            "best_model": str(rdir / BEST_MODEL_NAME),
            "final_model": str(rdir / FINAL_MODEL_NAME),
            "error": "" if err is None else err,
        })

        df = pd.DataFrame(rows)
        df = _sort_summary(df, args.primary_metric, args.tie_break_metric)
        df.to_csv(out_csv, index=False)

        status = "OK" if err is None else f"ERR: {err}"
        print(
            f" -> {status}; accuracy={m['accuracy']:.6f} min_tDCF={m['min_tDCF']:.6f}; "
            f"saved to {out_csv}"
        )

    df = pd.DataFrame(rows)
    df = _sort_summary(df, args.primary_metric, args.tie_break_metric)
    df.to_csv(out_csv, index=False)

    if not args.no_write_best_list:
        _write_best_list(df, ordered_txt)
        print(f"[i] Ordered combos written to: {ordered_txt}")

    if not df.empty:
        best = df.iloc[0]
        print(
            "[✓] Best combo: "
            f"{best['combo']} ({args.primary_metric}={best.get(args.primary_metric, np.nan)}, "
            f"{args.tie_break_metric}={best.get(args.tie_break_metric, np.nan)})"
        )

    print(f"[✓] Done. Summary at: {out_csv}")


if __name__ == "__main__":
    main()
