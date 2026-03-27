#!/usr/bin/env python3
"""
End-to-end benchmark extension:
  1) extract only new groups and merge with existing features table
  2) materialize explicit extension combos
  3) train + rank combos
  4) update final_model/ with the winner + metadata + PCA transforms
"""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "asvspoof") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "asvspoof"))

from constants import (  # type: ignore
    directory as CFG_DATA_ROOT,
    index_folder_name as INDEX_DIRNAME,
    results_folder as RESULTS_ROOT,
    save_combinations_file_name as SAVE_COMBOS_NAME,
    best_model_filename as BEST_MODEL_FILENAME,
    combo_primary_metric as DEFAULT_PRIMARY_METRIC,
    combo_tie_break_metric as DEFAULT_TIE_BREAK_METRIC,
)
from asvspoof.combos import _effective_letter_maps, normalize_codes_to_sorted_unique
from asvspoof.config import SSL_WAV2VEC_MODEL_ID, SSL_WAVLM_MODEL_ID, SSL_PCA_COMPONENTS


def _save_combos_csv_path() -> Path:
    name = SAVE_COMBOS_NAME
    if not name.lower().endswith(".csv"):
        name = name.rsplit(".", 1)[0] + ".csv"
    return REPO_ROOT / RESULTS_ROOT / name


def _results_dir_for(code: str) -> Path:
    return REPO_ROOT / RESULTS_ROOT / code


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _best_existing_code(primary_metric: str, tie_break_metric: str) -> str:
    rows: List[Dict[str, Any]] = []
    for mpath in (REPO_ROOT / RESULTS_ROOT).glob("*/metrics.json"):
        code = mpath.parent.name
        metrics = _read_json(mpath)
        tdcf = _read_json(mpath.parent / "tdcf_metrics.json")
        acc = float(metrics.get("accuracy", float("nan")))
        min_tdcf = float(tdcf.get("min_tDCF", float("nan")))
        rows.append({
            "combo": code,
            "accuracy": acc,
            "min_tDCF": min_tdcf,
        })

    if not rows:
        raise SystemExit("[!] Could not detect existing best code from results/*/metrics.json")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=[primary_metric, tie_break_metric],
        ascending=[primary_metric == "min_tDCF", tie_break_metric == "min_tDCF"],
        na_position="last",
    ).reset_index(drop=True)
    return str(df.iloc[0]["combo"])


def _generate_extension_codes(base_code: str, new_groups: List[str]) -> List[str]:
    forward, _ = _effective_letter_maps()
    new_letters = []
    for g in new_groups:
        g_norm = g.strip().lower()
        if g_norm not in forward:
            raise SystemExit(f"[!] Feature group '{g_norm}' not present in letter mapping.")
        new_letters.append(forward[g_norm])

    raw_codes = []
    for r in range(0, len(new_letters) + 1):
        for subset in itertools.combinations(new_letters, r):
            raw_codes.append(base_code + "".join(subset))
    return normalize_codes_to_sorted_unique(raw_codes)


def _run(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise SystemExit(f"[!] Command failed with code {ret.returncode}: {' '.join(cmd)}")


def _copy_final_artifacts(
    best_code: str,
    evaluated_codes: List[str],
    base_code: str,
    primary_metric: str,
    tie_break_metric: str,
) -> None:
    final_dir = REPO_ROOT / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    src_dir = _results_dir_for(best_code)
    src_model = src_dir / BEST_MODEL_FILENAME
    if not src_model.exists():
        raise SystemExit(f"[!] Winning model missing: {src_model}")

    shutil.copy2(src_model, final_dir / BEST_MODEL_FILENAME)

    # Keep labels consistent for binary classifier.
    (final_dir / "labels.txt").write_text("spoof\nbonafide\n", encoding="utf-8")

    # Copy PCA transforms for SSL groups.
    src_transforms = Path(CFG_DATA_ROOT).resolve() / INDEX_DIRNAME / "transforms"
    dst_transforms = final_dir / "transforms"
    dst_transforms.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("wav2vec_pca.joblib", "wavlm_pca.joblib"):
        src = src_transforms / name
        if src.exists():
            shutil.copy2(src, dst_transforms / name)
            copied.append(name)

    winner_metrics = _read_json(src_dir / "metrics.json")
    winner_tdcf = _read_json(src_dir / "tdcf_metrics.json")
    (final_dir / "winner_metrics.json").write_text(json.dumps(winner_metrics, indent=2), encoding="utf-8")
    (final_dir / "winner_tdcf_metrics.json").write_text(json.dumps(winner_tdcf, indent=2), encoding="utf-8")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "combo": best_code,
        "base_code": base_code,
        "evaluated_codes": evaluated_codes,
        "primary_metric": primary_metric,
        "tie_break_metric": tie_break_metric,
        "winner_metrics": winner_metrics,
        "winner_tdcf_metrics": winner_tdcf,
        "ssl": {
            "wav2vec_model_id": SSL_WAV2VEC_MODEL_ID,
            "wavlm_model_id": SSL_WAVLM_MODEL_ID,
            "pca_components": SSL_PCA_COMPONENTS,
            "copied_transforms": copied,
        },
    }
    (final_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extend best combo with LPCC/CQCC/wav2vec/WaveLM and benchmark.")
    ap.add_argument("--base-code", default="", help="Optional base combo override (default: best existing from results)")
    ap.add_argument("--groups", nargs="*", default=["lpcc", "cqcc", "wav2vec", "wavlm"])
    ap.add_argument("--primary-metric", default=DEFAULT_PRIMARY_METRIC, choices=["accuracy", "min_tDCF"])
    ap.add_argument("--tie-break-metric", default=DEFAULT_TIE_BREAK_METRIC, choices=["accuracy", "min_tDCF"])
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-combos", action="store_true")
    ap.add_argument("--no-update-final-model", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    base_code = args.base_code.strip().upper()
    if not base_code:
        base_code = _best_existing_code(args.primary_metric, args.tie_break_metric)
    base_code = normalize_codes_to_sorted_unique([base_code])[0]

    codes = _generate_extension_codes(base_code, [g.lower() for g in args.groups])
    print(f"[i] Base code: {base_code}")
    print(f"[i] Generated {len(codes)} extension codes: {', '.join(codes)}")

    if not args.skip_extract:
        parquet_path = Path(CFG_DATA_ROOT).resolve() / INDEX_DIRNAME / "features_all.parquet"
        if parquet_path.exists():
            _run([
                sys.executable,
                "-m",
                "asvspoof.cli",
                "extract",
                "--groups",
                *[g.lower() for g in args.groups],
                "--merge-existing",
            ])
        else:
            print("[i] features_all.parquet missing -> running full extract")
            _run([
                sys.executable,
                "-m",
                "asvspoof.cli",
                "extract",
            ])

    if not args.skip_combos:
        _run([
            sys.executable,
            "-m",
            "asvspoof.cli",
            "combos",
            "--codes",
            *codes,
        ])

    _run([
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_all_combos.py"),
        "--codes",
        *codes,
        "--primary-metric",
        args.primary_metric,
        "--tie-break-metric",
        args.tie_break_metric,
    ])

    out_csv = _save_combos_csv_path()
    if not out_csv.exists():
        raise SystemExit(f"[!] Summary CSV not found after training: {out_csv}")

    df = pd.read_csv(out_csv)
    if df.empty or "combo" not in df.columns:
        raise SystemExit(f"[!] Invalid summary CSV: {out_csv}")

    best_code = str(df.iloc[0]["combo"]).strip().upper()
    print(f"[✓] Winner: {best_code}")

    if not args.no_update_final_model:
        _copy_final_artifacts(best_code, codes, base_code, args.primary_metric, args.tie_break_metric)
        print("[✓] final_model/ updated with winner model + metadata + PCA transforms")


if __name__ == "__main__":
    main()
