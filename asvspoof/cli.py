#!/usr/bin/env python3
# === cli.py (strict: uses existing indices; fail fast on errors) ===
from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd

import constants as C
from .config import ExtractConfig, FEATURES_LIST, FEATURE_NAME_MAPPING
from .indexing import load_existing_indices
from .features import extract_all_features
from .io_utils import write_features_tables
from .combos import materialize_combos, all_combo_codes, normalize_codes_to_sorted_unique


def _verify_all_paths_exist(df_index: pd.DataFrame) -> None:
    """Abort immediately if any file is missing; print a short sample."""
    missing_mask = ~df_index["abs_path"].map(lambda p: Path(p).exists())
    missing_count = int(missing_mask.sum())
    if missing_count:
        sample = df_index.loc[missing_mask, "abs_path"].head(10).tolist()
        raise SystemExit(
            "[!] Some indexed audio files do not exist on disk.\n"
            f"    Missing: {missing_count} / {len(df_index)}\n"
            "    First examples:\n      - " + "\n      - ".join(map(str, sample))
        )


def _resolve_workers(arg_workers: int | None) -> int:
    """CLI > constants.workers > sensible default."""
    if arg_workers is not None:
        return int(arg_workers)
    if hasattr(C, "workers"):
        try:
            return int(getattr(C, "workers"))
        except Exception:
            pass
    return max(1, (os.cpu_count() or 4) // 2)


def _cmd_extract(args) -> None:
    data_root = Path(args.data_root or C.directory).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (data_root / C.index_folder_name)

    cfg = ExtractConfig(
        data_root=data_root,
        out_dir=out_dir,
        workers=_resolve_workers(args.workers),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading existing indices (train/val/test/eval)...")
    # Signature: (data_root, index_dirname)
    df_index = load_existing_indices(cfg.data_root, C.index_folder_name)

    print("[1.1] Verifying all referenced files exist...")
    _verify_all_paths_exist(df_index)

    print(f"[2/3] Extracting features strictly (workers={cfg.workers}) ...")
    feat_df = extract_all_features(df_index, cfg)  # will raise immediately on any per-file error

    meta_cols = {"split", "file_id", "path", "label", "target"}
    feat_cols = [c for c in feat_df.columns if c not in meta_cols]
    if len(feat_cols) == 0:
        # This should be unreachable now; kept as belt-and-suspenders.
        raise SystemExit("No features extracted — investigate extractor/deps.")

    print("[3/3] Writing Parquet + CSV to:", out_dir)
    write_features_tables(feat_df, out_dir)
    print("[✓] Done. Saved:", out_dir / "features_all.parquet")


def _cmd_combos(args) -> None:
    data_root = Path(args.data_root or C.directory).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (data_root / C.index_folder_name)
    parquet_path = out_dir / "features_all.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"Parquet not found: {parquet_path}. Run 'extract' first.")

    print("[1/2] Loading features parquet...")
    feat_df = pd.read_parquet(parquet_path)

    if args.all:
        codes = all_combo_codes()
    else:
        codes = normalize_codes_to_sorted_unique(args.codes)
        if not codes:
            raise SystemExit("No combo codes supplied. Use --all or --codes ...")

    print(f"[2/2] Materializing {len(codes)} combos into NPZs per split (train/val/test)...")
    materialize_combos(feat_df, out_dir, codes)
    print("[✓] Done.")


def _cmd_list(_args) -> None:
    print("Feature groups and their letters:")
    for g in FEATURES_LIST:
        print(f"  {FEATURE_NAME_MAPPING[g]}: {g}")
    print("\nAll non-empty combo count:", 2 ** len(FEATURES_LIST) - 1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASVspoof LA — strict extract+combos (no new splits)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract features using existing indices (fail fast)")
    pe.add_argument("--data-root", type=str, default=C.directory,
                    help="Dataset root (defaults to constants.directory)")
    pe.add_argument("--out-dir", type=str, default=None,
                    help="Output dir (defaults to <data_root>/<index_folder_name>)")
    pe.add_argument("--workers", type=int, default=None,
                    help="0 = sequential (no multiprocessing), >=1 = ProcessPool size")
    pe.set_defaults(func=_cmd_extract)

    pc = sub.add_parser("combos", help="Materialize feature combos (train/val/test)")
    pc.add_argument("--data-root", type=str, default=C.directory)
    pc.add_argument("--out-dir", type=str, default=None)
    pc.add_argument("--all", action="store_true", help="Generate ALL non-empty combos")
    pc.add_argument("--codes", nargs="*", default=[], help="Specific combo codes e.g. AB, K, ABLK")
    pc.set_defaults(func=_cmd_combos)

    pl = sub.add_parser("list", help="Show mapping and combo count")
    pl.set_defaults(func=_cmd_list)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
