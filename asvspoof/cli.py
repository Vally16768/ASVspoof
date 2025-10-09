#!/usr/bin/env python3
# === cli.py (minimal; uses existing indices only) ===
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from . import constants as C
from .config import ExtractConfig, FEATURES_LIST, FEATURE_NAME_MAPPING
from .indexing import load_existing_indices
from .features import extract_all_features
from .io_utils import write_features_tables
from .combos import materialize_combos, all_combo_codes, normalize_codes_to_sorted_unique


def _cmd_extract(args) -> None:
    data_root = Path(args.data_root or C.directory).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else (data_root / C.index_folder_name)

    cfg = ExtractConfig(data_root=data_root, out_dir=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading existing indices (train/val/test/eval)...")
    df_index = load_existing_indices(cfg.data_root, C.index_folder_name)

    # sanity: only keep rows that physically exist
    from pathlib import Path as _P
    df_index = df_index[df_index["abs_path"].map(lambda p: _P(p).exists())].reset_index(drop=True)

    print("[2/3] Extracting features (workers=", cfg.workers, ") ...", sep="")
    feat_df = extract_all_features(df_index, cfg)

    meta_cols = {"split", "file_id", "path", "label", "target"}
    feat_cols = [c for c in feat_df.columns if c not in meta_cols]
    if len(feat_cols) == 0:
        raise SystemExit("No features extracted — check audio paths & dependencies.")

    print("[3/3] Writing Parquet + CSV to:", out_dir)
    write_features_tables(feat_df, out_dir)
    print("[✓] Done. Saved:", out_dir / "features_all.parquet")


def _cmd_combos(args) -> None:
    data_root = Path(args.data_root or C.directory).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else (data_root / C.index_folder_name)
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
    p = argparse.ArgumentParser(description="ASVspoof LA — extract+combos (no new splits)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract features using existing indices")
    pe.add_argument("--data-root", type=str, default=C.directory)
    pe.add_argument("--out-dir", type=str, default=None)
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
