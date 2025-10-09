#cli.py
#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
from pathlib import Path

import pandas as pd

from .config import (
    ExtractConfig, SplitConfig, DEFAULTS,
    FEATURES_LIST, FEATURE_NAME_MAPPING
)
from .indexing import build_index_table
from .features import extract_all_features
from .splits import build_splits
from .io_utils import write_features_tables, write_meta
from .combos import (
    materialize_combos, all_combo_codes, normalize_codes_to_sorted_unique
)

def _cmd_extract(args) -> None:
    cfg = ExtractConfig(
        data_root=Path(args.data_root),
        out_dir=Path(args.out_dir or (Path(args.data_root) / "index")),
        n_mels=args.n_mels,
        n_frames=args.n_frames,
        sampling_rate=args.sampling_rate,
        fmax=args.fmax,
        window_length_ms=args.window_length_ms,
        workers=args.workers,
    )
    split_cfg = SplitConfig(
        validation_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building dataset index...")
    df_index = build_index_table(cfg.data_root)

    for sp in ("train", "dev", "eval"):
        total = int((df_index["split"] == sp).sum())
        exists = int(
            df_index[df_index["split"] == sp]["path"].map(lambda p: Path(p).exists()).sum()
        )
        print(f"  {sp}: {exists}/{total} audio files found on disk")

    df_index = df_index[df_index["path"].map(lambda p: Path(p).exists())].reset_index(drop=True)

    print("[2/4] Extracting features...")
    feat_df = extract_all_features(df_index, cfg)

    meta_cols = {"split", "file_id", "path", "label", "target"}
    feat_cols = [c for c in feat_df.columns if c not in meta_cols]
    if len(feat_cols) == 0:
        raise SystemExit(
            "No features extracted. Check dependencies (librosa, soundfile, pywt) "
            "and that .flac files exist on disk."
        )

    print("[3/4] Creating CV splits...")
    feat_df["cv_split"] = build_splits(feat_df, split_cfg)

    print("[4/4] Writing Parquet + CSV + meta...")
    write_features_tables(feat_df, cfg.out_dir)
    write_meta(cfg, split_cfg, FEATURES_LIST, FEATURE_NAME_MAPPING, cfg.out_dir)
    print(f"[✓] Done. Saved under: {cfg.out_dir}")

def _cmd_combos(args) -> None:
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir or (data_root / "index"))
    parquet_path = out_dir / "features_all.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"Parquet not found: {parquet_path}. Run 'extract' first.")

    print("[1/3] Loading features parquet...")
    feat_df = pd.read_parquet(parquet_path)

    if args.all:
        codes = all_combo_codes()
    else:
        codes = normalize_codes_to_sorted_unique(args.codes)
        if not codes:
            raise SystemExit("No combo codes supplied. Use --all or --codes ...")

    print(f"[2/3] Materializing {len(codes)} combinations...")
    materialize_combos(feat_df, out_dir, codes)

    print("[3/3] Done.")

def _cmd_list(_args) -> None:
    print("Feature groups and their letters:")
    for g in FEATURES_LIST:
        print(f"  {FEATURE_NAME_MAPPING[g]}: {g}")
    print("\nAll non-empty combo count:", 2 ** len(FEATURES_LIST) - 1)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASVspoof LA features + combos")
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    pe = sub.add_parser("extract", help="Extract features and build splits")
    pe.add_argument("--data-root", type=str, default="database/data/asvspoof2019")
    pe.add_argument("--out-dir", type=str, default=None)
    pe.add_argument("--n-mels", type=int, default=DEFAULTS["n_mels"])
    pe.add_argument("--n-frames", type=int, default=DEFAULTS["n_frames"])
    pe.add_argument("--sampling-rate", type=int, default=DEFAULTS["sampling_rate"])
    pe.add_argument("--fmax", type=int, default=DEFAULTS["fmax"])
    pe.add_argument("--window-length-ms", type=float, default=DEFAULTS["window_length_ms"])
    pe.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    pe.add_argument("--validation-size", type=float, default=DEFAULTS["validation_size"])
    pe.add_argument("--test-size", type=float, default=DEFAULTS["test_size"])
    pe.add_argument("--random-state", type=int, default=DEFAULTS["random_state"])
    pe.set_defaults(func=_cmd_extract)

    # combos
    pc = sub.add_parser("combos", help="Materialize feature combos into NPZs")
    pc.add_argument("--data-root", type=str, default="database/data/asvspoof2019")
    pc.add_argument("--out-dir", type=str, default=None)
    pc.add_argument("--all", action="store_true", help="Generate ALL non-empty combos")
    pc.add_argument("--codes", nargs="*", default=[], help="Specific combo codes, e.g. AB, AEM, M, ABLK")
    pc.set_defaults(func=_cmd_combos)

    # list
    pl = sub.add_parser("list", help="Show mapping and combo count")
    pl.set_defaults(func=_cmd_list)

    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
