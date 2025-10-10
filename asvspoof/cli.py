#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path
import pandas as pd

import constants as C
from .config import ExtractConfig, FEATURES_LIST, FEATURE_NAME_MAPPING, INDEX_FOLDER_NAME
from .indexing import load_existing_indices
from .features import extract_all_features
from .io_utils import write_features_tables
from .combos import materialize_combos, all_combo_codes, normalize_codes_to_sorted_unique

def _print(msg: str):
    print(msg, flush=True)

def _verify_all_paths_exist(df_index: pd.DataFrame) -> None:
    from pathlib import Path
    missing_mask = ~df_index["abs_path"].map(lambda p: Path(p).exists())
    missing_count = int(missing_mask.sum())
    if missing_count:
        sample = df_index.loc[missing_mask, "abs_path"].head(10).tolist()
        raise SystemExit(
            "[!] Unele fișiere lipsesc pe disc.\n"
            f"    Missing: {missing_count} / {len(df_index)}\n"
            "    Exemple:\n      - " + "\n      - ".join(map(str, sample))
        )

def _cmd_extract(args) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    data_root = Path(args.data_root or C.directory).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (data_root / INDEX_FOLDER_NAME)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExtractConfig(
        data_root=data_root,
        out_dir=out_dir,
    )

    _print("[1/3] Loading existing indices (train/val/test/eval)...")
    df_index = load_existing_indices(cfg.data_root, INDEX_FOLDER_NAME)

    _print("[1.1] Verifying all referenced files exist...")
    _verify_all_paths_exist(df_index)

    _print("[2/3] Extracting features (sequential) ...")
    feat_df = extract_all_features(df_index, cfg, verbose=True)

    meta_cols = {"split", "file_id", "path", "label", "target"}
    feat_cols = [c for c in feat_df.columns if c not in meta_cols]
    if not feat_cols:
        raise SystemExit("Nu s-au extras features — verifică extractorul/dependințele.")

    _print("[3/3] Writing Parquet + CSV to: " + str(out_dir))
    write_features_tables(feat_df, out_dir)
    _print("[✓] Done. Saved: " + str(out_dir / "features_all.parquet"))

def _cmd_combos(args) -> None:
    data_root = Path(args.data_root or C.directory).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (data_root / INDEX_FOLDER_NAME)
    parquet_path = out_dir / "features_all.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"Parquet not found: {parquet_path}. Rulează mai întâi 'extract'.")

    _print("[1/2] Loading features parquet...")
    feat_df = pd.read_parquet(parquet_path)

    codes = all_combo_codes() if args.all else normalize_codes_to_sorted_unique(args.codes)
    if not codes:
        raise SystemExit("Nu ai dat combo-uri. Folosește --all sau --codes ...")

    _print(f"[2/2] Materializing {len(codes)} combos în NPZ pe split (train/val/test)...")
    materialize_combos(feat_df, out_dir, codes)
    _print("[✓] Done.")

def _cmd_list(_args) -> None:
    _print("Feature groups & letters:")
    for g in FEATURES_LIST:
        _print(f"  {FEATURE_NAME_MAPPING[g]}: {g}")
    _print("\nTotal combos non-goale: " + str(2 ** len(FEATURES_LIST) - 1))

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASVspoof LA — extract (secvențial) + combos")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extrage features folosind indecșii existenți (strict, secvențial)")
    pe.add_argument("--data-root", type=str, default=C.directory)
    pe.add_argument("--out-dir", type=str, default=None)
    pe.set_defaults(func=_cmd_extract)

    pc = sub.add_parser("combos", help="Generează NPZ pentru combinații (train/val/test)")
    pc.add_argument("--data-root", type=str, default=C.directory)
    pc.add_argument("--out-dir", type=str, default=None)
    pc.add_argument("--all", action="store_true")
    pc.add_argument("--codes", nargs="*", default=[])
    pc.set_defaults(func=_cmd_combos)

    pl = sub.add_parser("list", help="Mapare features & nr. de combinații")
    pl.set_defaults(func=_cmd_list)
    return p

def main():
    args = build_arg_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
