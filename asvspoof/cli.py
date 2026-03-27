#!/usr/bin/env python3
from __future__ import annotations
import argparse, os
from pathlib import Path
import pandas as pd

import constants as C
from .config import ExtractConfig, FEATURES_LIST, FEATURE_NAME_MAPPING, INDEX_FOLDER_NAME
# Assuming you have these modules; keep as-is if already present
try:
    from .indexing import load_existing_indices
except Exception:
    # Fallback: minimal loader that expects CSVs in INDEX_FOLDER_NAME
    def load_existing_indices(data_root: Path, index_folder_name: str) -> pd.DataFrame:
        idx_dir = Path(data_root) / index_folder_name
        dfs = []
        for split, name in [("train","train.csv"), ("val","val.csv"), ("test","test.csv"), ("eval","eval.csv")]:
            p = idx_dir / name
            if p.exists():
                df = pd.read_csv(p)
                df["split"] = split
                dfs.append(df)
        if not dfs:
            raise SystemExit(f"No index CSVs found in {idx_dir}")
        df = pd.concat(dfs, ignore_index=True)
        # Normalise expected columns
        rename = {}
        for c in ["utt_id","file_id"]:
            if c in df.columns:
                rename[c] = "file_id"
                break
        if "path" not in df.columns and "abs_path" in df.columns:
            rename["abs_path"] = "path"
        if rename:
            df = df.rename(columns=rename)
        # Build target from label if present
        if "label" in df.columns and "target" not in df.columns:
            df["target"] = (df["label"].astype(str).str.lower() == "bonafide").astype("int16")
        # Ensure abs path
        df["abs_path"] = df["path"].map(lambda p: str(Path(p).resolve()))
        return df

from .features import extract_all_features, normalize_groups
try:
    from .io_utils import write_features_tables
except Exception:
    # Minimal writer
    def write_features_tables(feat_df: pd.DataFrame, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(out_dir / "features_all.parquet", index=False)
        feat_df.to_csv(out_dir / "features_all.csv", index=False)

from .combos import materialize_combos, all_combo_codes, normalize_codes_to_sorted_unique

def _print(msg: str):
    print(msg, flush=True)


def _merge_feature_tables(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["split", "file_id", "path"]
    meta_cols = ["split", "file_id", "path", "label", "target"]

    for c in key_cols:
        if c not in existing.columns or c not in new_df.columns:
            raise SystemExit(f"[!] Missing key column '{c}' for merge.")

    ex = existing.copy()
    nw = new_df.copy()

    ex_idx = ex.set_index(key_cols)
    nw_idx = nw.set_index(key_cols)

    if set(ex_idx.index) != set(nw_idx.index):
        raise SystemExit("[!] Existing features table and fresh extraction have different row keys.")

    nw_idx = nw_idx.reindex(ex_idx.index)
    new_feat_cols = [c for c in nw.columns if c not in meta_cols]
    for col in new_feat_cols:
        ex_idx[col] = nw_idx[col]

    merged = ex_idx.reset_index()
    cols_order = [c for c in meta_cols if c in merged.columns]
    other_cols = sorted([c for c in merged.columns if c not in cols_order])
    return merged[cols_order + other_cols]

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
    groups = normalize_groups(args.groups)

    _print("[1/3] Loading existing indices (train/val/test/eval)...")
    df_index = load_existing_indices(cfg.data_root, INDEX_FOLDER_NAME)

    _print("[1.1] Verifying all referenced files exist...")
    _verify_all_paths_exist(df_index)

    _print(f"[2/3] Extracting features (sequential) for groups: {', '.join(groups)} ...")
    fresh_df = extract_all_features(df_index, cfg, verbose=True, groups=groups, fit_ssl_pca=True)

    parquet_path = out_dir / "features_all.parquet"
    if args.merge_existing and parquet_path.exists():
        _print("[2.1] Merging new feature columns into existing features_all.parquet ...")
        existing_df = pd.read_parquet(parquet_path)
        feat_df = _merge_feature_tables(existing_df, fresh_df)
    else:
        feat_df = fresh_df

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
    from .combos import _effective_letter_maps
    fwd, _ = _effective_letter_maps()
    for g in FEATURES_LIST:
        letter = fwd[g]
        _print(f"  {letter}: {FEATURE_NAME_MAPPING[g]}  ({g})")
    _print("\nTotal combos non-goale: " + str(2 ** len(FEATURES_LIST) - 1))

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ASVspoof LA — extract (secvențial) + combos")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extrage features folosind indecșii existenți (strict, secvențial)")
    pe.add_argument("--data-root", type=str, default=C.directory)
    pe.add_argument("--out-dir", type=str, default=None)
    pe.add_argument("--groups", nargs="*", default=["all"], help="Subset grupuri features sau 'all'")
    pe.add_argument("--merge-existing", action="store_true", help="Adaugă/înlocuiește doar coloanele nou extrase în features_all existent")
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
