#combos.py
from __future__ import annotations
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

from .config import FEATURES_LIST, FEATURE_NAME_MAPPING, FEATURE_NAME_REVERSE_MAPPING

META_COLS = {"split", "file_id", "path", "label", "target", "cv_split"}

def group_columns_from_df(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Map feature group -> list of columns present in df.

    - Scalar groups use exact column name (e.g., zcr_mean).
    - Vector groups use prefix (e.g., mfcc_mean_*).
    """
    groups: Dict[str, List[str]] = {g: [] for g in FEATURES_LIST}
    for col in df.columns:
        if col in META_COLS:
            continue
        for g in FEATURES_LIST:
            if col == g or col.startswith(g + "_"):
                groups[g].append(col)
                break
    for g in groups:
        groups[g] = sorted(groups[g])
    return groups

def all_combo_codes() -> List[str]:
    """All non-empty combinations of feature groups, encoded as letters."""
    letters = [FEATURE_NAME_MAPPING[g] for g in FEATURES_LIST]
    codes: List[str] = []
    for r in range(1, len(letters) + 1):
        for combo in itertools.combinations(letters, r):
            codes.append("".join(sorted(combo)))
    return codes

def normalize_codes_to_sorted_unique(codes: Iterable[str]) -> List[str]:
    return sorted(set("".join(sorted(c.strip().upper())) for c in codes if c.strip()))

def columns_for_code(code: str, group_cols: Dict[str, List[str]]) -> List[str]:
    cols: List[str] = []
    for ch in code:
        g = FEATURE_NAME_REVERSE_MAPPING[ch]
        cols.extend(group_cols.get(g, []))
    return cols

def write_npz(out_path: Path, X: np.ndarray, y: Optional[np.ndarray], columns: List[str], combo_code: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32, copy=False),
        y=(y if y is not None else np.array([], dtype=np.int16)),
        columns=np.array(columns),
        combo_code=np.array(combo_code),
    )

def materialize_combos(feat_df: pd.DataFrame, out_dir: Path, codes: List[str]) -> None:
    """
    Create compressed NPZ matrices for each combo/split.

    Output
    ------
    {out_dir}/combos/{train|val|test}/{CODE}.npz
      - arrays: X (float32), y (int16), columns (str[]), combo_code (str)
    """
    out_dir = Path(out_dir)
    meta_path = out_dir / "combos_meta.json"

    group_cols = group_columns_from_df(feat_df)
    meta = {
        "features_list": FEATURES_LIST,
        "feature_name_mapping": FEATURE_NAME_MAPPING,
        "feature_name_reverse_mapping": FEATURE_NAME_REVERSE_MAPPING,
        "groups_to_columns": group_cols,
        "num_rows": int(len(feat_df)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    df_lab = feat_df[feat_df["cv_split"].isin(["train", "val", "test"])].copy()
    split_idx = {s: df_lab.index[df_lab["cv_split"] == s] for s in ["train", "val", "test"]}

    base_cols = sorted([c for c in df_lab.columns if c not in META_COLS])
    M = df_lab[base_cols]
    y_all = df_lab["target"].astype("int16").to_numpy()
    col_to_pos = {c: i for i, c in enumerate(base_cols)}

    def slice_for(code: str) -> Tuple[List[str], List[int]]:
        cols = columns_for_code(code, group_cols)
        pos = [col_to_pos[c] for c in cols]
        return cols, pos

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **k: x

    for code in tqdm(codes, desc="Combos"):
        cols, pos = slice_for(code)
        if not cols:
            continue
        for split in ["train", "val", "test"]:
            idx = split_idx[split]
            X = M.iloc[idx, pos].to_numpy(dtype=np.float32, copy=False)
            y = y_all[idx]
            out_path = out_dir / "combos" / split / f"{code}.npz"
            write_npz(out_path, X, y, cols, code)
