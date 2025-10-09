#io_utils.py
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd

from .config import ExtractConfig, SplitConfig

def write_features_tables(feat_df: pd.DataFrame, out_dir: Path) -> None:
    """Write parquet + csv for the master features table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "features_all.parquet"
    csv_path     = out_dir / "features_all.csv"
    feat_df.to_parquet(parquet_path, index=False)
    feat_df.to_csv(csv_path, index=False)

def write_meta(cfg: ExtractConfig, split_cfg: SplitConfig, features_list, mapping, out_dir: Path) -> None:
    """Write a small JSON meta that documents extraction + split config."""
    meta = {
        "extract_config": asdict(cfg),
        "split_config": asdict(split_cfg),
        "features_list": list(features_list),
        "feature_name_mapping": dict(mapping),
    }
    (out_dir / "features_meta.json").write_text(json.dumps(meta, indent=2, default=str))
