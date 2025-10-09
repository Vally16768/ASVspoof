#indexing.py
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

from .paths import protocol_paths, audio_dir_for_split

def read_cm_protocol(path: Path, split: str) -> pd.DataFrame:
    """
    Parse a CM protocol file into (split, file_id, label?) rows.

    Returns
    -------
    pd.DataFrame with columns: split, file_id, label (None for eval)
    """
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["split", "file_id", "label"])

    pat_file = re.compile(r"^LA_[TDE]_[0-9]+$")
    with path.open("r", encoding="utf-8") as f:
        for line in map(str.strip, f):
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            file_id = next((t for t in parts if pat_file.match(t)), None)
            label = parts[-1].lower() if parts and parts[-1].lower() in {"bonafide", "spoof"} else None
            if file_id:
                rows.append({"split": split, "file_id": file_id, "label": label})
    return pd.DataFrame(rows)

def build_index_table(root: Path) -> pd.DataFrame:
    """
    Build a master index for all protocol rows and attach absolute audio paths.

    Output
    ------
    Columns: split, file_id, path, label, target (1=bonafide, 0=spoof, NaN=eval)
    """
    paths = protocol_paths(root)
    dfs = [read_cm_protocol(paths[s], s) for s in ("train", "dev", "eval")]
    df = pd.concat(dfs, ignore_index=True)

    df["path"] = df.apply(
        lambda r: audio_dir_for_split(root, r["split"]) / f"{r['file_id']}.flac",
        axis=1,
    ).astype("string")

    df["label"] = df["label"].str.lower()
    df["target"] = df["label"].map({"bonafide": 1, "spoof": 0})
    return df
