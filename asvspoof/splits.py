#splits.py
from __future__ import annotations
import pandas as pd
from .config import SplitConfig

def build_splits(feat_df: pd.DataFrame, split_cfg: SplitConfig) -> pd.Series:
    """
    Create cv_split ∈ {train, val, test, eval}.

    Strategy
    --------
    - Labeled rows (target in {0,1}): stratified train/val/test
    - Unlabeled (eval): keep cv_split='eval'
    """
    from sklearn.model_selection import train_test_split

    df_lab  = feat_df[feat_df["target"].isin([0, 1])].copy()
    df_eval = feat_df[~feat_df["target"].isin([0, 1])].copy()

    df_trainval, df_test = train_test_split(
        df_lab,
        test_size=split_cfg.test_size,
        random_state=split_cfg.random_state,
        stratify=df_lab["target"],
    )

    val_size_rel = split_cfg.validation_size / (1.0 - split_cfg.test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_size_rel,
        random_state=split_cfg.random_state,
        stratify=df_trainval["target"],
    )

    cv_split = pd.Series(index=feat_df.index, dtype="string")
    cv_split.loc[df_train.index] = "train"
    cv_split.loc[df_val.index]   = "val"
    cv_split.loc[df_test.index]  = "test"
    cv_split.loc[df_eval.index]  = "eval"
    return cv_split
