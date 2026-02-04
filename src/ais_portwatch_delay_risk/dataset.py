from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ais_portwatch_delay_risk.config import ModelingConfig

logger = logging.getLogger(__name__)


def _time_split(df: pd.DataFrame, modeling_cfg: ModelingConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["outer_date"] = pd.to_datetime(df["outer_entry_time"]).dt.date
    train_mask = (df["outer_date"] >= pd.to_datetime(modeling_cfg.train_start).date()) & (
        df["outer_date"] <= pd.to_datetime(modeling_cfg.train_end).date()
    )
    test_mask = (df["outer_date"] >= pd.to_datetime(modeling_cfg.test_start).date()) & (
        df["outer_date"] <= pd.to_datetime(modeling_cfg.test_end).date()
    )
    return df[train_mask].copy(), df[test_mask].copy()


def build_model_dataset(
    voyages: pd.DataFrame,
    portwatch: pd.DataFrame,
    modeling_cfg: ModelingConfig,
    output_path: Path,
) -> Tuple[pd.DataFrame, float, pd.DataFrame, pd.DataFrame]:
    df = voyages.copy()
    df["outer_date"] = pd.to_datetime(df["outer_entry_time"]).dt.date
    portwatch_df = portwatch.copy()
    if "date" in portwatch_df.columns:
        portwatch_df["date"] = pd.to_datetime(portwatch_df["date"]).dt.date
    merged = df.merge(portwatch_df, left_on="outer_date", right_on="date", how="left", suffixes=("", "_pw"))

    train_df, _ = _time_split(merged, modeling_cfg)
    if train_df.empty:
        raise ValueError("No training data available after time split.")

    delay_threshold = float(np.nanpercentile(train_df["delay_minutes"], 75))
    merged["label_delay_risk"] = (merged["delay_minutes"] > delay_threshold).astype(int)

    train_df, test_df = _time_split(merged, modeling_cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info("Saved modeling dataset to %s", output_path)
    return merged, delay_threshold, train_df, test_df
