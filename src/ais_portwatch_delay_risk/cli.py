from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ais_portwatch_delay_risk.ais_preprocess import (
    concatenate_filtered,
    preprocess_ais_files,
)
from ais_portwatch_delay_risk.config import load_config
from ais_portwatch_delay_risk.dataset import build_model_dataset
from ais_portwatch_delay_risk.download_ais import download_ais
from ais_portwatch_delay_risk.modeling import train_and_evaluate
from ais_portwatch_delay_risk.portwatch_api import fetch_daily_trade, find_portid_los_angeles
from ais_portwatch_delay_risk.voyages import extract_voyage_instances

logger = logging.getLogger(__name__)


AIS_FEATURE_COLUMNS = [
    "n_points",
    "duration_minutes",
    "sog_mean",
    "sog_std",
    "sog_min",
    "sog_max",
    "frac_sog_lt_1",
    "frac_sog_lt_3",
    "cog_std",
    "heading_std",
    "tortuosity",
    "distance_to_inner_center_m",
]

PORTWATCH_FEATURE_COLUMNS = [
    "portcalls",
    "portcalls_cargo",
    "portcalls_container",
    "portcalls_tanker",
    "import",
    "export",
    "import_container",
    "export_container",
    "import_tanker",
    "export_tanker",
]


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_pipeline(config_path: Path, force: bool = False) -> None:
    cfg = load_config(config_path)

    logger.info("Stage 1: Download AIS")
    zst_files = download_ais(cfg.ais.date_start, cfg.ais.date_end, cfg.ais.raw_dir, cfg.ais.base_url)

    logger.info("Stage 2: Preprocess AIS")
    if force:
        for path in cfg.ais.filtered_dir.glob("ais_*.parquet"):
            path.unlink()
    filtered_files = preprocess_ais_files(zst_files, cfg.ais.filtered_dir, cfg.outer_bbox)
    combined_path = concatenate_filtered(filtered_files, cfg.ais.combined_path)

    logger.info("Stage 3: Extract voyage instances")
    if force and cfg.voyages.output_path.exists():
        cfg.voyages.output_path.unlink()
    if cfg.voyages.output_path.exists() and cfg.voyages.output_path.stat().st_size > 0:
        voyages_df = pd.read_parquet(cfg.voyages.output_path)
    else:
        ais_points = pd.read_parquet(combined_path)
        voyages_df = extract_voyage_instances(ais_points, cfg)
        cfg.voyages.output_path.parent.mkdir(parents=True, exist_ok=True)
        voyages_df.to_parquet(cfg.voyages.output_path, index=False)

    if voyages_df.empty:
        logger.warning("No voyage instances extracted. Exiting early.")
        return

    logger.info("Stage 4: Fetch PortWatch data")
    if force and cfg.portwatch.output_path.exists():
        cfg.portwatch.output_path.unlink()
    if cfg.portwatch.output_path.exists() and cfg.portwatch.output_path.stat().st_size > 0:
        portwatch_df = pd.read_parquet(cfg.portwatch.output_path)
    else:
        portid, portname, _, _ = find_portid_los_angeles(cfg.portwatch.ports_url)
        logger.info("Using PortWatch port %s (%s)", portid, portname)
        portwatch_df = fetch_daily_trade(cfg.portwatch.daily_trade_url, portid)
        cfg.portwatch.output_path.parent.mkdir(parents=True, exist_ok=True)
        portwatch_df.to_parquet(cfg.portwatch.output_path, index=False)

    logger.info("Stage 5: Build modeling dataset")
    if force and cfg.modeling.output_path.exists():
        cfg.modeling.output_path.unlink()
    merged_df, delay_threshold, train_df, test_df = build_model_dataset(
        voyages_df,
        portwatch_df,
        cfg.modeling,
        cfg.modeling.output_path,
    )
    logger.info("Delay threshold (train 75th percentile): %.2f minutes", delay_threshold)

    logger.info("Stage 6: Train and evaluate models")
    train_df = train_df.dropna(subset=AIS_FEATURE_COLUMNS + PORTWATCH_FEATURE_COLUMNS)
    test_df = test_df.dropna(subset=AIS_FEATURE_COLUMNS + PORTWATCH_FEATURE_COLUMNS)
    if train_df.empty or test_df.empty:
        logger.warning("Train or test split empty after dropping missing features. Exiting early.")
        return
    train_and_evaluate(train_df, test_df, AIS_FEATURE_COLUMNS, PORTWATCH_FEATURE_COLUMNS, cfg.modeling.results_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIS + PortWatch delay-risk pipeline")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--force", action="store_true")

    return parser


def main() -> None:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_pipeline(args.config, args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
