from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import polars as pl
import zstandard as zstd

from ais_portwatch_delay_risk.config import BBox

logger = logging.getLogger(__name__)


AIS_COLUMNS = [
    "MMSI",
    "BaseDateTime",
    "LAT",
    "LON",
    "SOG",
    "COG",
    "Heading",
    "VesselType",
]


def _decompress_to_csv(zst_path: Path, csv_path: Path) -> None:
    dctx = zstd.ZstdDecompressor()
    with zst_path.open("rb") as compressed, csv_path.open("wb") as output:
        with dctx.stream_reader(compressed) as reader:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)


def preprocess_ais_file(zst_path: Path, out_dir: Path, bbox: BBox) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    day_tag = zst_path.stem.replace("ais-", "").replace(".csv", "")
    out_path = out_dir / f"ais_{day_tag}.parquet"
    if out_path.exists() and out_path.stat().st_size > 0:
        logger.info("Skipping existing filtered file %s", out_path)
        return out_path

    tmp_csv = out_dir / f"tmp_{day_tag}.csv"
    logger.info("Decompressing %s", zst_path.name)
    _decompress_to_csv(zst_path, tmp_csv)

    logger.info("Filtering AIS data for %s", zst_path.name)
    scan = pl.scan_csv(tmp_csv, try_parse_dates=False)
    selected_columns = [col for col in AIS_COLUMNS if col in scan.columns]
    lazy_df = (
        scan.select(selected_columns)
        .filter(
            (pl.col("LAT") >= bbox.lat_min)
            & (pl.col("LAT") <= bbox.lat_max)
            & (pl.col("LON") >= bbox.lon_min)
            & (pl.col("LON") <= bbox.lon_max)
        )
        .with_columns(
            pl.col("BaseDateTime").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False)
        )
    )
    lazy_df.collect(streaming=True).write_parquet(out_path)
    tmp_csv.unlink(missing_ok=True)
    return out_path


def preprocess_ais_files(zst_files: Iterable[Path], out_dir: Path, bbox: BBox) -> List[Path]:
    outputs: List[Path] = []
    for zst_path in zst_files:
        outputs.append(preprocess_ais_file(zst_path, out_dir, bbox))
    return outputs


def concatenate_filtered(files: Iterable[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Skipping existing combined parquet %s", output_path)
        return output_path
    files_list = list(files)
    logger.info("Concatenating %d filtered files", len(files_list))
    frames = [pl.scan_parquet(path) for path in files_list]
    if not frames:
        raise ValueError("No filtered files to concatenate")
    pl.concat(frames).collect(streaming=True).write_parquet(output_path)
    return output_path
