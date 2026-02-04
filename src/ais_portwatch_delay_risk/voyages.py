from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from ais_portwatch_delay_risk.config import BBox, Config

logger = logging.getLogger(__name__)


def in_bbox(lat: float, lon: float, bbox: BBox) -> bool:
    return bbox.lat_min <= lat <= bbox.lat_max and bbox.lon_min <= lon <= bbox.lon_max


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def _path_length_m(latitudes: np.ndarray, longitudes: np.ndarray) -> float:
    if len(latitudes) < 2:
        return 0.0
    distances = [
        haversine_m(latitudes[i], longitudes[i], latitudes[i + 1], longitudes[i + 1])
        for i in range(len(latitudes) - 1)
    ]
    return float(np.sum(distances))


def _safe_std(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.nanstd(values))


def extract_voyage_instances(df_points: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    required = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselType"]
    missing = [col for col in required if col not in df_points.columns]
    if missing:
        raise ValueError(f"Missing columns in AIS data: {missing}")

    df = df_points.copy()
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], utc=True)
    df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

    voyages: List[dict] = []
    for mmsi, group in df.groupby("MMSI"):
        group = group.reset_index(drop=True)
        in_outer = group.apply(lambda row: in_bbox(row["LAT"], row["LON"], cfg.outer_bbox), axis=1)
        in_inner = group.apply(lambda row: in_bbox(row["LAT"], row["LON"], cfg.inner_bbox), axis=1)

        outer_entries = group.index[(in_outer) & (~in_outer.shift(1, fill_value=False))].tolist()
        last_outer_time = None
        for entry_idx in outer_entries:
            entry_time = group.at[entry_idx, "BaseDateTime"]
            if last_outer_time is not None:
                gap_hours = (entry_time - last_outer_time).total_seconds() / 3600
                if gap_hours < cfg.reentry_gap_hours:
                    continue
            post_entry = group.loc[entry_idx:]
            arrival_candidates = post_entry.index[in_inner.loc[entry_idx:]].tolist()
            if not arrival_candidates:
                continue
            arrival_idx = arrival_candidates[0]
            arrival_time = group.at[arrival_idx, "BaseDateTime"]
            hours_to_inner = (arrival_time - entry_time).total_seconds() / 3600
            if hours_to_inner > cfg.max_hours_outer_to_inner:
                continue

            outer_lat = group.at[entry_idx, "LAT"]
            outer_lon = group.at[entry_idx, "LON"]
            sog = group.at[entry_idx, "SOG"]
            if pd.isna(sog) or sog < cfg.min_sog_knots_for_eta:
                sog = cfg.min_sog_knots_for_eta
            speed_mps = float(sog) * 0.514444
            dist_m = haversine_m(outer_lat, outer_lon, cfg.inner_center.lat, cfg.inner_center.lon)
            eta_sec = dist_m / speed_mps if speed_mps > 0 else np.nan
            baseline_arrival = entry_time + pd.Timedelta(seconds=eta_sec)
            delay_minutes = (arrival_time - baseline_arrival).total_seconds() / 60

            window_end = min(arrival_time, entry_time + pd.Timedelta(minutes=cfg.feature_window_minutes))
            segment = group[(group["BaseDateTime"] >= entry_time) & (group["BaseDateTime"] <= window_end)]

            n_points = len(segment)
            duration_minutes = (
                (segment["BaseDateTime"].max() - segment["BaseDateTime"].min()).total_seconds() / 60
                if n_points > 1
                else 0.0
            )
            sog_values = segment["SOG"].to_numpy(dtype=float)
            cog_values = segment["COG"].to_numpy(dtype=float)
            heading_values = segment["Heading"].to_numpy(dtype=float)
            heading_values = heading_values[(~np.isnan(heading_values)) & (heading_values != 511)]

            path_length = _path_length_m(segment["LAT"].to_numpy(), segment["LON"].to_numpy())
            straight_line = (
                haversine_m(
                    segment.iloc[0]["LAT"],
                    segment.iloc[0]["LON"],
                    segment.iloc[-1]["LAT"],
                    segment.iloc[-1]["LON"],
                )
                if n_points > 1
                else np.nan
            )
            tortuosity = path_length / straight_line if straight_line and straight_line > 0 else np.nan

            voyages.append(
                {
                    "MMSI": mmsi,
                    "outer_entry_time": entry_time,
                    "arrival_time": arrival_time,
                    "baseline_eta_minutes": eta_sec / 60 if eta_sec else np.nan,
                    "delay_minutes": delay_minutes,
                    "outer_lat": outer_lat,
                    "outer_lon": outer_lon,
                    "n_points": n_points,
                    "duration_minutes": duration_minutes,
                    "sog_mean": float(np.nanmean(sog_values)) if n_points > 0 else np.nan,
                    "sog_std": _safe_std(sog_values),
                    "sog_min": float(np.nanmin(sog_values)) if n_points > 0 else np.nan,
                    "sog_max": float(np.nanmax(sog_values)) if n_points > 0 else np.nan,
                    "frac_sog_lt_1": float(np.mean(sog_values < 1.0)) if n_points > 0 else np.nan,
                    "frac_sog_lt_3": float(np.mean(sog_values < 3.0)) if n_points > 0 else np.nan,
                    "cog_std": _safe_std(cog_values),
                    "heading_std": _safe_std(heading_values),
                    "tortuosity": tortuosity,
                    "distance_to_inner_center_m": dist_m,
                    "vessel_type": group.at[entry_idx, "VesselType"],
                }
            )
            last_outer_time = entry_time
    voyages_df = pd.DataFrame(voyages)
    return voyages_df
