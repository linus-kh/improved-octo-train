from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class BBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class InnerCenter:
    lat: float
    lon: float


@dataclass(frozen=True)
class AISConfig:
    date_start: str
    date_end: str
    base_url: str
    raw_dir: Path
    filtered_dir: Path
    combined_path: Path


@dataclass(frozen=True)
class PortWatchConfig:
    ports_url: str
    daily_trade_url: str
    output_path: Path


@dataclass(frozen=True)
class VoyagesConfig:
    output_path: Path


@dataclass(frozen=True)
class ModelingConfig:
    output_path: Path
    results_dir: Path
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class Config:
    port_name_query: str
    outer_bbox: BBox
    inner_bbox: BBox
    inner_center: InnerCenter
    feature_window_minutes: int
    min_sog_knots_for_eta: float
    max_hours_outer_to_inner: int
    reentry_gap_hours: int
    ais: AISConfig
    portwatch: PortWatchConfig
    voyages: VoyagesConfig
    modeling: ModelingConfig


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


def _require_keys(data: Dict[str, Any], keys: list[str], scope: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ConfigError(f"Missing keys in {scope}: {missing}")


def _to_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def load_config(path: str | Path) -> Config:
    config_path = _to_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    _require_keys(
        raw,
        [
            "port_name_query",
            "outer_bbox",
            "inner_bbox",
            "inner_center",
            "feature_window_minutes",
            "min_sog_knots_for_eta",
            "max_hours_outer_to_inner",
            "reentry_gap_hours",
            "ais",
            "portwatch",
            "voyages",
            "modeling",
        ],
        "root",
    )

    outer_bbox = BBox(**raw["outer_bbox"])
    inner_bbox = BBox(**raw["inner_bbox"])
    inner_center = InnerCenter(**raw["inner_center"])

    ais_raw = raw["ais"]
    _require_keys(ais_raw, ["date_start", "date_end", "base_url", "raw_dir", "filtered_dir", "combined_path"], "ais")
    ais = AISConfig(
        date_start=ais_raw["date_start"],
        date_end=ais_raw["date_end"],
        base_url=ais_raw["base_url"],
        raw_dir=_to_path(ais_raw["raw_dir"]),
        filtered_dir=_to_path(ais_raw["filtered_dir"]),
        combined_path=_to_path(ais_raw["combined_path"]),
    )

    portwatch_raw = raw["portwatch"]
    _require_keys(portwatch_raw, ["ports_url", "daily_trade_url", "output_path"], "portwatch")
    portwatch = PortWatchConfig(
        ports_url=portwatch_raw["ports_url"],
        daily_trade_url=portwatch_raw["daily_trade_url"],
        output_path=_to_path(portwatch_raw["output_path"]),
    )

    voyages_raw = raw["voyages"]
    _require_keys(voyages_raw, ["output_path"], "voyages")
    voyages = VoyagesConfig(output_path=_to_path(voyages_raw["output_path"]))

    modeling_raw = raw["modeling"]
    _require_keys(
        modeling_raw,
        ["output_path", "results_dir", "train_start", "train_end", "test_start", "test_end"],
        "modeling",
    )
    modeling = ModelingConfig(
        output_path=_to_path(modeling_raw["output_path"]),
        results_dir=_to_path(modeling_raw["results_dir"]),
        train_start=modeling_raw["train_start"],
        train_end=modeling_raw["train_end"],
        test_start=modeling_raw["test_start"],
        test_end=modeling_raw["test_end"],
    )

    return Config(
        port_name_query=raw["port_name_query"],
        outer_bbox=outer_bbox,
        inner_bbox=inner_bbox,
        inner_center=inner_center,
        feature_window_minutes=int(raw["feature_window_minutes"]),
        min_sog_knots_for_eta=float(raw["min_sog_knots_for_eta"]),
        max_hours_outer_to_inner=int(raw["max_hours_outer_to_inner"]),
        reentry_gap_hours=int(raw["reentry_gap_hours"]),
        ais=ais,
        portwatch=portwatch,
        voyages=voyages,
        modeling=modeling,
    )
