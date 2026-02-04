from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def arcgis_query_all(
    base_query_url: str,
    where: str,
    out_fields: str = "*",
    batch_size: int = 1000,
    return_type: str = "pandas",
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "resultOffset": offset,
            "resultRecordCount": batch_size,
            "f": "json",
        }
        response = requests.get(base_query_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features", [])
        if not features:
            break
        records.extend(feature.get("attributes", {}) for feature in features)
        offset += batch_size

    df = pd.DataFrame(records)
    if return_type == "polars":
        import polars as pl

        return pl.from_pandas(df)
    if return_type != "pandas":
        raise ValueError(f"Unsupported return_type: {return_type}")
    return df


def find_portid_los_angeles(ports_url: str) -> Tuple[str, str, float, float]:
    where = "UPPER(portname) LIKE '%LOS ANGELES%' AND country = 'United States'"
    df = arcgis_query_all(ports_url, where, out_fields="portid,portname,fullname,latitude,longitude,country")
    if df.empty:
        raise ValueError("No Los Angeles port match found in PortWatch ports database.")

    df["portname_lower"] = df["portname"].str.lower()
    df["fullname_lower"] = df["fullname"].str.lower()
    exact = df[(df["portname_lower"] == "los angeles") | (df["fullname_lower"].str.contains("los angeles"))]
    chosen = exact.iloc[0] if not exact.empty else df.iloc[0]
    return str(chosen["portid"]), str(chosen["portname"]), float(chosen["latitude"]), float(chosen["longitude"])


def fetch_daily_trade(
    daily_trade_url: str,
    portid: str,
    year: int = 2025,
    month: int = 3,
    day_start: int = 1,
    day_end: int = 14,
) -> pd.DataFrame:
    where = (
        f"portid = '{portid}' AND year = {year} AND month = {month} "
        f"AND day >= {day_start} AND day <= {day_end}"
    )
    out_fields = (
        "year,month,day,portid,portname,country,portcalls,portcalls_cargo,"
        "portcalls_container,portcalls_tanker,import,export,import_container,"
        "export_container,import_tanker,export_tanker"
    )
    df = arcgis_query_all(daily_trade_url, where, out_fields=out_fields)
    if df.empty:
        logger.warning("No PortWatch daily trade data found for portid %s", portid)
        return df
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
    return df
