from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, List

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _coerce_date(value: str | dt.date | dt.datetime) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise TypeError(f"Unsupported date value: {value!r}")


def _date_range(date_start: str | dt.date | dt.datetime, date_end: str | dt.date | dt.datetime) -> Iterable[dt.date]:
    start = _coerce_date(date_start)
    end = _coerce_date(date_end)
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _stream_download(url: str, out_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0))
    with out_path.open("wb") as handle, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=out_path.name,
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
                progress.update(len(chunk))


def download_ais(
    date_start: str | dt.date | dt.datetime,
    date_end: str | dt.date | dt.datetime,
    out_dir: Path,
    base_url: str,
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for day in _date_range(date_start, date_end):
        filename = f"ais-{day.isoformat()}.csv.zst"
        target = out_dir / filename
        if target.exists() and target.stat().st_size > 0:
            logger.info("Skipping existing AIS file %s", target)
            downloaded.append(target)
            continue
        url = f"{base_url}/{filename}"
        logger.info("Downloading %s", url)
        _stream_download(url, target)
        downloaded.append(target)
    return downloaded
