# AIS + PortWatch Delay Risk (Port of Los Angeles)

This repository builds a lightweight demo dataset and trains/evaluates binary delay-risk models for inbound voyages to the Port of Los Angeles. It fuses:

1. NOAA MarineCadastre daily AIS data (CSV .zst) for **2025-03-01..2025-03-14**.
2. IMF PortWatch “Daily_Trade_Data” port-day indicators (ArcGIS FeatureServer).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run (single command)

```bash
ais-portwatch-delay-risk run --config configs/la_mar2025.yaml
```

The pipeline stages are restartable and will skip work if outputs already exist. Use `--force` to rerun all stages.

## Expected outputs

- `data/raw/ais/`: downloaded `ais-YYYY-MM-DD.csv.zst` files.
- `data/filtered/`: daily filtered Parquet files plus concatenated file.
- `data/derived/`:
  - `voyages_la_2025-03-01_2025-03-14.parquet`
  - `portwatch_la_2025-03-01_2025-03-14.parquet`
  - `model_dataset.parquet`
- `results/`:
  - `metrics.json`
  - `metrics_table.csv`
  - `fig_roc.png`, `fig_pr.png`, `fig_calibration.png`

## Notes & assumptions

- **Baseline ETA**: Uses a physics-based ETA from the outer entry point to the inner-center using SOG, clamped to a minimum SOG to avoid infinite ETA.
- **AIS noise**: AIS is noisy and sparse; segments are limited to a configurable feature window to keep it lightweight.
- **Feature simplifications**: COG/heading variability is treated as a simple standard deviation (not circular-aware).
- **Small time window**: The demo uses two weeks of data for reproducibility and laptop-scale processing.
