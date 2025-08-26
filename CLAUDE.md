# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple Python data pipeline for fetching and processing exoplanet data from the NASA Exoplanet Archive. The project focuses on retrieving data about planets discovered by the Transiting Exoplanet Survey Satellite (TESS).

## Architecture

- **main.py**: Single-file Python script containing the complete data pipeline
  - `fetch_exoplanet_data()`: Queries NASA Exoplanet Archive API for TESS-discovered exoplanets
  - `save_csv_data()`: Saves API response to timestamped CSV files
  - `main()`: Orchestrates the data pipeline execution

## Common Commands

- **Run the data pipeline**: `python main.py`
- **Check Python version**: `python --version`
- **Install dependencies**: `pip install requests` (only dependency)

## Data Flow

1. API call to NASA Exoplanet Archive with filters for:
   - Planet radius ≤ 1.8 Earth radii
   - Planet mass > 0
   - Discovered by TESS
2. Response saved as CSV with timestamp format: `tess_exoplanets_YYYYMMDD_HHMMSS.csv`
3. Pipeline reports number of records processed

## Dependencies

- Python 3.x
- `requests` library for HTTP API calls