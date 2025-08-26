# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a simple Python data pipeline for fetching and processing exoplanet data from the NASA Exoplanet Archive. The project focuses on retrieving data about planets discovered by the Transiting Exoplanet Survey Satellite (TESS).

## Architecture

- **main_pipeline.py**: Single-file Python script containing the complete data pipeline
  - `fetch_exoplanet_data()`: Queries NASA Exoplanet Archive API for TESS-discovered exoplanets
  - `save_csv_data()`: Deduplicates data by hostname/pl_name keeping most recent rowupdate, then saves to CSV
  - `main()`: Orchestrates the data pipeline execution
- **load_to_database.py**: Database loading script
  - `connect_to_database()`: Connects to PostgreSQL database (localhost:meetup_demo)
  - `create_table()`: Creates exoplanets table if it doesn't exist
  - `load_csv_to_database()`: Loads CSV data into database with duplicate handling
- **analyze_exoplanets.py**: Data analysis and visualization script
  - `load_data_from_database()`: Reads exoplanet data from database into pandas DataFrame
  - `create_plots()`: Generates 6-panel visualization with mass/radius plots, distributions, and classifications
  - `print_statistics()`: Outputs detailed statistics about the dataset
- **advanced_analytics.py**: Advanced statistical analysis and machine learning script
  - `create_advanced_plots()`: Generates 12-panel advanced visualization with ML clustering, habitability analysis, and theoretical models
  - `create_interactive_plots()`: Creates interactive 3D Plotly visualizations
  - Includes correlation analysis, statistical distributions, discovery timeline, and habitability scoring

## Common Commands

- **Run the data pipeline**: `python main_pipeline.py`
- **Load data to database**: `python load_to_database.py`
- **Basic analysis and plots**: `python analyze_exoplanets.py`
- **Advanced analytics**: `python advanced_analytics.py`
- **Check Python version**: `python --version`
- **Install dependencies**: `pip install requests psycopg2-binary pandas matplotlib seaborn scikit-learn plotly scipy`

## Data Flow

1. API call to NASA Exoplanet Archive with filters for:
   - Planet radius ≤ 1.8 Earth radii
   - Planet mass > 0
   - Discovered by TESS
2. Data is deduplicated by keeping only the most recent measurement for each hostname/pl_name pair
3. Response saved as CSV with timestamp format: `tess_exoplanets_YYYYMMDD_HHMMSS.csv`
4. Pipeline reports duplicate removal statistics and final record count

## Dependencies

- Python 3.x
- `requests` library for HTTP API calls
- `psycopg2-binary` for PostgreSQL database connections
- `pandas` for data manipulation and analysis
- `matplotlib` and `seaborn` for data visualization
- `numpy` for numerical computations
- `scikit-learn` for machine learning algorithms
- `scipy` for statistical analysis
- `plotly` for interactive visualizations

## Database Setup

The database loading script connects to:
- Host: localhost
- Database: meetup_demo
- Default user: postgres
- Environment variables: DB_USER, DB_PASSWORD, DB_PORT (optional)
