# TESS Exoplanet Data Pipeline

### Was honored to present the following project at the Data Engineering DC meetup. My presentation is located (here)[https://github.com/yli12313/Presentations-Given/blob/main/20250827_DEDC_Astronomy_PoC/20250827_Data_Engineering_DC_Astronomy_PoC.pdf]!

A Python data pipeline for fetching, analyzing, and visualizing exoplanet data from the *NASA Exoplanet Archive*, focusing on planets discovered by the **Transiting Exoplanet Survey Satellite (TESS)**.

<img width="1956" height="1420" alt="image" src="https://github.com/user-attachments/assets/0629361a-c8aa-4b61-9309-9964a2d4ac9f" />

## Overview

This project demonstrates a complete data engineering workflow for processing astronomical data. It fetches exoplanet data from NASA's archive, stores it in a PostgreSQL database, and generates comprehensive visualizations and statistical analyses.

## Features

- **Data Pipeline**: Automated fetching and deduplication of TESS exoplanet data
- **Database Integration**: PostgreSQL storage with duplicate handling
- **Basic Analytics**: 6-panel visualization suite with mass-radius plots and distributions  
- **Advanced Analytics**: Machine learning clustering, habitability analysis, and interactive 3D visualizations
- **Statistical Analysis**: Correlation matrices, distribution fitting, and discovery timeline analysis

## Quick Start

### Prerequisites

- Python 3.7+
- PostgreSQL database running on localhost
- Database named `meetup_demo` (or configure connection parameters)

### Installation

```bash
pip install requests psycopg2-binary pandas matplotlib seaborn scikit-learn plotly scipy
```

### Usage

1. **Fetch and store data**:
   ```bash
   python main_pipeline.py      # Fetch data from NASA API
   python load_to_database.py   # Load CSV into PostgreSQL
   ```

2. **Generate analyses**:
   ```bash
   python analyze_exoplanets.py    # Basic plots and statistics
   python advanced_analytics.py    # Advanced ML analysis and interactive plots
   ```

## Project Structure

- `main_pipeline.py` - Core data pipeline for fetching and processing TESS exoplanet data
- `load_to_database.py` - Database connection and CSV loading functionality  
- `analyze_exoplanets.py` - Basic statistical analysis and visualization
- `advanced_analytics.py` - Machine learning clustering and advanced visualizations
- `CLAUDE.md` - Development guidelines and architecture documentation

## Data Pipeline

The pipeline queries NASA's Exoplanet Archive for planets with:
- Radius ≤ 1.8 Earth radii  
- Mass > 0
- Discovered by TESS

Data is automatically deduplicated by hostname/planet name, keeping the most recent measurements. Results are saved as timestamped CSV files.

## Visualizations

### Basic Analysis
- Mass vs. Radius scatter plots
- Size and mass distribution histograms
- Planet classification by size categories
- Summary statistics

### Advanced Analysis  
- K-means clustering of planetary properties
- Habitability zone analysis
- Interactive 3D plots with Plotly
- Statistical distribution fitting
- Discovery timeline analysis
- Correlation matrices

## Database Configuration

Default connection settings:
- Host: `localhost`
- Database: `meetup_demo` 
- User: `postgres` (configurable via `DB_USER`)
- Password: empty (configurable via `DB_PASSWORD`)
- Port: `5432` (configurable via `DB_PORT`)

## Output Files

- `tess_exoplanets_YYYYMMDD_HHMMSS.csv` - Raw data exports
- `exoplanet_analysis_YYYYMMDD_HHMMSS.png` - Basic analysis plots
- `advanced_exoplanet_analysis_YYYYMMDD_HHMMSS.png` - Advanced analysis plots  
- `interactive_exoplanet_3d_YYYYMMDD_HHMMSS.html` - Interactive 3D visualizations

## Contributing

This project was created for the Data Engineering D.C. meetup (August 2025). Feel free to extend the analysis or add new visualization features.
