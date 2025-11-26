# Data Preprocessing Guide

## Overview
This document describes the data preprocessing pipeline for the OPSD Power System dataset.

## Script: `src/data_preprocessing.py`

### Phase 1: Data Loading & Exploration
- Loads raw OPSD time series data
- Identifies load columns for Austria (AT), Belgium (BE), and Bulgaria (BG)
- Performs data quality assessment
- Generates statistical summaries

### Phase 2: Data Cleaning & Preprocessing
- **Missing Value Handling**: Forward/backward fill for gaps
- **Outlier Detection**: IQR method (3×IQR bounds)
- **Train/Val/Test Split**: 80/10/10 split
- **Feature Engineering**: Hour, day of week, month, weekend indicator

## Output Files

### Preprocessed Data (`data/preprocessed/`)
- `train_data.csv` - Training set (80%)
- `val_data.csv` - Validation set (10%)
- `test_data.csv` - Test set (10%)
- `cleaned_full_data.csv` - Complete cleaned dataset

### Visualizations (`results/preprocessing/`)
- `load_time_series.png` - Full time series with split markers
- `load_distributions.png` - Load distribution histograms
- `hourly_patterns.png` - Average load by hour of day
- `weekly_patterns.png` - Average load by day of week
- `summary_statistics.csv` - Statistical summary table

## Usage

```bash
python src/data_preprocessing.py
```

## Data Quality Checks
- Missing value percentage per country
- Outlier detection and handling
- Statistical consistency verification
- Train/val/test temporal ordering

## Next Steps
After preprocessing, proceed to:
1. Phase 3: Model Building (SARIMA, LSTM, GRU, RNN)
2. Phase 4: Forecasting & Backtesting
3. Phase 5: Anomaly Detection
4. Phase 6: Live Monitoring
