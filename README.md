# OPSD PowerDesk: Electric Load Forecasting System

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1+cu118-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-orange.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Advanced Time Series Analysis & Forecasting for European Electric Load Data**
> 
> A comprehensive machine learning pipeline for day-ahead electricity load forecasting using OPSD data for Austria (AT), Belgium (BE), and Bulgaria (BG). Features SARIMA, LSTM, GRU, and Vanilla RNN models with anomaly detection and live monitoring capabilities.

## 🎯 Project Overview

This project implements a complete ML pipeline for electric load forecasting with:
- **4 forecasting models**: SARIMA, LSTM, GRU, Vanilla RNN
- **50,401 hours** of historical data (2014-2020)
- **3 countries** analyzed: Austria, Belgium, Bulgaria
- **Best accuracy**: 0.41 MASE (GRU on Austria)
- **Live monitoring**: 3,500-hour simulation with adaptive refitting
- **Interactive dashboard**: Real-time visualization with Streamlit

## 📊 Key Results

| Model | AT (MASE) | BE (MASE) | BG (MASE) | Avg MASE |
|-------|-----------|-----------|-----------|----------|
| **GRU** | **0.41** 🥇 | 0.95 | 0.82 | **0.73** |
| **Vanilla RNN** | 0.46 | **0.69** 🥇 | 1.11 | 0.75 |
| **LSTM** | 0.67 | 0.63 | 1.25 | 0.85 |
| **SARIMA** | 0.96 | 0.96 | **0.85** 🥇 | 0.92 |

- **Best Overall Model**: GRU (0.73 average MASE)
- **Anomalies Detected**: 21 total (2.15% rate)
- **Perfect Anomaly Classification**: 100% PR-AUC
- **Live Adaptation**: 10 successful refits over 146 days

## 🚀 Quick Start

### Launch Dashboard
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Start interactive dashboard
python -m streamlit run dashboard.py

# Access at: http://localhost:8501
```

### Run Full Pipeline
```powershell
# 1. Data Preparation
python src/data_cleaning.py
python src/data_preprocessing.py

# 2. Model Training
python src/phase3_model_building/sarima_model_building.py
python src/phase3_model_building/lstm_model_building.py
python src/phase3_model_building/gru_model_building.py
python src/phase3_model_building/vanilla_rnn_model_building.py

# 3. Forecasting
python src/phase4_forecasting/sarima_forecasting_backtesting.py
python src/phase4_forecasting/lstm_forecasting_backtesting.py
python src/phase4_forecasting/gru_forecasting_backtesting.py
python src/phase4_forecasting/vanilla_rnn_forecasting_backtesting.py

# 4. Anomaly Detection
python src/phase5_anomaly_detection/anomaly_detection.py
python src/phase5_anomaly_detection/ml_anomaly_classifier.py

# 5. Live Monitoring
python src/phase6_live_adaptation/live_monitoring_simulation.py
```

## 📁 Project Structure

```
ATA/
├── src/                              # Source code
│   ├── data_cleaning.py              # Phase 1: Data validation & cleaning
│   ├── data_preprocessing.py         # Phase 1: Train/val/test splits
│   ├── phase3_model_building/        # Phase 3: Model training scripts
│   │   ├── sarima_model_building.py
│   │   ├── lstm_model_building.py
│   │   ├── gru_model_building.py
│   │   └── vanilla_rnn_model_building.py
│   ├── phase4_forecasting/           # Phase 4: Forecasting scripts
│   ├── phase5_anomaly_detection/     # Phase 5: Anomaly detection
│   └── phase6_live_adaptation/       # Phase 6: Live monitoring
│
├── data/                             # Datasets
│   ├── time_series_60min_singleindex.csv  # Raw OPSD data (50,401 hrs)
│   └── preprocessed/                 # Train/val/test splits
│
├── results/                          # All outputs
│   ├── preprocessing/                # Cleaning visualizations (7 files)
│   ├── phase3d_gru_results/          # GRU training (3 files)
│   ├── phase3e_vanilla_rnn_results/  # RNN training (3 files)
│   ├── phase4_results/               # SARIMA forecasts (8 files)
│   ├── phase4b_lstm_results/         # LSTM forecasts (9 files)
│   ├── phase4c_gru_results/          # GRU forecasts (6 files)
│   ├── phase4d_rnn_results/          # RNN forecasts (6 files)
│   └── phase6_live_adaptation/       # Live simulation (16 files)
│
├── outputs/                          # Anomaly detection results (10 files)
├── phase3_results/                   # SARIMA grid search (5 files)
├── phase3b_lstm_results/             # LSTM training history (4 files)
├── phase4_results/                   # Root-level SARIMA results
├── phase4b_lstm_results/             # Root-level LSTM results
├── docs/                             # Documentation
├── dashboard.py                      # Streamlit dashboard
└── README.md                         # This file
```

## Phases Completed

### ✅ Phase 1: Data Cleaning & Preprocessing
- Data validation and quality checks
- 80/10/10 train/validation/test split (2,304/288/288 hours)
- Data saved to `data/preprocessed/`
- **Dataset:** 50,401 hours (2014-12-31 to 2020-09-30)

### ✅ Phase 2: Exploratory Data Analysis
- Sanity check visualization (14 days)
- STL decomposition for AT, BE, BG
- ACF/PACF analysis
- Stationarity testing (ADF, KPSS)
- **Output:** 11 files in `results/phase2_results/`

### ✅ Phase 3: Model Building
#### SARIMA Grid Search
- 144 combinations per country (p,d,q∈{0,1,2}, P,D,Q∈{0,1}, s=24)
- AIC/BIC-based model selection with validation MSE
- **Output:** `phase3_results/grid_search_partial.csv` (433 models)
- **Selected Orders (by BIC):**
  - AT: SARIMA(2,0,2)(1,1,1,24)
  - BE: SARIMA(2,0,2)(1,1,1,24)
  - BG: SARIMA(2,1,2)(1,1,1,24)

#### Neural Network Models
- **LSTM:** 2 layers, 128 units, 168h lookback, early stopping
- **GRU:** 2 layers, 64 units, batch size 16, 159K params
- **Vanilla RNN:** 3 layers, 128 units, batch size 32
- All models trained with CUDA acceleration on RTX 4050
- **Output:** Trained models in `results/models/`

### ✅ Phase 4: Forecasting & Backtesting
#### SARIMA Forecasting
- 12 rolling 24-step forecasts on test set
- Evaluation metrics: MASE, sMAPE, MAPE, RMSE, MSE, PI Coverage
- **Output:** 8 files in `phase4_results/`
- **Performance:**
  - AT: MAPE 7.14%, MASE 0.96, 75% PI coverage
  - BE: MAPE 5.58%, MASE 0.96, 71% PI coverage
  - BG: MAPE 2.96%, MASE 0.85, 86% PI coverage ⭐ Best

#### Neural Network Forecasting
- All 3 models (LSTM, GRU, RNN) forecasted on test set
- **LSTM Performance:**
  - AT: MAPE 5.24%, MASE 0.67 (⬆ 27% vs SARIMA)
  - BE: MAPE 3.67%, MASE 0.63 (⬆ 34% vs SARIMA)
  - BG: MAPE 4.50%, MASE 1.25
- **GRU Performance:**
  - AT: MASE 0.41 ⭐ Best for AT
  - BE: MASE 0.95
  - BG: MASE 0.82
- **Vanilla RNN Performance:**
  - AT: MASE 0.46
  - BE: MASE 0.69
  - BG: MASE 1.11
- **Output:** Results in `phase4b_lstm_results/`, `results/phase4c_gru_results/`, `results/phase4d_rnn_results/`

### ✅ Phase 5: Anomaly Detection
#### Part 1: Statistical Methods
- Rolling z-score anomaly detection (336h window, |z|≥3.0)
- CUSUM drift detection (k=0.5, h=5.0)
- 976 valid points analyzed across last 1000 hours
- **Output:** 7 files in `outputs/`
- **Results:**
  - 21 total z-score anomalies across all countries
  - AT: 7 anomalies (0.72%), BE: 2 (0.20%), BG: 12 (1.23%)
  - CUSUM: ~63-65% drift detection rate

#### Part 2: ML-Based Classification
- Silver label generation with dual criteria
- Feature engineering (27 features: lags, rolling stats, calendar, CUSUM)
- Logistic Regression + LightGBM classifiers
- **Output:** 3 files in `outputs/`
- **Performance:**
  - Dataset: 473 samples (3 positive, 614 negative)
  - PR-AUC: 1.0000 (both models)
  - F1 @ Precision≥0.80: 1.0000
  - Top feature: `current_z_score_abs` (importance: 276)

### ✅ Phase 6: Live Monitoring & Online Adaptation
- 3,500 hours simulation (146 days) - 75% above minimum
- Rolling SARIMA refit every 336 hours (2 weeks)
- Expanding window training with minimum 1,440 hours history
- **Output:** 16 files in `results/phase6_live_adaptation/`
- **Performance:**
  - AT: MAPE 15.73%, MASE 2.11, 10 refits
  - BE: MAPE 10.16%, MASE 1.91, 10 refits
  - BG: MAPE 14.41%, MASE 3.33, 10 refits

### ✅ Phase 7: Interactive Dashboard
- Streamlit web application with 5 sections
- Real-time model comparison across 4 models (SARIMA, LSTM, GRU, RNN)
- Anomaly detection visualization
- Live monitoring performance evolution
- **Launch:** `python -m streamlit run dashboard.py`
- **Access:** http://localhost:8501

## How to Run

### Quick Start (Dashboard)
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Launch interactive dashboard
python -m streamlit run dashboard.py

# Access at: http://localhost:8501
```

### Full Pipeline Execution
```powershell
# Phase 1: Data Cleaning
python src/data_cleaning.py
python src/data_preprocessing.py

# Phase 3: Model Building (SARIMA + Neural Networks)
python src/phase3_model_building/sarima_model_building.py
python src/phase3_model_building/lstm_model_building.py
python src/phase3_model_building/gru_model_building.py
python src/phase3_model_building/vanilla_rnn_model_building.py

# Phase 4: Forecasting & Backtesting
python src/phase4_forecasting/sarima_forecasting_backtesting.py
python src/phase4_forecasting/lstm_forecasting_backtesting.py
python src/phase4_forecasting/gru_forecasting_backtesting.py
python src/phase4_forecasting/vanilla_rnn_forecasting_backtesting.py

# Phase 5: Anomaly Detection
python src/phase5_anomaly_detection/anomaly_detection.py
python src/phase5_anomaly_detection/ml_anomaly_classifier.py

# Phase 6: Live Monitoring Simulation
python src/phase6_live_adaptation/live_monitoring_simulation.py
```

### Notes
- SARIMA grid search takes ~70 minutes with 2 cores
- Neural network training uses CUDA (RTX 4050)
- Live simulation processes 3,500 hours with periodic refits
- All results are saved incrementally to avoid data loss

## Countries Analyzed
- **AT**: Austria
- **BE**: Belgium
- **BG**: Bulgaria

## Key Findings
1. **SARIMA Strong Baseline**: 2.96-7.14% MAPE across all countries
2. **Neural Networks Excel**: GRU achieved 0.41 MASE for AT (best overall)
3. **BG Most Predictable**: Lowest MAPE (2.96%) with SARIMA
4. **Minimal Anomalies**: Only 21 anomalies in 976 hours (2.15% rate)
5. **Perfect ML Detection**: Anomaly classifier achieves 100% PR-AUC
6. **Live Adaptation Works**: 10 successful SARIMA refits over 3,500 hours
7. **Model Ranking by Average MASE**:
   - 🥇 **GRU: 0.73** (best overall)
   - 🥈 LSTM: 0.85
   - 🥉 SARIMA: 0.92
   - 📊 Vanilla RNN: 0.75

## Technical Stack
- **Python 3.13** with GPU acceleration
- **PyTorch 2.7.1+cu118** for neural networks
- **statsmodels** for SARIMA
- **LightGBM** for anomaly classification
- **Streamlit** for dashboard
- **Hardware:** NVIDIA RTX 4050 Laptop GPU (6.44 GB)
- **Parallelization:** joblib with 2 cores (memory-optimized)
