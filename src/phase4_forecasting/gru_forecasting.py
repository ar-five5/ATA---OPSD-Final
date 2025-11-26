"""
Phase 4d: GRU Forecasting & Backtesting
Rolling window forecasting with GRU models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("PHASE 4d: GRU ROLLING WINDOW FORECASTING")
print("="*80)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ============================================================================
# GRU MODEL ARCHITECTURE (must match phase3d)
# ============================================================================

class GRUForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def calculate_mase(actual, forecast, train_data, seasonal_period=24):
    naive_errors = np.abs(train_data.values[seasonal_period:] - train_data.values[:-seasonal_period])
    scale = np.mean(naive_errors)
    errors = np.abs(actual - forecast)
    return np.mean(errors) / scale

def calculate_smape(actual, forecast):
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    return np.mean(np.abs(actual - forecast) / denominator) * 100

def calculate_mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

def calculate_rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast) ** 2))

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/4] Loading dataset...")

df = pd.read_csv('data/time_series_60min_singleindex.csv', index_col=0, parse_dates=True)

countries = ['AT', 'BE', 'BG']
load_columns = {
    'AT': 'AT_load_actual_entsoe_transparency',
    'BE': 'BE_load_actual_entsoe_transparency',
    'BG': 'BG_load_actual_entsoe_transparency'
}

# Prepare data splits
data_splits = {}
for country, load_col in load_columns.items():
    data = df[load_col].dropna()
    hours_120_days = 120 * 24
    if len(data) > hours_120_days:
        data = data.iloc[-hours_120_days:]
    n = len(data)
    train_end = int(0.8 * n)
    dev_end = int(0.9 * n)
    data_splits[country] = {
        'train': data.iloc[:train_end],
        'dev': data.iloc[train_end:dev_end],
        'test': data.iloc[dev_end:]
    }

# ============================================================================
# ROLLING WINDOW FORECASTING
# ============================================================================
print("\n[2/4] Performing rolling window forecasting...")

LOOKBACK = 168
all_results = {}

for country in countries:
    print(f"\n{'='*80}")
    print(f"Forecasting: {country}")
    print(f"{'='*80}")
    
    train_data = data_splits[country]['train']
    test_data = data_splits[country]['test']
    
    # Load trained model
    checkpoint = torch.load(f'results/phase3d_gru_results/{country}_gru_model.pt')
    
    model = GRUForecaster(
        input_size=1,
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        output_size=24,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    
    # Rolling window forecasting
    all_forecasts = []
    all_actual = []
    all_timestamps = []
    
    test_length = len(test_data)
    n_forecasts = test_length // 24
    
    print(f"Generating {n_forecasts} rolling 24-step forecasts...")
    
    for i in range(n_forecasts):
        forecast_origin = i * 24
        
        # Prepare input data
        if forecast_origin == 0:
            fit_data = train_data
        else:
            fit_data = pd.concat([train_data, test_data.iloc[:forecast_origin]])
        
        # Get last LOOKBACK hours
        input_data = fit_data.iloc[-LOOKBACK:].values
        input_normalized = scaler.transform(input_data.reshape(-1, 1)).flatten()
        
        # Forecast
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)
            forecast_normalized = model(input_tensor).cpu().numpy().flatten()
        
        # Inverse transform
        forecast = scaler.inverse_transform(forecast_normalized.reshape(-1, 1)).flatten()
        
        # Get actual values
        actual_window = test_data.iloc[forecast_origin:forecast_origin+24]
        
        all_forecasts.extend(forecast[:len(actual_window)])
        all_actual.extend(actual_window.values)
        all_timestamps.extend(actual_window.index)
    
    # Convert to arrays
    forecasts = np.array(all_forecasts)
    actuals = np.array(all_actual)
    
    # Calculate metrics
    mase = calculate_mase(actuals, forecasts, train_data)
    smape = calculate_smape(actuals, forecasts)
    mape = calculate_mape(actuals, forecasts)
    rmse = calculate_rmse(actuals, forecasts)
    
    print(f"✓ MASE: {mase:.4f} | MAPE: {mape:.2f}% | RMSE: {rmse:.2f} MW")
    
    # Store results
    all_results[country] = {
        'forecasts': forecasts,
        'actuals': actuals,
        'timestamps': all_timestamps,
        'metrics': {
            'MASE': mase,
            'sMAPE_%': smape,
            'MAPE_%': mape,
            'RMSE_MW': rmse,
            'n_forecasts': n_forecasts
        }
    }
    
    # Save forecast data
    forecast_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'actual': actuals,
        'forecast': forecasts,
        'error': actuals - forecasts,
        'abs_error': np.abs(actuals - forecasts),
        'pct_error': ((actuals - forecasts) / actuals) * 100
    })
    forecast_df.to_csv(f'results/phase3d_gru_results/forecast_{country}.csv')

# ============================================================================
# SAVE METRICS SUMMARY
# ============================================================================
print("\n[3/4] Saving metrics summary...")

metrics_summary = {}
for country in countries:
    metrics_summary[country] = all_results[country]['metrics']

with open('results/phase3d_gru_results/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

# Create comparison CSV
comparison_data = []
for country in countries:
    m = all_results[country]['metrics']
    comparison_data.append({
        'Country': country,
        'MASE': round(m['MASE'], 4),
        'MAPE_%': round(m['MAPE_%'], 2),
        'RMSE_MW': round(m['RMSE_MW'], 2)
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('results/phase3d_gru_results/metrics_comparison.csv', index=False)

print("\n" + "="*80)
print("GRU FORECAST METRICS")
print("="*80)
print(comparison_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[4/4] Creating visualizations...")

for country in countries:
    fig, ax = plt.subplots(figsize=(16, 6))
    
    results = all_results[country]
    timestamps = pd.to_datetime(results['timestamps'])
    
    ax.plot(timestamps, results['actuals'], label='Actual', linewidth=2, alpha=0.8)
    ax.plot(timestamps, results['forecasts'], label='GRU Forecast', linewidth=2, alpha=0.8)
    
    metrics = results['metrics']
    title = f"{country} - GRU Forecast: MAPE={metrics['MAPE_%']:.2f}%, RMSE={metrics['RMSE_MW']:.1f} MW, MASE={metrics['MASE']:.4f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Load (MW)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/phase3d_gru_results/forecast_{country}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("✓ Visualizations saved")

print("\n" + "="*80)
print("PHASE 4d COMPLETE: GRU forecasting finished!")
print("="*80)
print("\nResults saved to: results/phase3d_gru_results/")
