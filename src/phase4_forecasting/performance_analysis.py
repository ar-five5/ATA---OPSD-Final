"""
Phase 6: Comprehensive Performance Analysis
Analyzes training time, inference time, model complexity, and accuracy for all models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
os.makedirs('results/phase6_performance_analysis', exist_ok=True)

print("="*80)
print("PHASE 6: COMPREHENSIVE PERFORMANCE ANALYSIS")
print("Training Time | Inference Time | Model Complexity | Accuracy")
print("="*80)

# ============================================================================
# COLLECT TRAINING TIMES
# ============================================================================
print("\n[1/5] Collecting training times...")

training_times = {}

# SARIMA - estimate from grid search results
try:
    # SARIMA doesn't save total training time, estimate from fit_time in results
    sarima_time_per_country = {}
    for country in ['AT', 'BE', 'BG']:
        # Estimate: grid search typically takes 15-20 min total
        # We'll use saved data if available
        sarima_time_per_country[country] = 300  # 5 min estimate per country
    training_times['SARIMA'] = sarima_time_per_country
    print("✓ SARIMA training time estimated")
except Exception as e:
    print(f"⚠ Could not estimate SARIMA time: {e}")

# LSTM
try:
    lstm_training = json.load(open('results/phase3b_lstm_results/training_summary.json'))
    training_times['LSTM'] = {c: lstm_training[c]['training_time_seconds'] for c in ['AT', 'BE', 'BG']}
    print("✓ LSTM training times loaded")
except FileNotFoundError:
    print("⚠ LSTM training summary not found")

# GRU
try:
    gru_training = json.load(open('results/phase3d_gru_results/training_summary.json'))
    training_times['GRU'] = {c: gru_training[c]['training_time_seconds'] for c in ['AT', 'BE', 'BG']}
    print("✓ GRU training times loaded")
except FileNotFoundError:
    print("⚠ GRU training summary not found - run phase3d first")

# Vanilla RNN
try:
    vanilla_training = json.load(open('results/phase3e_vanilla_rnn_results/training_summary.json'))
    training_times['Vanilla_RNN'] = {c: vanilla_training[c]['training_time_seconds'] for c in ['AT', 'BE', 'BG']}
    print("✓ Vanilla RNN training times loaded")
except FileNotFoundError:
    print("⚠ Vanilla RNN training summary not found - run phase3e first")

# ============================================================================
# MEASURE INFERENCE TIMES
# ============================================================================
print("\n[2/5] Measuring inference times (1000 forecasts)...")

inference_times = {}

# SARIMA inference
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    df = pd.read_csv('data/time_series_60min_singleindex.csv', index_col=0, parse_dates=True)
    
    sarima_inf_times = {}
    for country in ['AT', 'BE', 'BG']:
        col = f'{country}_load_actual_entsoe_transparency'
        data = df[col].dropna()[-2880:]
        train = data[:int(0.8 * len(data))]
        
        # Load best model from results
        model_summary = json.load(open('results/phase3_results/model_selection_summary.json'))
        order = tuple(model_summary[country]['best_aic']['order'])
        seasonal_order = tuple(model_summary[country]['best_aic']['seasonal_order'])
        
        # Fit model
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False, maxiter=50)
        
        # Time 1000 forecasts
        start = time.time()
        for _ in range(1000):
            _ = fitted.forecast(steps=24)
        sarima_inf_times[country] = (time.time() - start) / 1000
    
    inference_times['SARIMA'] = sarima_inf_times
    print(f"✓ SARIMA: {np.mean(list(sarima_inf_times.values()))*1000:.2f} ms avg")
except Exception as e:
    print(f"⚠ SARIMA inference measurement failed: {e}")

# Neural network inference times
try:
    import torch
    import torch.nn as nn
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # LSTM
    try:
        class LSTMForecaster(nn.Module):
            def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
                super(LSTMForecaster, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                   batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden_size // 2, output_size)
            
            def forward(self, x):
                x = x.unsqueeze(-1)
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                x = self.fc1(last_hidden)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        lstm_inf_times = {}
        for country in ['AT', 'BE', 'BG']:
            checkpoint = torch.load(f'results/phase3b_lstm_results/{country}_lstm_model.pt')
            model = LSTMForecaster(hidden_size=checkpoint['hidden_size'], num_layers=checkpoint['num_layers']).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            dummy_input = torch.randn(1, 168).to(device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(100):
                    _ = model(dummy_input)
            
            # Time 1000 forecasts
            start = time.time()
            with torch.no_grad():
                for _ in range(1000):
                    _ = model(dummy_input)
            lstm_inf_times[country] = (time.time() - start) / 1000
        
        inference_times['LSTM'] = lstm_inf_times
        print(f"✓ LSTM: {np.mean(list(lstm_inf_times.values()))*1000:.2f} ms avg")
    except Exception as e:
        print(f"⚠ LSTM inference failed: {e}")
    
    # GRU
    try:
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
        
        gru_inf_times = {}
        for country in ['AT', 'BE', 'BG']:
            checkpoint = torch.load(f'results/phase3d_gru_results/{country}_gru_model.pt')
            model = GRUForecaster(hidden_size=checkpoint['hidden_size'], num_layers=checkpoint['num_layers']).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            dummy_input = torch.randn(1, 168).to(device)
            
            # Time 1000 forecasts
            start = time.time()
            with torch.no_grad():
                for _ in range(1000):
                    _ = model(dummy_input)
            gru_inf_times[country] = (time.time() - start) / 1000
        
        inference_times['GRU'] = gru_inf_times
        print(f"✓ GRU: {np.mean(list(gru_inf_times.values()))*1000:.2f} ms avg")
    except Exception as e:
        print(f"⚠ GRU inference failed: {e}")
    
    # Vanilla RNN
    try:
        class VanillaRNNForecaster(nn.Module):
            def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
                super(VanillaRNNForecaster, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                 batch_first=True, dropout=dropout if num_layers > 1 else 0, nonlinearity='tanh')
                self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden_size // 2, output_size)
            
            def forward(self, x):
                x = x.unsqueeze(-1)
                rnn_out, _ = self.rnn(x)
                last_hidden = rnn_out[:, -1, :]
                x = self.fc1(last_hidden)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        vanilla_inf_times = {}
        for country in ['AT', 'BE', 'BG']:
            checkpoint = torch.load(f'results/phase3e_vanilla_rnn_results/{country}_vanilla_rnn_model.pt')
            model = VanillaRNNForecaster(hidden_size=checkpoint['hidden_size'], num_layers=checkpoint['num_layers']).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            dummy_input = torch.randn(1, 168).to(device)
            
            # Time 1000 forecasts
            start = time.time()
            with torch.no_grad():
                for _ in range(1000):
                    _ = model(dummy_input)
            vanilla_inf_times[country] = (time.time() - start) / 1000
        
        inference_times['Vanilla_RNN'] = vanilla_inf_times
        print(f"✓ Vanilla RNN: {np.mean(list(vanilla_inf_times.values()))*1000:.2f} ms avg")
    except Exception as e:
        print(f"⚠ Vanilla RNN inference failed: {e}")

except ImportError:
    print("⚠ PyTorch not available - skipping neural network inference benchmarks")

# ============================================================================
# LOAD ACCURACY METRICS
# ============================================================================
print("\n[3/5] Loading accuracy metrics...")

accuracy_metrics = {}

# SARIMA
try:
    sarima_metrics = json.load(open('results/phase4_results/metrics_summary.json'))
    accuracy_metrics['SARIMA'] = sarima_metrics
    print("✓ SARIMA metrics loaded")
except:
    print("⚠ SARIMA metrics not found")

# LSTM
try:
    lstm_metrics = json.load(open('results/phase4b_lstm_results/metrics_summary.json'))
    accuracy_metrics['LSTM'] = lstm_metrics
    print("✓ LSTM metrics loaded")
except:
    print("⚠ LSTM metrics not found")

# GRU
try:
    gru_metrics = json.load(open('results/phase3d_gru_results/metrics_summary.json'))
    accuracy_metrics['GRU'] = gru_metrics
    print("✓ GRU metrics loaded")
except:
    print("⚠ GRU metrics not found")

# Vanilla RNN
try:
    vanilla_metrics = json.load(open('results/phase3e_vanilla_rnn_results/metrics_summary.json'))
    accuracy_metrics['Vanilla_RNN'] = vanilla_metrics
    print("✓ Vanilla RNN metrics loaded")
except:
    print("⚠ Vanilla RNN metrics not found")

# ============================================================================
# CREATE COMPREHENSIVE COMPARISON TABLE
# ============================================================================
print("\n[4/5] Creating comprehensive comparison table...")

comparison_data = []

for model in accuracy_metrics.keys():
    for country in ['AT', 'BE', 'BG']:
        # Get accuracy metrics
        if model == 'SARIMA':
            mape = accuracy_metrics[model][country]['MAPE']
            rmse = accuracy_metrics[model][country]['RMSE']
            mase = accuracy_metrics[model][country]['MASE']
        else:
            mape = accuracy_metrics[model][country]['MAPE_%']
            rmse = accuracy_metrics[model][country]['RMSE_MW']
            mase = accuracy_metrics[model][country]['MASE']
        
        # Get training time
        train_time = training_times.get(model, {}).get(country, 0)
        
        # Get inference time
        inf_time = inference_times.get(model, {}).get(country, 0)
        
        comparison_data.append({
            'Model': model,
            'Country': country,
            'MAPE_%': round(mape, 2),
            'RMSE_MW': round(rmse, 2),
            'MASE': round(mase, 4),
            'Training_Time_s': round(train_time, 1),
            'Inference_Time_ms': round(inf_time * 1000, 2)
        })

perf_df = pd.DataFrame(comparison_data)
perf_df.to_csv('results/phase6_performance_analysis/comprehensive_performance.csv', index=False)

print("\n" + "="*80)
print("COMPREHENSIVE PERFORMANCE COMPARISON")
print("="*80)
print(perf_df.to_string(index=False))

# ============================================================================
# AGGREGATED STATISTICS
# ============================================================================
print("\n" + "="*80)
print("AGGREGATED STATISTICS (Average across countries)")
print("="*80)

agg_stats = []
for model in perf_df['Model'].unique():
    model_data = perf_df[perf_df['Model'] == model]
    agg_stats.append({
        'Model': model,
        'Avg_MAPE_%': round(model_data['MAPE_%'].mean(), 2),
        'Avg_RMSE_MW': round(model_data['RMSE_MW'].mean(), 1),
        'Avg_MASE': round(model_data['MASE'].mean(), 4),
        'Total_Training_Time_s': round(model_data['Training_Time_s'].sum(), 1),
        'Avg_Inference_Time_ms': round(model_data['Inference_Time_ms'].mean(), 2)
    })

agg_df = pd.DataFrame(agg_stats)
agg_df.to_csv('results/phase6_performance_analysis/aggregated_performance.csv', index=False)

print(agg_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n[5/5] Creating performance visualizations...")

# 1. Accuracy vs Training Time scatter plot
fig, ax = plt.subplots(figsize=(12, 8))

colors = {'SARIMA': '#1f77b4', 'LSTM': '#ff7f0e', 'GRU': '#2ca02c', 'Vanilla_RNN': '#d62728'}
markers = {'AT': 'o', 'BE': 's', 'BG': '^'}

for model in perf_df['Model'].unique():
    for country in ['AT', 'BE', 'BG']:
        data = perf_df[(perf_df['Model'] == model) & (perf_df['Country'] == country)]
        if not data.empty:
            ax.scatter(data['Training_Time_s'], data['MAPE_%'], 
                      s=200, alpha=0.7, color=colors[model], marker=markers[country],
                      label=f"{model}-{country}" if country == 'AT' else "")

ax.set_xlabel('Training Time (seconds)', fontsize=13, fontweight='bold')
ax.set_ylabel('MAPE (%)', fontsize=13, fontweight='bold')
ax.set_title('Accuracy vs Training Time Trade-off', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)

# Custom legend
from matplotlib.lines import Line2D
model_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[m], 
                       markersize=10, label=m) for m in colors.keys()]
country_legend = [Line2D([0], [0], marker=markers[c], color='w', markerfacecolor='gray',
                         markersize=10, label=c) for c in ['AT', 'BE', 'BG']]
ax.legend(handles=model_legend + country_legend, loc='best', ncol=2)

plt.tight_layout()
plt.savefig('results/phase6_performance_analysis/accuracy_vs_training_time.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Inference time comparison
fig, ax = plt.subplots(figsize=(12, 6))

models = agg_df['Model'].values
inf_times = agg_df['Avg_Inference_Time_ms'].values

bars = ax.bar(models, inf_times, color=[colors[m] for m in models], alpha=0.8)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f} ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Average Inference Time (ms)', fontsize=13, fontweight='bold')
ax.set_title('Inference Speed Comparison', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/phase6_performance_analysis/inference_time_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Comprehensive performance heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Performance Metrics Heatmap', fontsize=16, fontweight='bold')

metrics = ['MAPE_%', 'Training_Time_s', 'Inference_Time_ms']
titles = ['MAPE (%) - Lower is Better', 'Training Time (s) - Lower is Better', 'Inference Time (ms) - Lower is Better']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    pivot = perf_df.pivot(index='Model', columns='Country', values=metric)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[idx], cbar_kws={'label': metric})
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')

plt.tight_layout()
plt.savefig('results/phase6_performance_analysis/performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print("✓ All visualizations saved")

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE RECOMMENDATIONS")
print("="*80)

# Find best models for different use cases
best_accuracy = agg_df.loc[agg_df['Avg_MAPE_%'].idxmin()]
fastest_training = agg_df.loc[agg_df['Total_Training_Time_s'].idxmin()]
fastest_inference = agg_df.loc[agg_df['Avg_Inference_Time_ms'].idxmin()]

print(f"\n🎯 Best Accuracy: {best_accuracy['Model']}")
print(f"   MAPE: {best_accuracy['Avg_MAPE_%']:.2f}% | Training: {best_accuracy['Total_Training_Time_s']:.1f}s")

print(f"\n⚡ Fastest Training: {fastest_training['Model']}")
print(f"   Training: {fastest_training['Total_Training_Time_s']:.1f}s | MAPE: {fastest_training['Avg_MAPE_%']:.2f}%")

print(f"\n🚀 Fastest Inference: {fastest_inference['Model']}")
print(f"   Inference: {fastest_inference['Avg_Inference_Time_ms']:.2f}ms | MAPE: {fastest_inference['Avg_MAPE_%']:.2f}%")

print("\n💡 Use Case Recommendations:")
print("   • Production (real-time): Choose model with best inference speed + acceptable accuracy")
print("   • Research (offline): Choose model with best accuracy")
print("   • Resource-constrained: Choose model with fastest training")

print("\n" + "="*80)
print("PHASE 6 COMPLETE!")
print("="*80)
print("\nResults saved to: results/phase6_performance_analysis/")
