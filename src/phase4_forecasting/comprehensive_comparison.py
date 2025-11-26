"""
Phase 5: Comprehensive Model Comparison
Compare SARIMA, LSTM, GRU, and Vanilla RNN models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("PHASE 5: COMPREHENSIVE MODEL COMPARISON")
print("SARIMA vs LSTM vs GRU vs Vanilla RNN")
print("="*80)

# ============================================================================
# LOAD ALL RESULTS
# ============================================================================
print("\n[1/3] Loading results from all models...")

# SARIMA results
sarima_metrics = json.load(open('results/phase4_results/metrics_summary.json'))

# LSTM results
lstm_metrics = json.load(open('results/phase4b_lstm_results/metrics_summary.json'))

# GRU results (will be generated)
try:
    gru_metrics = json.load(open('results/phase3d_gru_results/metrics_summary.json'))
    gru_available = True
except FileNotFoundError:
    print("  ⚠ GRU results not found - run phase3d and phase4d first")
    gru_available = False

# Vanilla RNN results (will be generated)
try:
    vanilla_rnn_metrics = json.load(open('results/phase3e_vanilla_rnn_results/metrics_summary.json'))
    vanilla_available = True
except FileNotFoundError:
    print("  ⚠ Vanilla RNN results not found - run phase3e and phase4e first")
    vanilla_available = False

# ============================================================================
# CREATE COMPARISON TABLE
# ============================================================================
print("\n[2/3] Creating comparison table...")

countries = ['AT', 'BE', 'BG']
comparison_data = []

for country in countries:
    # SARIMA
    comparison_data.append({
        'Country': country,
        'Model': 'SARIMA',
        'MAPE_%': sarima_metrics[country]['MAPE'],
        'RMSE_MW': sarima_metrics[country]['RMSE'],
        'MASE': sarima_metrics[country]['MASE']
    })
    
    # LSTM
    comparison_data.append({
        'Country': country,
        'Model': 'LSTM',
        'MAPE_%': lstm_metrics[country]['MAPE_%'],
        'RMSE_MW': lstm_metrics[country]['RMSE_MW'],
        'MASE': lstm_metrics[country]['MASE']
    })
    
    # GRU
    if gru_available:
        comparison_data.append({
            'Country': country,
            'Model': 'GRU',
            'MAPE_%': gru_metrics[country]['MAPE_%'],
            'RMSE_MW': gru_metrics[country]['RMSE_MW'],
            'MASE': gru_metrics[country]['MASE']
        })
    
    # Vanilla RNN
    if vanilla_available:
        comparison_data.append({
            'Country': country,
            'Model': 'Vanilla_RNN',
            'MAPE_%': vanilla_rnn_metrics[country]['MAPE_%'],
            'RMSE_MW': vanilla_rnn_metrics[country]['RMSE_MW'],
            'MASE': vanilla_rnn_metrics[country]['MASE']
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv('results/all_models_comparison.csv', index=False)

print("\n" + "="*80)
print("ALL MODELS COMPARISON")
print("="*80)
print(comparison_df.to_string(index=False))

# ============================================================================
# FIND BEST MODELS
# ============================================================================
print("\n" + "="*80)
print("BEST MODEL PER COUNTRY (by MAPE)")
print("="*80)

for country in countries:
    country_data = comparison_df[comparison_df['Country'] == country]
    best = country_data.loc[country_data['MAPE_%'].idxmin()]
    print(f"{country}: {best['Model']:12s} → MAPE={best['MAPE_%']:.2f}%, RMSE={best['RMSE_MW']:.1f} MW, MASE={best['MASE']:.4f}")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n[3/3] Creating comparison visualizations...")

# Prepare data for plotting
models_available = ['SARIMA', 'LSTM']
if gru_available:
    models_available.append('GRU')
if vanilla_available:
    models_available.append('Vanilla_RNN')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Model Performance Comparison Across Countries', fontsize=16, fontweight='bold')

metrics_to_plot = ['MAPE_%', 'RMSE_MW', 'MASE']
titles = ['MAPE (%)', 'RMSE (MW)', 'MASE']
colors = {'SARIMA': '#1f77b4', 'LSTM': '#ff7f0e', 'GRU': '#2ca02c', 'Vanilla_RNN': '#d62728'}

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx]
    
    x = np.arange(len(countries))
    width = 0.2
    
    for i, model in enumerate(models_available):
        values = [comparison_df[(comparison_df['Country']==c) & (comparison_df['Model']==model)][metric].values[0] 
                  for c in countries]
        offset = (i - len(models_available)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[model], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel(title, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/all_models_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.close()

print(" Comparison chart saved")

# Create detailed comparison plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Model Performance by Country', fontsize=16, fontweight='bold')

for idx, country in enumerate(countries):
    ax = axes[idx]
    country_data = comparison_df[comparison_df['Country'] == country]
    
    x = np.arange(len(models_available))
    mape_vals = [country_data[country_data['Model']==m]['MAPE_%'].values[0] for m in models_available]
    
    bars = ax.bar(x, mape_vals, color=[colors[m] for m in models_available], alpha=0.8)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{country}: MAPE Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_available, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight best model
    best_idx = np.argmin(mape_vals)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)

plt.tight_layout()
plt.savefig('results/all_models_by_country.png', dpi=150, bbox_inches='tight')
plt.close()

print(" Country comparison chart saved")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for model in models_available:
    model_data = comparison_df[comparison_df['Model'] == model]
    avg_mape = model_data['MAPE_%'].mean()
    avg_rmse = model_data['RMSE_MW'].mean()
    avg_mase = model_data['MASE'].mean()
    
    print(f"\n{model}:")
    print(f"  Avg MAPE: {avg_mape:.2f}%")
    print(f"  Avg RMSE: {avg_rmse:.1f} MW")
    print(f"  Avg MASE: {avg_mase:.4f}")

print("\n" + "="*80)
print("PHASE 5 COMPLETE!")
print("="*80)
print("\nKey findings:")
print("  • All results saved to results/all_models_comparison.csv")
print("  • Visualizations saved to results/")
print("\nTo complete your project:")
if not gru_available:
    print("  1. Run: python src/phase3d_gru_model_building.py")
    print("  2. Run: python src/phase4d_gru_forecasting.py")
if not vanilla_available:
    print("  3. Run: python src/phase3e_vanilla_rnn_model_building.py")
    print("  4. Run: python src/phase4e_vanilla_rnn_forecasting.py")
if gru_available and vanilla_available:
    print("   All models complete! You now have comprehensive comparison.")
