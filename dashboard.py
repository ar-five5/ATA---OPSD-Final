"""
OPSD PowerDesk: Interactive Dashboard
Day-Ahead Forecasting, Anomaly Detection, and Live Monitoring Dashboard

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="OPSD PowerDesk Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚡ OPSD PowerDesk Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Day-Ahead Forecasting, Anomaly Detection & Live Monitoring")
st.markdown("---")

# LOAD DATA

@st.cache_data
def load_sarima_forecasts():
    """Load SARIMA forecast results"""
    forecasts = {}
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'results/phase4_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
            forecasts[country] = df
        except Exception:
            st.warning(f"Could not load SARIMA forecasts for {country}")
    return forecasts

@st.cache_data
def load_lstm_forecasts():
    """Load LSTM forecast results"""
    forecasts = {}
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'results/phase4b_lstm_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
            forecasts[country] = df
        except Exception:
            pass
    return forecasts

@st.cache_data
def load_gru_forecasts():
    """Load GRU forecast results"""
    forecasts = {}
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'results/phase4c_gru_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
            forecasts[country] = df
        except Exception:
            pass
    return forecasts

@st.cache_data
def load_rnn_forecasts():
    """Load Vanilla RNN forecast results"""
    forecasts = {}
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'results/phase4d_rnn_results/forecast_data_{country}.csv', parse_dates=['timestamp'])
            forecasts[country] = df
        except Exception:
            pass
    return forecasts

@st.cache_data
def load_anomalies():
    """Load anomaly detection results"""
    anomalies = {}
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'outputs/{country}_anomalies.csv', parse_dates=['timestamp'])
            anomalies[country] = df
        except Exception:
            st.warning(f"Could not load anomalies for {country}")
    return anomalies

@st.cache_data
def load_live_simulation():
    """Load live monitoring simulation results for all 4 models"""
    live_data = {'SARIMA': {}, 'LSTM': {}, 'GRU': {}, 'RNN': {}}
    
    # Load SARIMA results
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'results/phase6_live_adaptation/{country}_live_simulation.csv', parse_dates=['timestamp'])
            live_data['SARIMA'][country] = df
        except Exception:
            pass
    
    # Load LSTM results
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'results/phase6_live_adaptation/{country}_lstm_live_simulation.csv', parse_dates=['timestamp'])
            live_data['LSTM'][country] = df
        except Exception:
            pass
    
    # Load GRU results
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'results/phase6_live_adaptation/{country}_gru_live_simulation.csv', parse_dates=['timestamp'])
            live_data['GRU'][country] = df
        except Exception:
            pass
    
    # Load Vanilla RNN results
    for country in ['AT', 'BE', 'BG']:
        try:
            df = pd.read_csv(f'results/phase6_live_adaptation/{country}_rnn_live_simulation.csv', parse_dates=['timestamp'])
            live_data['RNN'][country] = df
        except Exception:
            pass
    
    return live_data

@st.cache_data
def load_metrics():
    """Load model comparison metrics"""
    try:
        sarima_metrics = pd.read_csv('results/phase4_results/metrics_comparison.csv', index_col=0)
        lstm_metrics = pd.read_csv('results/phase4b_lstm_results/metrics_comparison.csv', index_col=0)
        gru_metrics = pd.read_csv('results/phase4c_gru_results/metrics_comparison.csv', index_col=0)
        rnn_metrics = pd.read_csv('results/phase4d_rnn_results/metrics_comparison.csv', index_col=0)
        return sarima_metrics, lstm_metrics, gru_metrics, rnn_metrics
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None, None, None, None

# Load all data
with st.spinner("Loading data..."):
    sarima_forecasts = load_sarima_forecasts()
    lstm_forecasts = load_lstm_forecasts()
    gru_forecasts = load_gru_forecasts()
    rnn_forecasts = load_rnn_forecasts()
    anomalies = load_anomalies()
    live_data = load_live_simulation()
    sarima_metrics, lstm_metrics, gru_metrics, rnn_metrics = load_metrics()

# SIDEBAR

st.sidebar.header("⚙️ Settings")

# Country selection
countries = st.sidebar.multiselect(
    "Select Countries",
    options=['AT', 'BE', 'BG'],
    default=['AT', 'BE', 'BG']
)

# Section selection
section = st.sidebar.radio(
    "Navigate to Section",
    options=[
        "📊 Overview",
        "📈 Forecast Comparison",
        "🚨 Anomaly Detection",
        "🔄 Live Monitoring",
        "🏆 Model Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This dashboard visualizes day-ahead electric load forecasting results "
    "for Austria (AT), Belgium (BE), and Bulgaria (BG) using OPSD data. "
    "\n\n**Models:** SARIMA, LSTM, GRU, Vanilla RNN"
)

# SECTION 1: OVERVIEW

if section == "📊 Overview":
    st.header("📊 Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Countries Analyzed", "3", help="AT, BE, BG")
    
    with col2:
        st.metric("Models Tested", "4", help="SARIMA, LSTM, GRU, Vanilla RNN")
    
    with col3:
        st.metric("Live Simulation", "3,500 hrs", help="146 days of streaming data")
    
    with col4:
        if sarima_metrics is not None:
            best_mape = sarima_metrics['MAPE'].min()
            st.metric("Best MAPE", f"{best_mape:.2f}%", help="Bulgaria SARIMA")
    
    st.markdown("---")
    
    # Key Findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Findings")
        st.markdown("""
        - **Excellent Accuracy:** SARIMA achieved 3.31-7.14% MAPE across all countries
        - **LSTM Improvement:** Neural models outperformed SARIMA by 27-34% in MASE
        - **Live Adaptation:** 10 successful model refits over 3,500 hours
        - **Anomaly Detection:** 21 total anomalies detected using z-score + CUSUM
        - **Data Quality:** 2× minimum training requirement (96 vs 60 days)
        """)
    
    with col2:
        st.subheader("📋 Assignment Compliance")
        st.markdown("""
        ✅ **Phase 1:** Data preprocessing & STL decomposition  
        ✅ **Phase 2:** ACF/PACF analysis & SARIMA grid search  
        ✅ **Phase 3:** Model building (SARIMA + 3 neural models)  
        ✅ **Phase 4:** 24-step forecasting with all metrics  
        ✅ **Phase 5:** Anomaly detection (z-score, CUSUM, ML)  
        ✅ **Phase 6:** Live monitoring (3,500 hours simulation)  
        ✅ **Phase 7:** Interactive dashboard (this app!)  
        """)
    
    # Dataset Overview
    st.markdown("---")
    st.subheader("📁 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Training Data**")
        st.write("• 2,304 hours (96 days)")
        st.write("• 80% split")
        st.write("• June 3 - Sep 6, 2020")
    
    with col2:
        st.markdown("**Validation Data**")
        st.write("• 288 hours (12 days)")
        st.write("• 10% split")
        st.write("• Sequential after training")
    
    with col3:
        st.markdown("**Test Data**")
        st.write("• 288 hours (12 days)")
        st.write("• 10% split")
        st.write("• Sep 19-30, 2020")

# SECTION 2: FORECAST COMPARISON

elif section == "📈 Forecast Comparison":
    st.header("📈 Forecast Comparison")
    
    if not countries:
        st.warning("Please select at least one country from the sidebar.")
    else:
        # Country tabs
        tabs = st.tabs(countries)
        
        for idx, country in enumerate(countries):
            with tabs[idx]:
                st.subheader(f"{country} - Day-Ahead Forecasts")
                
                # Get data
                sarima_df = sarima_forecasts.get(country)
                lstm_df = lstm_forecasts.get(country)
                
                if sarima_df is None:
                    st.error(f"No forecast data available for {country}")
                    continue
                
                # Create forecast plot
                fig = go.Figure()
                
                # Actual load
                fig.add_trace(go.Scatter(
                    x=sarima_df['timestamp'],
                    y=sarima_df['actual'],
                    name='Actual Load',
                    line=dict(color='black', width=2),
                    mode='lines'
                ))
                
                # SARIMA forecast
                fig.add_trace(go.Scatter(
                    x=sarima_df['timestamp'],
                    y=sarima_df['forecast'],
                    name='SARIMA Forecast',
                    line=dict(color='blue', width=1.5),
                    mode='lines'
                ))
                
                # SARIMA prediction intervals
                fig.add_trace(go.Scatter(
                    x=sarima_df['timestamp'],
                    y=sarima_df['upper_bound_80%'],
                    fill=None,
                    mode='lines',
                    line=dict(color='lightblue', width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=sarima_df['timestamp'],
                    y=sarima_df['lower_bound_80%'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='lightblue', width=0),
                    name='80% Prediction Interval',
                    fillcolor='rgba(173, 216, 230, 0.3)'
                ))
                
                # LSTM forecast if available
                if lstm_df is not None:
                    fig.add_trace(go.Scatter(
                        x=lstm_df['timestamp'],
                        y=lstm_df['forecast'],
                        name='LSTM Forecast',
                        line=dict(color='green', width=1.5, dash='dash'),
                        mode='lines'
                    ))
                
                # GRU forecast if available
                gru_df = gru_forecasts.get(country)
                if gru_df is not None:
                    fig.add_trace(go.Scatter(
                        x=gru_df['timestamp'],
                        y=gru_df['forecast'],
                        name='GRU Forecast',
                        line=dict(color='orange', width=1.5, dash='dot'),
                        mode='lines'
                    ))
                
                # Vanilla RNN forecast if available
                rnn_df = rnn_forecasts.get(country)
                if rnn_df is not None:
                    fig.add_trace(go.Scatter(
                        x=rnn_df['timestamp'],
                        y=rnn_df['forecast'],
                        name='Vanilla RNN Forecast',
                        line=dict(color='red', width=1.5, dash='dashdot'),
                        mode='lines'
                    ))
                
                fig.update_layout(
                    title=f"{country} - Actual Load vs Forecasts",
                    xaxis_title="Timestamp",
                    yaxis_title="Load (MW)",
                    hovermode='x unified',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics table
                st.markdown("#### Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if sarima_metrics is not None and country in sarima_metrics.index:
                        st.markdown("**SARIMA Model**")
                        metrics_df = sarima_metrics.loc[[country]]
                        st.dataframe(metrics_df.style.format({
                            'MASE': '{:.4f}',
                            'sMAPE': '{:.2f}',
                            'MAPE': '{:.2f}',
                            'RMSE': '{:.2f}',
                            'MSE': '{:.2f}',
                            'PI_Coverage_80%': '{:.2f}'
                        }), use_container_width=True)
                
                with col2:
                    if lstm_metrics is not None and country in lstm_metrics.index:
                        st.markdown("**LSTM Model**")
                        lstm_df_metrics = lstm_metrics.loc[[country]]
                        st.dataframe(lstm_df_metrics.style.format({
                            'MASE': '{:.4f}',
                            'sMAPE_%': '{:.2f}',
                            'MAPE_%': '{:.2f}',
                            'RMSE_MW': '{:.2f}',
                            'MSE': '{:.2f}'
                        }), use_container_width=True)
                
                with col3:
                    if gru_metrics is not None and country in gru_metrics.index:
                        st.markdown("**GRU Model**")
                        gru_df_metrics = gru_metrics.loc[[country]]
                        st.dataframe(gru_df_metrics.style.format({
                            'MASE': '{:.4f}',
                            'sMAPE_%': '{:.2f}',
                            'MAPE_%': '{:.2f}',
                            'RMSE_MW': '{:.2f}',
                            'MSE': '{:.2f}'
                        }), use_container_width=True)
                
                with col4:
                    if rnn_metrics is not None and country in rnn_metrics.index:
                        st.markdown("**Vanilla RNN Model**")
                        rnn_df_metrics = rnn_metrics.loc[[country]]
                        st.dataframe(rnn_df_metrics.style.format({
                            'MASE': '{:.4f}',
                            'sMAPE_%': '{:.2f}',
                            'MAPE_%': '{:.2f}',
                            'RMSE_MW': '{:.2f}',
                            'MSE': '{:.2f}'
                        }), use_container_width=True)

# SECTION 3: ANOMALY DETECTION

elif section == "🚨 Anomaly Detection":
    st.header("🚨 Anomaly Detection Results")
    
    if not countries:
        st.warning("Please select at least one country from the sidebar.")
    else:
        # Country tabs
        tabs = st.tabs(countries)
        
        for idx, country in enumerate(countries):
            with tabs[idx]:
                st.subheader(f"{country} - Detected Anomalies")
                
                anom_df = anomalies.get(country)
                
                if anom_df is None:
                    st.error(f"No anomaly data available for {country}")
                    continue
                
                # Anomaly statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_points = len(anom_df)
                    st.metric("Total Points", f"{total_points:,}")
                
                with col2:
                    z_anomalies = anom_df['flag_z'].sum()
                    z_rate = (z_anomalies / total_points * 100)
                    st.metric("Z-score Anomalies", z_anomalies, f"{z_rate:.2f}%")
                
                with col3:
                    if 'flag_cusum' in anom_df.columns:
                        cusum_anomalies = anom_df['flag_cusum'].sum()
                        cusum_rate = (cusum_anomalies / total_points * 100)
                        st.metric("CUSUM Anomalies", cusum_anomalies, f"{cusum_rate:.2f}%")
                
                with col4:
                    max_z = anom_df['z_resid'].abs().max()
                    st.metric("Max |Z-score|", f"{max_z:.2f}")
                
                # Create anomaly visualization
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        "Load with Anomalies Flagged",
                        "Rolling Z-score of Residuals"
                    ),
                    vertical_spacing=0.12,
                    row_heights=[0.6, 0.4]
                )
                
                # Plot 1: Load with anomalies
                fig.add_trace(
                    go.Scatter(
                        x=anom_df['timestamp'],
                        y=anom_df['y_true'],
                        name='Actual Load',
                        line=dict(color='black', width=1),
                        mode='lines'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=anom_df['timestamp'],
                        y=anom_df['yhat'],
                        name='Forecast',
                        line=dict(color='blue', width=1),
                        mode='lines',
                        opacity=0.6
                    ),
                    row=1, col=1
                )
                
                # Highlight anomalies
                anomaly_points = anom_df[anom_df['flag_z'] == 1]
                if len(anomaly_points) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_points['timestamp'],
                            y=anomaly_points['y_true'],
                            name='Z-score Anomaly',
                            mode='markers',
                            marker=dict(color='red', size=8, symbol='circle')
                        ),
                        row=1, col=1
                    )
                
                # Plot 2: Z-scores
                fig.add_trace(
                    go.Scatter(
                        x=anom_df['timestamp'],
                        y=anom_df['z_resid'],
                        name='Z-score',
                        line=dict(color='steelblue', width=1),
                        mode='lines'
                    ),
                    row=2, col=1
                )
                
                # Add threshold lines
                fig.add_hline(y=3.0, line_dash="dash", line_color="red", row=2, col=1, 
                             annotation_text="Threshold (+3σ)")
                fig.add_hline(y=-3.0, line_dash="dash", line_color="red", row=2, col=1,
                             annotation_text="Threshold (-3σ)")
                fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1,
                             opacity=0.5)
                
                fig.update_xaxes(title_text="Timestamp", row=2, col=1)
                fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
                fig.update_yaxes(title_text="Z-score", row=2, col=1)
                
                fig.update_layout(
                    height=700,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details table
                if len(anomaly_points) > 0:
                    st.markdown("#### Detected Anomaly Details")
                    anomaly_display = anomaly_points[['timestamp', 'y_true', 'yhat', 'z_resid']].copy()
                    anomaly_display.columns = ['Timestamp', 'Actual Load', 'Forecast', 'Z-score']
                    st.dataframe(
                        anomaly_display.style.format({
                            'Actual Load': '{:.2f}',
                            'Forecast': '{:.2f}',
                            'Z-score': '{:.2f}'
                        }),
                        use_container_width=True
                    )

# SECTION 4: LIVE MONITORING

elif section == "🔄 Live Monitoring":
    st.header("🔄 Live Monitoring Simulation Results")
    
    st.markdown("""
    Simulated **3,500 hours** (146 days) of live data streaming with **periodic model retraining**.
    All models were refitted every **336 hours** (2 weeks) using an expanding window for fair comparison.
    """)
    
    if not countries:
        st.warning("Please select at least one country from the sidebar.")
    else:
        # Model selector
        model_choice = st.selectbox(
            "Select Model to View",
            options=['SARIMA', 'LSTM', 'GRU', 'RNN', 'All Models (Comparison)'],
            index=4
        )
        
        # Summary metrics for all models
        st.subheader("📊 3,500-Hour Simulation Summary")
        
        # Create summary table
        summary_data = []
        for model_name in ['SARIMA', 'LSTM', 'GRU', 'RNN']:
            for country in countries:
                model_data = live_data.get(model_name, {})
                live_df = model_data.get(country)
                if live_df is not None:
                    summary_data.append({
                        'Model': model_name,
                        'Country': country,
                        'Avg MAPE (%)': live_df['mape_rolling_24h'].mean(),
                        'Avg MASE': live_df['mase_rolling_24h'].mean(),
                        'Avg RMSE': live_df['rmse_rolling_24h'].mean(),
                        'Refits': int(live_df['refit_flag'].sum())
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Display as table
            cols = st.columns(len(countries))
            for idx, country in enumerate(countries):
                with cols[idx]:
                    st.markdown(f"### {country}")
                    country_summary = summary_df[summary_df['Country'] == country][['Model', 'Avg MAPE (%)', 'Avg MASE', 'Refits']]
                    st.dataframe(
                        country_summary.style.format({
                            'Avg MAPE (%)': '{:.2f}',
                            'Avg MASE': '{:.4f}',
                            'Refits': '{:.0f}'
                        }).highlight_min(subset=['Avg MAPE (%)'], color='lightgreen'),
                        use_container_width=True,
                        hide_index=True
                    )
        
        st.markdown("---")
        
        # Performance evolution chart
        st.subheader("📈 Performance Evolution Over Time")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            metric_choice = st.radio(
                "Select Metric",
                options=['MAPE (%)', 'MASE (scaled)', 'RMSE (MW)'],
                horizontal=True
            )
        with col2:
            smooth_data = st.checkbox("Smooth curves (4-hour avg)", value=False)
        
        metric_map = {
            'MAPE (%)': 'mape_rolling_24h',
            'MASE (scaled)': 'mase_rolling_24h',
            'RMSE (MW)': 'rmse_rolling_24h'
        }
        
        ylabel_map = {
            'MAPE (%)': 'Error (%)',
            'MASE (scaled)': 'Scaled Error',
            'RMSE (MW)': 'Error (MW)'
        }
        
        color_map = {
            'SARIMA': '#FF6B6B',
            'LSTM': '#4ECDC4',
            'GRU': '#95E1D3',
            'RNN': '#F38181'
        }
        
        if model_choice == 'All Models (Comparison)':
            # Show all models for selected countries in separate subplots
            for country in countries:
                st.markdown(f"#### {country}")
                fig = go.Figure()
                
                for model_name in ['SARIMA', 'LSTM', 'GRU', 'RNN']:
                    model_data = live_data.get(model_name, {})
                    live_df = model_data.get(country)
                    if live_df is not None:
                        y_data = live_df[metric_map[metric_choice]]
                        
                        # Apply smoothing if requested
                        if smooth_data:
                            y_data = y_data.rolling(window=4, center=True).mean()
                        
                        fig.add_trace(go.Scatter(
                            x=live_df['timestamp'],
                            y=y_data,
                            name=model_name,
                            mode='lines',
                            line=dict(width=2.5, color=color_map.get(model_name)),
                            hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
                        ))
                
                fig.update_layout(
                    xaxis=dict(
                        title="Time Period",
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        tickformat='%b %d'
                    ),
                    yaxis=dict(
                        title=ylabel_map[metric_choice],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(128,128,128,0.2)',
                        zeroline=True,
                        zerolinewidth=2,
                        zerolinecolor='rgba(128,128,128,0.3)'
                    ),
                    height=450,
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5
                    ),
                    plot_bgcolor='white',
                    margin=dict(l=60, r=40, t=40, b=60)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Show single model for all countries
            fig = go.Figure()
            
            model_data = live_data.get(model_choice, {})
            country_colors = {'AT': '#FF6B6B', 'BE': '#4ECDC4', 'BG': '#95E1D3'}
            
            for country in countries:
                live_df = model_data.get(country)
                if live_df is not None:
                    y_data = live_df[metric_map[metric_choice]]
                    
                    # Apply smoothing if requested
                    if smooth_data:
                        y_data = y_data.rolling(window=4, center=True).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=live_df['timestamp'],
                        y=y_data,
                        name=country,
                        mode='lines',
                        line=dict(width=2.5, color=country_colors.get(country)),
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Value: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Mark refits with vertical lines instead of scatter points
                    refit_points = live_df[live_df['refit_flag'] == 1]
                    if len(refit_points) > 0 and not smooth_data:
                        for idx, refit_time in enumerate(refit_points['timestamp']):
                            fig.add_vline(
                                x=refit_time,
                                line_dash="dot",
                                line_color=country_colors.get(country),
                                opacity=0.4,
                                annotation_text="Refit" if idx == 0 else "",
                                annotation_position="top"
                            )
            
            fig.update_layout(
                xaxis=dict(
                    title="Time Period",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%b %d'
                ),
                yaxis=dict(
                    title=ylabel_map[metric_choice],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='rgba(128,128,128,0.3)'
                ),
                height=500,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                plot_bgcolor='white',
                margin=dict(l=60, r=40, t=40, b=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model age chart (for single model view only)
        if model_choice != 'All Models (Comparison)':
            st.markdown("---")
            st.subheader("🔧 Model Freshness (Days Since Last Refit)")
            
            fig2 = go.Figure()
            
            model_data = live_data.get(model_choice, {})
            country_colors = {'AT': '#FF6B6B', 'BE': '#4ECDC4', 'BG': '#95E1D3'}
            
            for country in countries:
                live_df = model_data.get(country)
                if live_df is not None and 'model_age_hours' in live_df.columns:
                    # Convert hours to days for better readability
                    model_age_days = live_df['model_age_hours'] / 24
                    
                    fig2.add_trace(go.Scatter(
                        x=live_df['timestamp'],
                        y=model_age_days,
                        name=country,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(width=2, color=country_colors.get(country)),
                        fillcolor=f"rgba{tuple(list(int(country_colors.get(country)[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}",
                        hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Age: %{y:.1f} days<extra></extra>'
                    ))
            
            # Add refit threshold line (14 days = 336 hours)
            fig2.add_hline(
                y=14,
                line_dash="dash",
                line_color="rgba(255,0,0,0.5)",
                line_width=2,
                annotation_text="Refit Threshold (14 days)",
                annotation_position="right"
            )
            
            fig2.update_layout(
                xaxis=dict(
                    title="Time Period",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    tickformat='%b %d'
                ),
                yaxis=dict(
                    title="Days",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    range=[0, 15]
                ),
                hovermode='x unified',
                height=350,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                plot_bgcolor='white',
                margin=dict(l=60, r=40, t=40, b=60)
            )
            
            st.plotly_chart(fig2, use_container_width=True)

# SECTION 5: MODEL COMPARISON

elif section == "🏆 Model Comparison":
    st.header("🏆 Model Performance Comparison")
    
    # Metrics comparison table
    st.subheader("📊 All Models - Test Set Metrics")
    
    if sarima_metrics is not None and lstm_metrics is not None and gru_metrics is not None and rnn_metrics is not None:
        # Combine metrics
        comparison = pd.DataFrame()
        
        for country in ['AT', 'BE', 'BG']:
            if country in sarima_metrics.index:
                sarima_row = sarima_metrics.loc[country].copy()
                sarima_row.name = f"{country} - SARIMA"
                comparison = pd.concat([comparison, pd.DataFrame([sarima_row])])
            
            if country in lstm_metrics.index:
                lstm_row = lstm_metrics.loc[country].copy()
                lstm_row.name = f"{country} - LSTM"
                if 'MAPE_%' in lstm_row.index:
                    lstm_row['MAPE'] = lstm_row['MAPE_%']
                if 'sMAPE_%' in lstm_row.index:
                    lstm_row['sMAPE'] = lstm_row['sMAPE_%']
                if 'RMSE_MW' in lstm_row.index:
                    lstm_row['RMSE'] = lstm_row['RMSE_MW']
                comparison = pd.concat([comparison, pd.DataFrame([lstm_row])])
            
            if country in gru_metrics.index:
                gru_row = gru_metrics.loc[country].copy()
                gru_row.name = f"{country} - GRU"
                if 'MAPE_%' in gru_row.index:
                    gru_row['MAPE'] = gru_row['MAPE_%']
                if 'sMAPE_%' in gru_row.index:
                    gru_row['sMAPE'] = gru_row['sMAPE_%']
                if 'RMSE_MW' in gru_row.index:
                    gru_row['RMSE'] = gru_row['RMSE_MW']
                comparison = pd.concat([comparison, pd.DataFrame([gru_row])])
            
            if country in rnn_metrics.index:
                rnn_row = rnn_metrics.loc[country].copy()
                rnn_row.name = f"{country} - Vanilla RNN"
                if 'MAPE_%' in rnn_row.index:
                    rnn_row['MAPE'] = rnn_row['MAPE_%']
                if 'sMAPE_%' in rnn_row.index:
                    rnn_row['sMAPE'] = rnn_row['sMAPE_%']
                if 'RMSE_MW' in rnn_row.index:
                    rnn_row['RMSE'] = rnn_row['RMSE_MW']
                comparison = pd.concat([comparison, pd.DataFrame([rnn_row])])
        
        # Display table
        display_cols = ['MASE', 'MAPE', 'sMAPE', 'RMSE', 'MSE']
        available_cols = [col for col in display_cols if col in comparison.columns]
        
        st.dataframe(
            comparison[available_cols].style.format({
                'MASE': '{:.4f}',
                'MAPE': '{:.2f}',
                'sMAPE': '{:.2f}',
                'RMSE': '{:.2f}',
                'MSE': '{:.2f}'
            }).highlight_min(axis=0, color='lightgreen'),
            use_container_width=True
        )
    
    # Visual comparison
    st.markdown("---")
    st.subheader("📊 Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MAPE comparison
        if sarima_metrics is not None and lstm_metrics is not None and gru_metrics is not None and rnn_metrics is not None:
            mape_data = []
            for country in ['AT', 'BE', 'BG']:
                if country in sarima_metrics.index:
                    mape_data.append({
                        'Country': country,
                        'Model': 'SARIMA',
                        'MAPE': sarima_metrics.loc[country, 'MAPE']
                    })
                if country in lstm_metrics.index:
                    mape_val = lstm_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in lstm_metrics.columns else lstm_metrics.loc[country, 'MAPE']
                    mape_data.append({
                        'Country': country,
                        'Model': 'LSTM',
                        'MAPE': mape_val
                    })
                if country in gru_metrics.index:
                    mape_val = gru_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in gru_metrics.columns else gru_metrics.loc[country, 'MAPE']
                    mape_data.append({
                        'Country': country,
                        'Model': 'GRU',
                        'MAPE': mape_val
                    })
                if country in rnn_metrics.index:
                    mape_val = rnn_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in rnn_metrics.columns else rnn_metrics.loc[country, 'MAPE']
                    mape_data.append({
                        'Country': country,
                        'Model': 'Vanilla RNN',
                        'MAPE': mape_val
                    })
            
            mape_df = pd.DataFrame(mape_data)
            fig = px.bar(
                mape_df,
                x='Country',
                y='MAPE',
                color='Model',
                barmode='group',
                title='MAPE Comparison by Country',
                labels={'MAPE': 'MAPE (%)'},
                color_discrete_map={'SARIMA': 'blue', 'LSTM': 'green', 'GRU': 'orange', 'Vanilla RNN': 'red'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MASE comparison
        if sarima_metrics is not None and lstm_metrics is not None and gru_metrics is not None and rnn_metrics is not None:
            mase_data = []
            for country in ['AT', 'BE', 'BG']:
                if country in sarima_metrics.index:
                    mase_data.append({
                        'Country': country,
                        'Model': 'SARIMA',
                        'MASE': sarima_metrics.loc[country, 'MASE']
                    })
                if country in lstm_metrics.index:
                    mase_data.append({
                        'Country': country,
                        'Model': 'LSTM',
                        'MASE': lstm_metrics.loc[country, 'MASE']
                    })
                if country in gru_metrics.index:
                    mase_data.append({
                        'Country': country,
                        'Model': 'GRU',
                        'MASE': gru_metrics.loc[country, 'MASE']
                    })
                if country in rnn_metrics.index:
                    mase_data.append({
                        'Country': country,
                        'Model': 'Vanilla RNN',
                        'MASE': rnn_metrics.loc[country, 'MASE']
                    })
            
            mase_df = pd.DataFrame(mase_data)
            fig = px.bar(
                mase_df,
                x='Country',
                y='MASE',
                color='Model',
                barmode='group',
                title='MASE Comparison by Country',
                labels={'MASE': 'MASE (lower is better)'},
                color_discrete_map={'SARIMA': 'blue', 'LSTM': 'green', 'GRU': 'orange', 'Vanilla RNN': 'red'}
            )
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                         annotation_text="Baseline (MASE=1.0)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 Model Recommendations")
    
    if sarima_metrics is not None and lstm_metrics is not None and gru_metrics is not None and rnn_metrics is not None:
        # Calculate best model per country based on MAPE
        recommendations = {}
        
        for country in ['AT', 'BE', 'BG']:
            country_results = {}
            
            if country in sarima_metrics.index:
                country_results['SARIMA'] = {
                    'MAPE': sarima_metrics.loc[country, 'MAPE'],
                    'MASE': sarima_metrics.loc[country, 'MASE']
                }
            
            if country in lstm_metrics.index:
                mape = lstm_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in lstm_metrics.columns else lstm_metrics.loc[country, 'MAPE']
                country_results['LSTM'] = {
                    'MAPE': mape,
                    'MASE': lstm_metrics.loc[country, 'MASE']
                }
            
            if country in gru_metrics.index:
                mape = gru_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in gru_metrics.columns else gru_metrics.loc[country, 'MAPE']
                country_results['GRU'] = {
                    'MAPE': mape,
                    'MASE': gru_metrics.loc[country, 'MASE']
                }
            
            if country in rnn_metrics.index:
                mape = rnn_metrics.loc[country, 'MAPE_%'] if 'MAPE_%' in rnn_metrics.columns else rnn_metrics.loc[country, 'MAPE']
                country_results['Vanilla RNN'] = {
                    'MAPE': mape,
                    'MASE': rnn_metrics.loc[country, 'MASE']
                }
            
            # Find best model (lowest MAPE)
            if country_results:
                best_model = min(country_results.items(), key=lambda x: x[1]['MAPE'])
                recommendations[country] = {
                    'best_model': best_model[0],
                    'best_mape': best_model[1]['MAPE'],
                    'best_mase': best_model[1]['MASE'],
                    'all_results': country_results
                }
        
        # Display per-country recommendations
        col1, col2, col3 = st.columns(3)
        
        country_names = {'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria'}
        
        for col, country in zip([col1, col2, col3], ['AT', 'BE', 'BG']):
            with col:
                if country in recommendations:
                    rec = recommendations[country]
                    st.markdown(f"**{country_names[country]} ({country})**")
                    st.success(f"✅ **{rec['best_model']}** - Best performance")
                    st.metric("MAPE", f"{rec['best_mape']:.2f}%", delta=None)
                    st.metric("MASE", f"{rec['best_mase']:.4f}", delta=None)
                    
                    # Show ranking
                    sorted_models = sorted(rec['all_results'].items(), key=lambda x: x[1]['MAPE'])
                    st.caption("**Ranking (by MAPE):**")
                    for i, (model, metrics) in enumerate(sorted_models, 1):
                        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                        st.caption(f"{emoji} {i}. {model}: {metrics['MAPE']:.2f}%")
        
        # Overall best model recommendation
        st.markdown("---")
        st.subheader("🏅 Overall Best Model")
        
        # Calculate average MAPE across all countries for each model
        model_averages = {}
        
        for model in ['SARIMA', 'LSTM', 'GRU', 'Vanilla RNN']:
            mapes = []
            for country in ['AT', 'BE', 'BG']:
                if country in recommendations and model in recommendations[country]['all_results']:
                    mapes.append(recommendations[country]['all_results'][model]['MAPE'])
            
            if mapes:
                model_averages[model] = {
                    'avg_mape': sum(mapes) / len(mapes),
                    'countries': len(mapes)
                }
        
        if model_averages:
            best_overall = min(model_averages.items(), key=lambda x: x[1]['avg_mape'])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### 🏆 **{best_overall[0]}**")
                st.metric("Average MAPE", f"{best_overall[1]['avg_mape']:.2f}%")
                st.caption(f"Evaluated across {best_overall[1]['countries']} countries")
            
            with col2:
                st.markdown("**Summary:**")
                sorted_overall = sorted(model_averages.items(), key=lambda x: x[1]['avg_mape'])
                
                for i, (model, metrics) in enumerate(sorted_overall, 1):
                    if i == 1:
                        st.success(f"🥇 **{model}**: {metrics['avg_mape']:.2f}% MAPE (Best)")
                    elif i == 2:
                        st.info(f"🥈 {model}: {metrics['avg_mape']:.2f}% MAPE")
                    elif i == 3:
                        st.info(f"🥉 {model}: {metrics['avg_mape']:.2f}% MAPE")
                    else:
                        st.warning(f"📊 {model}: {metrics['avg_mape']:.2f}% MAPE")
                
                # Key insights
                st.markdown("---")
                st.markdown("**Key Insights:**")
                winner_count = sum(1 for c in recommendations.values() if c['best_model'] == best_overall[0])
                st.write(f"• {best_overall[0]} achieves the best performance in **{winner_count} out of 3 countries**")
                st.write(f"• Average improvement over baseline: **{((model_averages.get('SARIMA', {}).get('avg_mape', 0) - best_overall[1]['avg_mape']) / model_averages.get('SARIMA', {}).get('avg_mape', 1) * 100):.1f}%**")
        st.write("LSTM: 4.50% MAPE, 0.755 MASE")

# FOOTER

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 2rem 0;'>
        <p><strong>OPSD PowerDesk Dashboard</strong></p>
        <p>Advanced Time Series Analysis Project | November 2025</p>
        <p>Data Source: Open Power System Data (OPSD) | Countries: Austria, Belgium, Bulgaria</p>
    </div>
    """,
    unsafe_allow_html=True
)
