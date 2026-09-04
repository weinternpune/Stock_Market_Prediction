"""
app.py
------
Interactive Streamlit Application for Nifty 500 Stock Market Prediction Pipeline.
Implements Phases 48 through 56 (Steps 54 through 60) of the Project Roadmap.

Pages:
1. 📌 Executive Overview & Objectives
2. 📊 Historical Market Explorer (NSE vs. BSE)
3. 🔍 Quantitative EDA & Volatility Clustering
4. 📈 Technical Indicators & Feature Importances
5. 🏆 Model Performance Benchmark Scorecard
6. 🔮 Actual vs. Predicted Backtesting Visualizer
7. 🚀 Future Horizon Forecaster (T+1 to T+30)
8. 🖥️ Executive Presentation Deck & Slides
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Setup paths
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

st.set_page_config(
    page_title="Nifty 500 Market Prediction & Quantitative Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_datasets():
    nse_df = pd.read_csv(PROCESSED_DIR / "NIFTY500_clean.csv")
    bse_df = pd.read_csv(PROCESSED_DIR / "BSE500_clean.csv")
    preds_df = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    forecast_df = pd.read_csv(PROCESSED_DIR / "future_forecast_t30.csv")
    
    nse_df['Date'] = pd.to_datetime(nse_df['Date'])
    bse_df['Date'] = pd.to_datetime(bse_df['Date'])
    preds_df['Date'] = pd.to_datetime(preds_df['Date'])
    forecast_df['Forecast_Date'] = pd.to_datetime(forecast_df['Forecast_Date'])
    
    with open(MODELS_DIR / "metrics_summary.json", "r") as f:
        metrics_summary = json.load(f)
        
    feat_imp = pd.read_csv(MODELS_DIR / "feature_importances.csv")
    outliers_df = pd.read_csv(MODELS_DIR / "outlier_investigation.csv")
    
    return nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df


try:
    nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df = load_datasets()
except Exception as e:
    st.error(f"Error loading project artifacts: {e}. Please run `python src/pipeline.py` first.")
    st.stop()


# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/Nifty_500_Logo.svg", width=160)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Dashboard Section:",
    [
        "📌 Executive Overview & Objectives",
        "📊 Historical Market Explorer",
        "🔍 Quantitative EDA & Volatility",
        "📈 Technical Indicators & Features",
        "🏆 Model Performance Scorecard",
        "🔮 Actual vs. Predicted Backtesting",
        "🚀 Future Horizon Forecaster",
        "🖥️ Executive Presentation Deck"
    ]
)

# Global Disclaimer in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**⚠️ Regulatory Disclaimer:**\n"
    "This system is developed strictly for educational, academic, and analytical skill-building. "
    "None of the models, forecasts, or findings constitute financial advice or investment recommendations."
)


# ==============================================================================
# PAGE 1: Executive Overview
# ==============================================================================
if page == "📌 Executive Overview & Objectives":
    st.markdown("<div class='main-title'>📈 Nifty 500 Stock Market Prediction Pipeline</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Data Analytics Intern Capstone Project · Product Requirements Document (PRD v1.1)</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analysis Period", "5 Years", "1,240 Sessions")
    col2.metric("Date Range", "Sep 2021 – Aug 2026", "Zero Gaps")
    col3.metric("Modeling Target", "P(t+1) Close", "Zero Lookahead")
    col4.metric("Best Level RMSE", "209.33 Pts", "Naive Baseline")
    
    st.markdown("""
    ### 🎯 Project Objectives & Problem Framing
    This project builds a quantitative time-series forecasting pipeline for the **Nifty 500 index**, representing ~96% of the free-float market capitalization of the National Stock Exchange of India (NSE).
    
    The pipeline strictly adheres to the 60-step roadmap across 15 phases:
    1. **Primary Data Source:** 5-year official historical archive of the NSE Nifty 500 index (`Date, Open, High, Low, Close`).
    2. **Secondary Cross-Exchange Proxy:** BSE 500 index (`BSE-500.BO`) used strictly as a secondary benchmark for calendar reconciliation, volatility cross-checking, and macro validation (Price correlation: **0.99989**). No price averaging or merging.
    3. **Multi-Model Family Benchmarking:**
       - **Naive Baseline:** Persistence Random Walk ($P_{t+1} = P_t$) & 5-Day SMA.
       - **Statistical Model:** Pure walk-forward ARIMA(1, 1, 1) rolling one-step ahead with zero lookahead bias.
       - **Classical Machine Learning:** Random Forest and XGBoost Regressors with 5-fold expanding window cross-validation.
       - **Deep Learning:** PyTorch Stacked Long Short-Term Memory (LSTM) Neural Network.
    4. **Scientific Rigor & Transparent Assessment:** Honest, documented reporting on why financial asset prices behave as near-martingales and why beating the persistence benchmark on level-price RMSE is structurally difficult.
    """)
    
    st.markdown("<div class='disclaimer-box'><b>PRD Educational Disclaimer:</b> This dashboard was constructed for analytical exploration and machine learning evaluation under noisy financial time series. Predictions should not be used as the basis for actual capital allocation or live trading.</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: Historical Market Explorer
# ==============================================================================
elif page == "📊 Historical Market Explorer":
    st.markdown("<div class='main-title'>📊 Historical Market Data & Exchange Proxy Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Explore 5 years of daily trading history (September 1, 2021 to August 31, 2026)</div>", unsafe_allow_html=True)
    
    # Date filter
    min_date = nse_df['Date'].min().date()
    max_date = nse_df['Date'].max().date()
    
    c1, c2 = st.columns(2)
    start_d = c1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_d = c2.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    mask = (nse_df['Date'].dt.date >= start_d) & (nse_df['Date'].dt.date <= end_d)
    filt_nse = nse_df[mask].copy()
    filt_bse = bse_df[mask].copy()
    
    tab1, tab2 = st.tabs(["🕯️ Nifty 500 Candlestick & Volume", "🔄 NSE vs. BSE Cross-Market Co-Movement"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=filt_nse['Date'],
            open=filt_nse['Open'],
            high=filt_nse['High'],
            low=filt_nse['Low'],
            close=filt_nse['Close'],
            name="NSE Nifty 500"
        ))
        fig.update_layout(
            title=f"Nifty 500 OHLC Price Series ({start_d} to {end_d})",
            xaxis_title="Date",
            yaxis_title="Index Level (Points)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(filt_nse.tail(10), use_container_width=True)
        
    with tab2:
        # Normalized co-movement plot (rebased to 100)
        norm_nse = (filt_nse['Close'] / filt_nse['Close'].iloc[0]) * 100
        norm_bse = (filt_bse['Close'] / filt_bse['Close'].iloc[0]) * 100
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=filt_nse['Date'], y=norm_nse, name="NSE Nifty 500 (Rebased to 100)", line=dict(color='#2563EB', width=2)))
        fig2.add_trace(go.Scatter(x=filt_bse['Date'], y=norm_bse, name="BSE 500 Proxy (Rebased to 100)", line=dict(color='#EA580C', width=2, dash='dot')))
        fig2.update_layout(
            title="Normalized Broad-Market Co-Movement (Base = 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Index Level",
            template="plotly_white",
            height=480
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.info("💡 **Cross-Market Insight:** 5-year price correlation is **0.99989** and return correlation is **0.99930**. This establishes BSE 500 as an empirical proxy while maintaining independent price series per PRD Phase 4.")


# ==============================================================================
# PAGE 3: Quantitative EDA & Volatility
# ==============================================================================
elif page == "🔍 Quantitative EDA & Volatility":
    st.markdown("<div class='main-title'>🔍 Quantitative EDA & Volatility Clustering</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Statistical properties, return distribution, and macroeconomic regime shocks</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Return distribution
        returns = nse_df['Daily_Return'].dropna() * 100
        fig_hist = px.histogram(
            returns, nbins=80, 
            title="Nifty 500 Daily Return Distribution (%)",
            labels={'value': 'Daily Return (%)'},
            color_discrete_sequence=['#3B82F6']
        )
        fig_hist.update_layout(template="plotly_white", showlegend=False, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("""
        **Statistical Summary:**
        - **Mean Daily Return:** +0.043% (Annualized: ~11.26%)
        - **Skewness:** -0.678 (Asymmetric downside tails)
        - **Excess Kurtosis:** 4.585 (Leptokurtic fat tails)
        - **Jarque-Bera Test:** $p < 10^{-250}$ (Strict non-normality)
        """)
        
    with col2:
        # Volatility clustering
        nse_df['Rolling_Vol_20'] = returns.rolling(20).std() * np.sqrt(250)
        fig_vol = px.line(
            nse_df, x='Date', y='Rolling_Vol_20',
            title="20-Day Rolling Annualized Volatility (%) · Volatility Clustering",
            labels={'Rolling_Vol_20': 'Annualized Vol (%)'},
            color_discrete_sequence=['#EF4444']
        )
        fig_vol.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
        
        st.markdown("""
        **Volatility Insights (Mandelbrot Clustering):**
        - High-volatility periods group together (e.g. 2022 Ukraine War, 2024 General Elections).
        - Calm periods exhibit persistence.
        """)
        
    st.markdown("### ⚠️ Historical Macroeconomic Outlier Event Audit (Phase 10)")
    st.dataframe(outliers_df, use_container_width=True)


# ==============================================================================
# PAGE 4: Technical Indicators & Feature Importances
# ==============================================================================
elif page == "📈 Technical Indicators & Features":
    st.markdown("<div class='main-title'>📈 Technical Indicators & Feature Importance Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>15+ Engineered Technical Features: Moving Averages, RSI, MACD, Bollinger Bands</div>", unsafe_allow_html=True)
    
    features_path = FEATURES_DIR / "nifty_500_features.csv"
    if features_path.exists():
        feat_df = pd.read_csv(features_path)
        feat_df['Date'] = pd.to_datetime(feat_df['Date'])
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # RSI & MACD plot
            fig_ind = go.Figure()
            fig_ind.add_trace(go.Scatter(x=feat_df['Date'].tail(300), y=feat_df['RSI_14'].tail(300), name="RSI (14-Day)", line=dict(color='#8B5CF6')))
            fig_ind.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_ind.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig_ind.update_layout(title="14-Day Relative Strength Index (RSI) - Recent 300 Sessions", template="plotly_white", height=350)
            st.plotly_chart(fig_ind, use_container_width=True)
            
            fig_bb = go.Figure()
            sub_bb = feat_df.tail(200)
            fig_bb.add_trace(go.Scatter(x=sub_bb['Date'], y=sub_bb['Close'], name="Close Price", line=dict(color='black')))
            fig_bb.add_trace(go.Scatter(x=sub_bb['Date'], y=sub_bb['BB_Upper'], name="Upper Band", line=dict(color='rgba(239, 68, 68, 0.5)')))
            fig_bb.add_trace(go.Scatter(x=sub_bb['Date'], y=sub_bb['BB_Lower'], name="Lower Band", line=dict(color='rgba(16, 185, 129, 0.5)'), fill='tonexty', fillcolor='rgba(243, 244, 246, 0.5)'))
            fig_bb.update_layout(title="Bollinger Bands (20-Day, ±2σ) - Recent 200 Sessions", template="plotly_white", height=380)
            st.plotly_chart(fig_bb, use_container_width=True)
            
        with c2:
            st.markdown("### Top Feature Importances (Random Forest & XGBoost)")
            st.dataframe(feat_imp.head(15), use_container_width=True)
            
            fig_imp = px.bar(
                feat_imp.head(10), 
                x='RF_Importance', y='Feature', orientation='h',
                title="Top 10 Random Forest Features",
                color='RF_Importance', color_continuous_scale='Blues'
            )
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, template="plotly_white", height=400)
            st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# PAGE 5: Model Performance Scorecard
# ==============================================================================
elif page == "🏆 Model Performance Scorecard":
    st.markdown("<div class='main-title'>🏆 Model Performance Benchmark Scorecard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Evaluated strictly on the held-out out-of-sample test split (Oct 2025 to Aug 2026, 208 sessions)</div>", unsafe_allow_html=True)
    
    scorecard_df = pd.DataFrame(metrics_summary)
    display_cols = ["Model", "RMSE", "MAE", "MAPE_Pct", "Directional_Accuracy", "vs_Naive_RMSE"]
    
    st.dataframe(
        scorecard_df[display_cols].style.highlight_min(subset=["RMSE", "MAE", "MAPE_Pct"], color="#DCFCE7"),
        use_container_width=True
    )
    
    c1, c2 = st.columns(2)
    with c1:
        fig_rmse = px.bar(
            scorecard_df, x='Model', y='RMSE', color='Model',
            title="Out-of-Sample Level RMSE (Points) · Lower is Better",
            text='RMSE', color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_rmse.update_layout(template="plotly_white", showlegend=False, height=400)
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    with c2:
        # Directional Hit Rate Bar Chart (excluding Naive where hit rate is N/A)
        dir_df = scorecard_df[scorecard_df['raw_dir_acc'].notna()].copy()
        fig_dir = px.bar(
            dir_df, x='Model', y='raw_dir_acc', color='Model',
            title="Directional Hit Rate (%) · Benchmark Threshold = 50%",
            text='raw_dir_acc', color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_dir.add_hline(y=50.0, line_dash="dash", line_color="gray", annotation_text="Random Guess (50%)")
        fig_dir.update_layout(template="plotly_white", showlegend=False, yaxis_range=[40, 65], height=400)
        st.plotly_chart(fig_dir, use_container_width=True)
        
    st.markdown("""
    ---
    ### 🔬 Deep Empirical & Theoretical Findings (PRD Section 12 & Phase 44)
    
    1. **Why the Naive Persistence Model Achieved the Lowest Level-Price RMSE (209.33):**
       Under the **Martingale Property of Asset Prices**:
       $$\\mathbb{E}[P_{t+1} \\mid \\mathcal{F}_t] \\approx P_t$$
       Today's closing price $P_t$ is the minimum-variance quadratic unbiased estimator of tomorrow's price level $P_{t+1}$ in an informationally efficient market. Any predictive model forecasting non-zero price delta incurs quadratic variance penalties whenever the market moves sideways.
       
    2. **Why Directional Accuracy is `N/A` for Naive Persistence:**
       The naive persistence model predicts zero price delta ($P_{t+1} - P_t = 0$), rather than an directional signal. Labeling it as 0% or opposite is mathematically improper.
       
    3. **Deep Learning (LSTM) Tradeoff:**
       While the PyTorch LSTM captured sequential trend momentum (**52.88% directional hit rate**), its price-level RMSE (560.27) was higher due to sliding sequence drift under non-stationary price distributions.
    """)


# ==============================================================================
# PAGE 6: Actual vs. Predicted Backtesting
# ==============================================================================
elif page == "🔮 Actual vs. Predicted Backtesting":
    st.markdown("<div class='main-title'>🔮 Actual vs. Predicted Backtesting Visualizer</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Inspect model tracking accuracy and prediction residuals across the test period (208 sessions)</div>", unsafe_allow_html=True)
    
    models_selected = st.multiselect(
        "Select Models to Display Overlay:",
        options=["Naive_Persistence", "Moving_Average_5D", "ARIMA_1_1_1", "Random_Forest", "XGBoost", "LSTM"],
        default=["Naive_Persistence", "Random_Forest", "LSTM"]
    )
    
    fig_back = go.Figure()
    fig_back.add_trace(go.Scatter(
        x=preds_df['Date'], y=preds_df['Actual_Target'],
        name="Actual Close (P_{t+1})",
        line=dict(color='black', width=3)
    ))
    
    palette = {
        "Naive_Persistence": "#3B82F6",
        "Moving_Average_5D": "#10B981",
        "ARIMA_1_1_1": "#F59E0B",
        "Random_Forest": "#8B5CF6",
        "XGBoost": "#EC4899",
        "LSTM": "#EF4444"
    }
    
    for m in models_selected:
        fig_back.add_trace(go.Scatter(
            x=preds_df['Date'], y=preds_df[m],
            name=m.replace("_", " "),
            line=dict(color=palette.get(m, "#6B7280"), width=1.8, dash='dot')
        ))
        
    fig_back.update_layout(
        title="Out-of-Sample Backtesting: Actual vs. Model Predictions",
        xaxis_title="Date",
        yaxis_title="Index Level (Points)",
        template="plotly_white",
        height=520
    )
    st.plotly_chart(fig_back, use_container_width=True)
    
    # Residual error plot
    if models_selected:
        st.markdown("### Prediction Residuals ($P_{\\text{actual}} - P_{\\text{predicted}}$)")
        fig_res = go.Figure()
        for m in models_selected:
            residuals = preds_df['Actual_Target'] - preds_df[m]
            fig_res.add_trace(go.Scatter(
                x=preds_df['Date'], y=residuals,
                name=f"{m} Residual",
                line=dict(color=palette.get(m, "#6B7280"), width=1)
            ))
        fig_res.add_hline(y=0, line_dash="solid", line_color="black")
        fig_res.update_layout(title="Prediction Residuals Over Time", template="plotly_white", height=350)
        st.plotly_chart(fig_res, use_container_width=True)


# ==============================================================================
# PAGE 7: Future Horizon Forecaster
# ==============================================================================
elif page == "🚀 Future Horizon Forecaster":
    st.markdown("<div class='main-title'>🚀 Future Horizon Forecaster (T+1 to T+30)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Recursive forward price projections with expanding volatility-based 95% confidence bands</div>", unsafe_allow_html=True)
    
    horizon_days = st.slider("Select Forecast Horizon (Trading Sessions Ahead):", min_value=1, max_value=30, value=15)
    
    sub_forecast = forecast_df.head(horizon_days)
    
    target_row = sub_forecast.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Base Closing Price (2026-08-31)", f"₹{sub_forecast['Naive_Persistence'].iloc[0]:,.2f}")
    col2.metric(f"Random Forest ({target_row['Step']} Projection)", f"₹{target_row['Random_Forest']:,.2f}")
    col3.metric(f"XGBoost ({target_row['Step']} Projection)", f"₹{target_row['XGBoost']:,.2f}")
    
    fig_fwd = go.Figure()
    # History context (last 60 sessions)
    hist_ctx = nse_df.tail(60)
    fig_fwd.add_trace(go.Scatter(
        x=hist_ctx['Date'], y=hist_ctx['Close'],
        name="Historical Close (Actual)",
        line=dict(color='black', width=2.5)
    ))
    
    # Confidence interval band
    fig_fwd.add_trace(go.Scatter(
        x=sub_forecast['Forecast_Date'], y=sub_forecast['Upper_95_CI'],
        line=dict(color='rgba(147, 197, 253, 0.4)'),
        showlegend=False
    ))
    fig_fwd.add_trace(go.Scatter(
        x=sub_forecast['Forecast_Date'], y=sub_forecast['Lower_95_CI'],
        line=dict(color='rgba(147, 197, 253, 0.4)'),
        fill='tonexty',
        fillcolor='rgba(219, 234, 254, 0.5)',
        name="95% Forecast Confidence Band"
    ))
    
    fig_fwd.add_trace(go.Scatter(
        x=sub_forecast['Forecast_Date'], y=sub_forecast['Random_Forest'],
        name="Random Forest Forecast",
        line=dict(color='#8B5CF6', width=2.5)
    ))
    fig_fwd.add_trace(go.Scatter(
        x=sub_forecast['Forecast_Date'], y=sub_forecast['XGBoost'],
        name="XGBoost Forecast",
        line=dict(color='#EC4899', width=2.5, dash='dash')
    ))
    fig_fwd.add_trace(go.Scatter(
        x=sub_forecast['Forecast_Date'], y=sub_forecast['Naive_Persistence'],
        name="Naive Persistence Baseline",
        line=dict(color='#3B82F6', width=1.5, dash='dot')
    ))
    
    fig_fwd.update_layout(
        title=f"Nifty 500 Forward Price Rollout ({sub_forecast['Step'].iloc[0]} to {sub_forecast['Step'].iloc[-1]})",
        xaxis_title="Date",
        yaxis_title="Index Level (Points)",
        template="plotly_white",
        height=520
    )
    st.plotly_chart(fig_fwd, use_container_width=True)
    
    st.markdown("### Forward Forecast Data Table")
    st.dataframe(sub_forecast, use_container_width=True)


# ==============================================================================
# PAGE 8: Presentation Deck & Slides
# ==============================================================================
elif page == "🖥️ Executive Presentation Deck":
    st.markdown("<div class='main-title'>🖥️ Executive Presentation Deck</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>15-Slide Presentation Deck Aligned with PRD Phase 60</div>", unsafe_allow_html=True)
    
    slides = [
        ("Slide 1: Title", "Nifty 500 Stock Market Prediction Pipeline & Analytics Dashboard", "Data Analytics Intern Project · PRD v1.1 · September 2026"),
        ("Slide 2: Problem Statement", "Forecasting Broad-Market Indices", "Evaluating whether statistical, classical ML, or deep learning models can consistently forecast index price movements."),
        ("Slide 3: Project Objective", "End-to-End Quantitative Rigor", "Predict next-day closing price P(t+1) using 5 years of historical data (1,240 trading sessions) without lookahead bias."),
        ("Slide 4: Data Sources & Scope", "Authoritative NSE & Proxy BSE", "NSE Nifty 500 (Primary, 1,240 rows) + BSE 500 (Secondary Proxy, 1,240 rows). Correlation = 0.99989."),
        ("Slide 5: Data Cleaning & Outliers", "Exchange Trading Calendar Integrity", "0.00% missing data post-cleaning. Extreme market shocks (Ukraine War, Election Day) audited and retained to avoid censorship bias."),
        ("Slide 6: Quantitative EDA Findings", "Stylized Facts of Financial Returns", "Non-normal leptokurtic distribution (Jarque-Bera p < 1e-250), heavy downside skewness, and Mandelbrot volatility clustering."),
        ("Slide 7: Feature Engineering", "15+ Technical & Volatility Indicators", "Moving averages (SMA 10, 20, 50, 200; EMA 12, 26), RSI 14, MACD, Bollinger Bands, and return lags with 200-day warm-up cutoff."),
        ("Slide 8: Model Architectures", "Multi-Family Benchmark Setup", "Naive Persistence, 5-Day SMA, walk-forward ARIMA(1,1,1), Random Forest, XGBoost, and PyTorch Stacked LSTM."),
        ("Slide 9: Out-of-Sample Scorecard", "Master Evaluation Table (208 Test Sessions)", "Naive Persistence: RMSE 209.33 (Best) | Random Forest: RMSE 229.66 | ARIMA: RMSE 291.08 | LSTM: RMSE 560.27."),
        ("Slide 10: Actual vs. Predicted Backtest", "Tracking Dynamics Across Market Regimes", "Models track overall trajectory but incur quadratic variance penalties on sideways days compared to persistence."),
        ("Slide 11: Best Model Selection", "Model Selection & Efficiency Tradeoff", "Naive Persistence achieves lowest RMSE under the Martingale property. Random Forest is the strongest feature-driven model."),
        ("Slide 12: Streamlit Dashboard", "Interactive Analytics & Forward Forecaster", "8-page dashboard offering interactive data filtering, backtesting visualizers, and recursive forward projections."),
        ("Slide 13: Key Scientific Findings", "The Martingale Property & Efficient Markets", "Asset prices behave as near-martingales: E[P(t+1)|F_t] = P(t). Level-price persistence is structurally optimal for RMSE."),
        ("Slide 14: Project Limitations", "Financial & Data Constraints", "No intraday granularity, absence of real-time macroeconomic news/sentiment, and non-stationarity drift in raw price levels."),
        ("Slide 15: Conclusion & Recommendations", "Strategic Next Steps", "Quantitative finance pipelines should predict stationary log returns rather than non-stationary raw price levels.")
    ]
    
    slide_num = st.selectbox("Select Slide:", range(1, 16), format_func=lambda x: f"Slide {x}: {slides[x-1][1]}")
    
    curr = slides[slide_num - 1]
    st.markdown(f"""
    <div style='background-color: #F8FAFC; border: 2px solid #3B82F6; border-radius: 12px; padding: 30px; margin-top: 20px;'>
        <h3 style='color: #1E3A8A; margin-top: 0;'>{curr[0]}</h3>
        <h2 style='color: #0F172A;'>{curr[1]}</h2>
        <hr style='border: 1px solid #E2E8F0;'>
        <p style='font-size: 1.15rem; color: #334155; line-height: 1.6;'>{curr[2]}</p>
    </div>
    """, unsafe_allow_html=True)
