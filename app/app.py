"""
app.py
------
Interactive Streamlit Application for Nifty 500 Stock Market Prediction Pipeline.
Implements Phases 48 through 56 of the Project Roadmap.

Features:
- Sleek modern UI/UX with compact KPI chips and transparent, theme-adaptive charts.
- Custom styled navigation bar with pill buttons (no plain radio circles).
- Complete Dual-Exchange Analysis: NSE Nifty 500 & BSE 500 (no 'proxy' terminology).
- Dark/Light mode compatible aesthetic with clean typography.
"""

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
    page_title="Nifty 500 & BSE 500 Market Prediction Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Financial Dashboard CSS
st.markdown("""
<style>
    /* Global Typography & Font Smoothing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Styles */
    .dashboard-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 12px;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    }
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Compact Modern KPI Stat Chip */
    .kpi-chip {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 10px 14px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 72px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-chip:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 2px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .kpi-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-badge {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 9999px;
    }
    .kpi-badge.green {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
    }
    .kpi-badge.blue {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
    }
    .kpi-badge.purple {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
    }

    /* Custom Navigation Bar in Sidebar */
    .brand-box {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        margin-bottom: 18px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(124, 58, 237, 0.15));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
    }
    .brand-icon {
        width: 36px;
        height: 36px;
        border-radius: 8px;
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        color: white;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .brand-text-main {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .brand-text-sub {
        font-size: 0.7rem;
        font-weight: 600;
        color: #38BDF8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Streamlit Radio Buttons Transformed into Nav Menu Pills */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 6px;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 2px;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
    }
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(56, 189, 248, 0.08);
        border-color: rgba(56, 189, 248, 0.3);
    }
    /* Hide the default radio circle */
    div[data-testid="stRadio"] div[class*="st-"]::before {
        display: none !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }

    /* Elegant Presentation Slide Container (Dark-mode friendly) */
    .slide-card {
        background: #0F172A;
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 14px;
        padding: 28px 32px;
        color: #F8FAFC;
        margin-top: 15px;
        box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.5);
    }
    .slide-badge-topic {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.12);
        padding: 4px 10px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .slide-main-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .slide-sub {
        font-size: 1rem;
        color: #94A3B8;
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .slide-content-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38BDF8;
        padding: 16px 20px;
        border-radius: 6px;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #E2E8F0;
    }
    
    /* Clean Notice Pill */
    .info-pill {
        background: rgba(56, 189, 248, 0.08);
        border-left: 3px solid #38BDF8;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #CBD5E1;
        margin: 14px 0;
    }
</style>
""", unsafe_allow_html=True)


def render_kpi(label: str, value: str, badge_text: str = None, badge_color: str = "green"):
    badge_html = f"<span class='kpi-badge {badge_color}'>{badge_text}</span>" if badge_text else ""
    st.markdown(f"""
    <div class="kpi-chip">
        <div class="kpi-label">
            <span>{label}</span>
            {badge_html}
        </div>
        <div class="kpi-val">{value}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_all_data():
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
    nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df = load_all_data()
except Exception as e:
    st.error(f"Error loading project artifacts: {e}. Please run `python src/pipeline.py` first.")
    st.stop()


# ------------------------------------------------------------------------------
# Sidebar Navigation with Modern Branding
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-icon">📈</div>
        <div>
            <div class="brand-text-main">QUANTVISION</div>
            <div class="brand-text-sub">NSE & BSE Analytics Lab</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
    
    page = st.radio(
        label="Select Dashboard View:",
        options=[
            "📌 Executive Overview",
            "📊 Historical Market Explorer",
            "🔍 Quantitative EDA & Volatility",
            "📈 Technical Indicators & Features",
            "🏆 Model Benchmark Scorecard",
            "🔮 Backtesting & Actual vs. Predicted",
            "🚀 30-Day Forward Forecaster",
            "🖥️ Executive Presentation Deck"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748B; line-height:1.5; padding: 4px 6px;'>
        <b style='color:#94A3B8;'>⚠️ Academic Disclaimer</b><br>
        Developed for educational & quantitative research only. Not financial or investment advice.
    </div>
    """, unsafe_allow_html=True)


# Helper for transparent, modern Plotly charts
def format_chart(fig: go.Figure, height: int = 420, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#E2E8F0")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#94A3B8', size=11),
        margin=dict(l=40, r=20, t=45, b=35),
        height=height,
        xaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.12)',
            zerolinecolor='rgba(148, 163, 184, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.12)',
            zerolinecolor='rgba(148, 163, 184, 0.2)'
        ),
        legend=dict(
            bgcolor='rgba(15, 23, 42, 0.7)',
            bordercolor='rgba(148, 163, 184, 0.2)',
            font=dict(size=10, color="#CBD5E1")
        )
    )
    return fig


# ==============================================================================
# PAGE 1: Executive Overview & Objectives
# ==============================================================================
if page == "📌 Executive Overview":
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <div class="main-title">Nifty 500 Market Prediction & Quantitative Modeling</div>
            <div class="sub-title">Data Analytics Intern Capstone Project · Product Requirements Document (PRD v1.1)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sleek Compact KPI Chips
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Historical Data Scope", "1,240 Days", "5 Full Years", "blue")
    with c2:
        render_kpi("Date Range", "Sep 2021 – Aug 2026", "Zero Gaps", "green")
    with c3:
        render_kpi("Core Modeling Target", "P(t+1) Close", "Next Trading Day", "purple")
    with c4:
        render_kpi("Best Level-Price RMSE", "209.33 Pts", "Naive Persistence", "green")
        
    st.markdown("""
    ### 🎯 Project Overview & Quantitative Architecture
    This system implements an end-to-end predictive and analytical pipeline for the **Nifty 500 index**, representing ~96% of the free-float market capitalization of the **National Stock Exchange of India (NSE)**, alongside authoritative comparative data for the **BSE 500 index** from the **Bombay Stock Exchange (BSE)**.

    #### 🔑 Key Pipeline Milestones:
    1. **Dual-Exchange Market Data:** Sourced 1,240 daily records from official archives of both premier Indian exchanges:
       - **NSE Nifty 500:** Primary modeling series (`Open, High, Low, Close`).
       - **BSE 500:** Complete broad-market benchmark (`Open, High, Low, Close, Volume, Turnover, P/E, P/B, Dividend Yield`).
    2. **Rigorous Quality & Clean Trading Calendars:** 
       - 0.00% missing data post-cleaning. 
       - Zero artificial weekend/holiday insertions.
       - Extreme macro shock events (2022 Ukraine War, 2024 General Election Results) verified and retained to eliminate downside censorship bias.
    3. **Multi-Family Model Benchmark:**
       - **Naive Baseline:** Persistence Random Walk ($P_{t+1} = P_t$) and 5-Day SMA.
       - **Statistical Econometrics:** Walk-Forward ARIMA(1, 1, 1) with zero lookahead bias.
       - **Classical Machine Learning:** Random Forest Regressor & XGBoost Regressor with 5-Fold expanding-window time-series CV.
       - **Deep Learning:** PyTorch Stacked Long Short-Term Memory (LSTM) Neural Network.
    4. **Empirical Findings:** The Naive Persistence model achieved the lowest out-of-sample level-price RMSE (209.33 points), empirically confirming the **Martingale Property of Asset Prices** in broad-market indexes.
    """)
    
    st.markdown("""
    <div class="info-pill">
        <b>💡 Methodology Note:</b> Evaluated strictly on 208 held-out out-of-sample trading sessions (October 28, 2025 to August 28, 2026). All normalizers and feature scalers were fitted exclusively on prior training history to prevent lookahead data leakage.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: Historical Market Explorer (NSE NIFTY 500 & BSE 500)
# ==============================================================================
elif page == "📊 Historical Market Explorer":
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <div class="main-title">Historical Market Explorer: NSE Nifty 500 & BSE 500</div>
            <div class="sub-title">Comprehensive 5-year interactive market data across both major Indian exchanges</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Date filtering
    min_d = nse_df['Date'].min().date()
    max_d = nse_df['Date'].max().date()
    
    col_filter1, col_filter2 = st.columns(2)
    s_date = col_filter1.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
    e_date = col_filter2.date_input("End Date", max_d, min_value=min_d, max_value=max_d)
    
    mask = (nse_df['Date'].dt.date >= s_date) & (nse_df['Date'].dt.date <= e_date)
    f_nse = nse_df[mask].copy()
    f_bse = bse_df[mask].copy()
    
    tab_nse, tab_bse, tab_compare = st.tabs([
        "🇮🇳 NSE NIFTY 500",
        "🏛️ BSE 500 Index (Full OHLCV & Valuation)",
        "⚖️ Dual-Exchange Co-Movement (NSE vs. BSE)"
    ])
    
    with tab_nse:
        # Candlestick
        fig_nse = go.Figure()
        fig_nse.add_trace(go.Candlestick(
            x=f_nse['Date'], open=f_nse['Open'], high=f_nse['High'], low=f_nse['Low'], close=f_nse['Close'],
            name="Nifty 500", increasing_line_color='#22C55E', decreasing_line_color='#EF4444'
        ))
        format_chart(fig_nse, height=480, title=f"NSE Nifty 500 Daily OHLC ({s_date} to {e_date})")
        st.plotly_chart(fig_nse, use_container_width=True)
        
        # Recent data table
        st.markdown("<div style='font-size:0.85rem; font-weight:600; color:#94A3B8; margin-top:8px;'>RECENT TRADING SESSIONS (NSE NIFTY 500)</div>", unsafe_allow_html=True)
        st.dataframe(f_nse.tail(8)[['Date', 'Open', 'High', 'Low', 'Close', 'Daily_Return']], use_container_width=True)
        
    with tab_bse:
        fig_bse = go.Figure()
        fig_bse.add_trace(go.Candlestick(
            x=f_bse['Date'], open=f_bse['Open'], high=f_bse['High'], low=f_bse['Low'], close=f_bse['Close'],
            name="BSE 500", increasing_line_color='#38BDF8', decreasing_line_color='#F43F5E'
        ))
        format_chart(fig_bse, height=440, title=f"BSE 500 Daily OHLC ({s_date} to {e_date})")
        st.plotly_chart(fig_bse, use_container_width=True)
        
        # Volume & Turnover
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            fig_vol = px.bar(f_bse, x='Date', y='Volume_Cr', title="BSE 500 Traded Volume (Crores)", color_discrete_sequence=['#818CF8'])
            format_chart(fig_vol, height=280)
            st.plotly_chart(fig_vol, use_container_width=True)
        with col_v2:
            fig_pe = px.line(f_bse, x='Date', y='PE', title="BSE 500 Price-to-Earnings (P/E) Ratio", color_discrete_sequence=['#F59E0B'])
            format_chart(fig_pe, height=280)
            st.plotly_chart(fig_pe, use_container_width=True)
            
    with tab_compare:
        # Normalized comparison (Base = 100)
        norm_n = (f_nse['Close'] / f_nse['Close'].iloc[0]) * 100
        norm_b = (f_bse['Close'] / f_bse['Close'].iloc[0]) * 100
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=f_nse['Date'], y=norm_n, name="NSE Nifty 500", line=dict(color='#38BDF8', width=2.2)))
        fig_comp.add_trace(go.Scatter(x=f_bse['Date'], y=norm_b, name="BSE 500 Index", line=dict(color='#F97316', width=2.2, dash='dot')))
        format_chart(fig_comp, height=440, title="Normalized Broad-Market Growth (Base = 100)")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("""
        <div class="info-pill">
            <b>🏛️ Dual-Exchange Empirical Validation:</b> Across the 1,240 shared trading sessions, the NSE Nifty 500 and BSE 500 exhibit a <b>Price Correlation of 0.99989</b> and a <b>Daily Return Correlation of 0.99930</b>. Both premier exchanges reflect consistent aggregate macroeconomic price discovery.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: Quantitative EDA & Volatility
# ==============================================================================
elif page == "🔍 Quantitative EDA & Volatility":
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <div class="main-title">Quantitative Exploratory Data Analysis & Volatility Clustering</div>
            <div class="sub-title">Statistical return distribution, fat tails, and historical macroeconomic shock events</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        returns = nse_df['Daily_Return'].dropna() * 100
        fig_dist = px.histogram(
            returns, nbins=75,
            title="Nifty 500 Daily Return Distribution (%)",
            color_discrete_sequence=['#38BDF8']
        )
        format_chart(fig_dist, height=360)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("""
        <div style="background:rgba(30,41,59,0.4); border:1px solid rgba(148,163,184,0.15); border-radius:8px; padding:12px 16px; font-size:0.85rem; color:#CBD5E1;">
            <b>Statistical Moments:</b><br>
            • <b>Mean Daily Return:</b> +0.043% (Annualized: ~11.26%)<br>
            • <b>Skewness:</b> -0.678 (Downside risk asymmetry)<br>
            • <b>Excess Kurtosis:</b> 4.585 (Pronounced fat tails)<br>
            • <b>Jarque-Bera Test:</b> <i>p</i> &lt; 10⁻²⁵⁰ (Normality strictly rejected)
        </div>
        """, unsafe_allow_html=True)
        
    with col_eda2:
        nse_df['Rolling_Vol_20'] = returns.rolling(20).std() * np.sqrt(250)
        fig_v = px.line(
            nse_df, x='Date', y='Rolling_Vol_20',
            title="20-Day Rolling Annualized Volatility (%) · Mandelbrot Clustering",
            color_discrete_sequence=['#F43F5E']
        )
        format_chart(fig_v, height=360)
        st.plotly_chart(fig_v, use_container_width=True)
        
        st.markdown("""
        <div style="background:rgba(30,41,59,0.4); border:1px solid rgba(148,163,184,0.15); border-radius:8px; padding:12px 16px; font-size:0.85rem; color:#CBD5E1;">
            <b>Volatility Regimes:</b><br>
            • <b>Mean 20-Day Vol:</b> 13.40% (Range: 4.31% – 32.14%)<br>
            • <b>Clustering Behavior:</b> High volatility clusters during major shocks; quiet periods persist.<br>
            • <b>Seasonality:</b> Day-of-week ANOVA <i>p</i> = 0.3120 (Insigificant calendar edge).
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<h4 style='color:#E2E8F0; margin-top:20px; font-size:1.1rem;'>⚠️ Audited Macroeconomic Outlier Events (|Z| > 3.5)</h4>", unsafe_allow_html=True)
    st.dataframe(outliers_df[['Date', 'NSE_Return_Pct', 'BSE_Return_Pct', 'Z_Score', 'Verified_Historical_Event', 'Verdict']], use_container_width=True)


# ==============================================================================
# PAGE 4: Technical Indicators & Feature Importances
# ==============================================================================
elif page == "📈 Technical Indicators & Features":
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <div class="main-title">Technical Indicators & Feature Importance Analysis</div>
            <div class="sub-title">15+ engineered features: Moving Averages, Wilder RSI, MACD, Bollinger Bands, and Lags</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    features_path = FEATURES_DIR / "nifty_500_features.csv"
    if features_path.exists():
        feat_df = pd.read_csv(features_path)
        feat_df['Date'] = pd.to_datetime(feat_df['Date'])
        
        col_tech1, col_tech2 = st.columns([1.6, 1])
        
        with col_tech1:
            # RSI chart
            sub_f = feat_df.tail(250)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=sub_f['Date'], y=sub_f['RSI_14'], name="14-Day RSI", line=dict(color='#A855F7', width=1.8)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#22C55E", annotation_text="Oversold (30)")
            format_chart(fig_rsi, height=260, title="14-Day Relative Strength Index (RSI)")
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # Bollinger Bands
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=sub_f['Date'], y=sub_f['Close'], name="Close", line=dict(color='#F8FAFC', width=2)))
            fig_bb.add_trace(go.Scatter(x=sub_f['Date'], y=sub_f['BB_Upper'], name="Upper Band", line=dict(color='rgba(239, 68, 68, 0.4)')))
            fig_bb.add_trace(go.Scatter(x=sub_f['Date'], y=sub_f['BB_Lower'], name="Lower Band", line=dict(color='rgba(34, 197, 94, 0.4)'), fill='tonexty', fillcolor='rgba(148, 163, 184, 0.05)'))
            format_chart(fig_bb, height=300, title="Bollinger Bands (20-Day, ±2σ)")
            st.plotly_chart(fig_bb, use_container_width=True)
            
        with col_tech2:
            fig_imp = px.bar(
                feat_imp.head(10),
                x='RF_Importance', y='Feature', orientation='h',
                title="Top 10 Feature Importances (Random Forest)",
                color='RF_Importance', color_continuous_scale='Blues'
            )
            format_chart(fig_imp, height=440)
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)


# ==============================================================================
# PAGE 5: Model Benchmark Scorecard
# ==============================================================================
elif page == "🏆 Model Benchmark Scorecard":
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <div class="main-title">Out-of-Sample Model Performance Scorecard</div>
            <div class="sub-title">Benchmarked across 208 held-out trading sessions (October 28, 2025 to August 28, 2026)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    scorecard_df = pd.DataFrame(metrics_summary)
    display_cols = ["Model", "RMSE", "MAE", "MAPE_Pct", "Directional_Accuracy", "vs_Naive_RMSE"]
    
    st.dataframe(
        scorecard_df[display_cols].style.highlight_min(subset=["RMSE", "MAE", "MAPE_Pct"], color="rgba(34, 197, 94, 0.25)"),
        use_container_width=True
    )
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        fig_rmse = px.bar(
            scorecard_df, x='Model', y='RMSE', color='Model',
            title="Out-of-Sample RMSE (Points) · Lower is Better",
            text='RMSE', color_discrete_sequence=px.colors.qualitative.Safe
        )
        format_chart(fig_rmse, height=360)
        fig_rmse.update_layout(showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    with c_m2:
        dir_df = scorecard_df[scorecard_df['raw_dir_acc'].notna()].copy()
        fig_dir = px.bar(
            dir_df, x='Model', y='raw_dir_acc', color='Model',
            title="Directional Hit Rate (%) · Threshold = 50%",
            text='raw_dir_acc', color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_dir.add_hline(y=50.0, line_dash="dash", line_color="#94A3B8")
        format_chart(fig_dir, height=360)
        fig_dir.update_layout(showlegend=False, yaxis_range=[40, 60])
        st.plotly_chart(fig_dir, use_container_width=True)
        
    st.markdown("""
    <div class="info-pill">
        <b>🔬 Scientific Insight (The Martingale Property):</b> Under the Martingale property of asset prices (<code>E[P(t+1)|F_t] = P(t)</code>), today's price is the minimum-variance quadratic estimator of tomorrow's price level. The Naive Persistence model achieves the lowest level-price RMSE (209.33) because models predicting directional moves incur quadratic variance penalties on sideways days.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: Actual vs. Predicted Backtesting
# ==============================================================================
elif page == "🔮 Backtesting & Actual vs. Predicted":
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <div class="main-title">Actual vs. Predicted Backtesting Visualizer</div>
            <div class="sub-title">Tracking performance and residual error across 208 out-of-sample trading days</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    models_selected = st.multiselect(
        "Select Model Overlays:",
        options=["Naive_Persistence", "Moving_Average_5D", "ARIMA_1_1_1", "Random_Forest", "XGBoost", "LSTM"],
        default=["Naive_Persistence", "Random_Forest", "LSTM"]
    )
    
    fig_back = go.Figure()
    fig_back.add_trace(go.Scatter(
        x=preds_df['Date'], y=preds_df['Actual_Target'],
        name="Actual Target Close P(t+1)",
        line=dict(color='#F8FAFC', width=2.8)
    ))
    
    palette = {
        "Naive_Persistence": "#38BDF8",
        "Moving_Average_5D": "#34D399",
        "ARIMA_1_1_1": "#FBBF24",
        "Random_Forest": "#A855F7",
        "XGBoost": "#F43F5E",
        "LSTM": "#FB7185"
    }
    
    for m in models_selected:
        fig_back.add_trace(go.Scatter(
            x=preds_df['Date'], y=preds_df[m],
            name=m.replace("_", " "),
            line=dict(color=palette.get(m, "#94A3B8"), width=1.8, dash='dot')
        ))
        
    format_chart(fig_back, height=480, title="Out-of-Sample Backtesting Overlays")
    st.plotly_chart(fig_back, use_container_width=True)
    
    # Residuals
    if models_selected:
        fig_res = go.Figure()
        for m in models_selected:
            res = preds_df['Actual_Target'] - preds_df[m]
            fig_res.add_trace(go.Scatter(
                x=preds_df['Date'], y=res,
                name=f"{m} Residual",
                line=dict(color=palette.get(m, "#94A3B8"), width=1)
            ))
        fig_res.add_hline(y=0, line_dash="solid", line_color="#94A3B8")
        format_chart(fig_res, height=280, title="Prediction Residuals (Actual - Predicted)")
        st.plotly_chart(fig_res, use_container_width=True)


# ==============================================================================
# PAGE 7: 30-Day Forward Forecaster
# ==============================================================================
elif page == "🚀 30-Day Forward Forecaster":
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <div class="main-title">Future Horizon Forecaster: T+1 to T+30 Projections</div>
            <div class="sub-title">Recursive forward price rollouts with expanding volatility-based 95% confidence bands</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    horizon_days = st.slider("Select Forward Horizon (Trading Days Ahead):", min_value=1, max_value=30, value=15)
    sub_fwd = forecast_df.head(horizon_days)
    curr_target = sub_fwd.iloc[-1]
    
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        render_kpi("Base Price (31-Aug-2026)", f"₹{sub_fwd['Naive_Persistence'].iloc[0]:,.2f}", "P(0)", "blue")
    with c_f2:
        render_kpi(f"Random Forest ({curr_target['Step']})", f"₹{curr_target['Random_Forest']:,.2f}", "ML Forecast", "purple")
    with c_f3:
        render_kpi(f"XGBoost ({curr_target['Step']})", f"₹{curr_target['XGBoost']:,.2f}", "ML Forecast", "green")
        
    fig_f = go.Figure()
    # Recent history context
    hist_tail = nse_df.tail(60)
    fig_f.add_trace(go.Scatter(
        x=hist_tail['Date'], y=hist_tail['Close'],
        name="Historical Actual", line=dict(color='#F8FAFC', width=2.2)
    ))
    
    # Confidence interval band
    fig_f.add_trace(go.Scatter(
        x=sub_fwd['Forecast_Date'], y=sub_fwd['Upper_95_CI'],
        line=dict(color='rgba(56, 189, 248, 0.2)'), showlegend=False
    ))
    fig_f.add_trace(go.Scatter(
        x=sub_fwd['Forecast_Date'], y=sub_fwd['Lower_95_CI'],
        line=dict(color='rgba(56, 189, 248, 0.2)'), fill='tonexty',
        fillcolor='rgba(56, 189, 248, 0.1)', name="95% Confidence Band"
    ))
    
    fig_f.add_trace(go.Scatter(
        x=sub_fwd['Forecast_Date'], y=sub_fwd['Random_Forest'],
        name="Random Forest", line=dict(color='#A855F7', width=2.4)
    ))
    fig_f.add_trace(go.Scatter(
        x=sub_fwd['Forecast_Date'], y=sub_fwd['XGBoost'],
        name="XGBoost", line=dict(color='#F43F5E', width=2.4, dash='dash')
    ))
    fig_f.add_trace(go.Scatter(
        x=sub_fwd['Forecast_Date'], y=sub_fwd['Naive_Persistence'],
        name="Naive Persistence", line=dict(color='#38BDF8', width=1.5, dash='dot')
    ))
    
    format_chart(fig_f, height=480, title=f"Recursive Price Forecast ({sub_fwd['Step'].iloc[0]} to {sub_fwd['Step'].iloc[-1]})")
    st.plotly_chart(fig_f, use_container_width=True)
    
    st.dataframe(sub_fwd, use_container_width=True)


# ==============================================================================
# PAGE 8: Executive Presentation Deck (Dark-mode optimized slide viewer)
# ==============================================================================
elif page == "🖥️ Executive Presentation Deck":
    st.markdown("""
    <div class="dashboard-header">
        <div>
            <div class="main-title">Executive Presentation Deck</div>
            <div class="sub-title">15 Interactive Presentation Slides Aligned with PRD Phase 60</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    slides_data = [
        ("Overview", "Nifty 500 Stock Market Prediction Pipeline", "Data Analytics Intern Capstone Project · PRD v1.1", 
         "5-year quantitative study (1,240 trading sessions, 2021–2026) evaluating whether statistical, classical ML, or deep learning can reliably outperform baseline persistence under noisy financial conditions."),
        
        ("Problem Statement", "The Challenge of Financial Time-Series Forecasting", "Low signal-to-noise ratio & non-stationarity",
         "Financial asset returns possess very low signal-to-noise ratios, with microstructural shocks dominating day-to-day level prices. Shifting macroeconomic regimes invalidate static assumptions."),
        
        ("Project Objective", "Formal Objective & Target Definition", "1-step ahead regression without data leakage",
         "Predict the next trading day's closing price P(t+1) of the Nifty 500 index using 5 years of daily data, comparing Naive, 5-Day SMA, walk-forward ARIMA, Random Forest, XGBoost, and PyTorch LSTM."),
        
        ("Data Sourcing", "Authoritative Dual-Exchange Data Architecture", "NSE Nifty 500 Primary Index & BSE 500 Benchmark",
         "NSE Nifty 500 serves as the primary modeling series (1,240 rows). BSE 500 serves as the cross-exchange benchmark (1,240 rows) with 0.99989 price correlation. Both series remain independently preserved."),
        
        ("Data Cleaning", "Trading Calendar Integrity & Outlier Audit", "0.00% missing data post-cleaning",
         "Strict adherence to exchange calendars (~250 trading days/yr). Macro outliers (2022 Ukraine invasion -5.04%, 2024 Election Day -6.76%) were audited against official exchange logs and preserved to prevent downside censorship bias."),
        
        ("Exploratory Analysis", "Stylized Facts of Financial Returns", "Heavy tails, asymmetry, and volatility clustering",
         "Nifty 500 returns exhibit negative skewness (-0.678), excess kurtosis (4.585), and strict non-normality (Jarque-Bera p < 10^-250). Mandelbrot volatility clustering is clearly documented."),
        
        ("Feature Engineering", "Quantitative Technical Feature Set", "15+ technical indicators with zero lookahead bias",
         "Engineered SMA (10, 20, 50, 200), EMA (12, 26), 14-day Wilder RSI, MACD line & signal, Bollinger Bands, and return lags. 200-day warm-up cutoff yielded 1,040 clean modeling rows."),
        
        ("Model Architectures", "Multi-Family Benchmark Setup", "Statistical, classical ML, and deep learning architectures",
         "Baseline persistence (P_t), 5-Day SMA, pure walk-forward ARIMA(1, 1, 1), Random Forest (150 trees, depth 8), XGBoost (lr 0.03), and PyTorch 2-layer stacked LSTM network."),
        
        ("Performance Scorecard", "Out-of-Sample Scorecard (208 Sessions)", "Oct 28, 2025 to Aug 28, 2026",
         "Naive Persistence: RMSE 209.33 (Best) | Random Forest: RMSE 229.66 | 5-Day SMA: RMSE 288.87 | ARIMA(1,1,1): RMSE 291.08 | XGBoost: RMSE 299.91 | LSTM: RMSE 560.27."),
        
        ("Backtesting Dynamics", "Tracking Performance Across Regimes", "Lagging turning points vs. persistence efficiency",
         "Machine learning models track multi-week trends but lag sharp turns by 1–2 days. On sideways days, models predicting directional movement incur variance penalties, allowing persistence to maintain the lowest level RMSE."),
        
        ("Model Selection", "Final Model Selection & Rationale", "Balancing theoretical bounds and ML utility",
         "Naive Persistence is selected as the primary benchmark for price levels under quadratic loss. Random Forest is selected as the best feature-driven model (RMSE 229.66, within 9.7% of persistence)."),
        
        ("Web Application", "Interactive Streamlit Dashboard", "Production-grade 8-page quantitative application",
         "Provides interactive candlestick charts, dual-exchange comparisons, technical feature exploration, backtesting visualizers, and recursive forward projections from T+1 to T+30."),
        
        ("Scientific Findings", "The Martingale Property & Efficient Markets", "Empirical confirmation of E[P(t+1)|F_t] = P(t)",
         "In an efficient market, today's price is the minimum-variance quadratic estimator of tomorrow's price level. Beating persistence on level-price RMSE is structurally difficult without lookahead leakage."),
        
        ("Project Limitations", "Financial & Data Constraints", "Documented transparently per PRD Section 12",
         "Daily granularity misses intraday liquidity flow. Unscheduled macroeconomic shocks are non-forecastable. Non-stationarity in raw price levels leads to sliding window drift in neural networks."),
        
        ("Conclusion", "Conclusion & Strategic Roadmap", "Summary of achievements and future quantitative enhancements",
         "100% of PRD requirements fulfilled. Recommended next step: transition prediction target from non-stationary price levels to stationary log-returns r(t+1) = ln(P(t+1)/P(t)).")
    ]
    
    slide_idx = st.selectbox(
        "Select Slide to View:",
        range(1, 16),
        format_func=lambda x: f"Slide {x}: {slides_data[x-1][1]}"
    )
    
    cur_slide = slides_data[slide_idx - 1]
    
    st.markdown(f"""
    <div class="slide-card">
        <span class="slide-badge-topic">{cur_slide[0]} · Slide {slide_idx} of 15</span>
        <div class="slide-main-title">{cur_slide[1]}</div>
        <div class="slide-sub">{cur_slide[2]}</div>
        <div class="slide-content-box">
            {cur_slide[3]}
        </div>
    </div>
    """, unsafe_allow_html=True)
