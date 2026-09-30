"""
11_nifty500_overview.py
-----------------------
NIFTY 500 & BSE 500 Executive Overview and High-Level Modeling Architecture.
"""

from pathlib import Path
import sys
import streamlit as st

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.components.styling import apply_custom_styles
from app.components.cards import metric_card
from app.components.nifty500_data import load_nifty500_data

def render_page():
    apply_custom_styles()
    
    st.title("📌 Nifty 500 & BSE 500 Executive Overview")
    st.markdown("##### Broad-market Indian equity benchmark prediction and dual-exchange modeling architecture.")
    
    nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df = load_nifty500_data()
    
    # Modern KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Historical Data Scope", f"{len(nse_df):,} Sessions", subtext="5 Full Years (2021–2026)")
    with k2:
        date_range = f"{nse_df['Date'].min().strftime('%b %Y')} – {nse_df['Date'].max().strftime('%b %Y')}"
        metric_card("Trading Calendar", date_range, subtext="100% Valid Trading Sessions")
    with k3:
        metric_card("Core Modeling Target", "P(t+1) Close", subtext="Next Trading Day Regression")
    with k4:
        metric_card("Benchmark Best RMSE", "209.33 Pts", subtext="Naive Persistence (Martingale)")
        
    st.markdown('<div class="section-header">Project Overview & Quantitative Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    This system implements an end-to-end predictive and analytical pipeline for the **Nifty 500 index**, 
    representing ~96% of the free-float market capitalization of the **National Stock Exchange of India (NSE)**, 
    alongside authoritative comparative data for the **BSE 500 index** from the **Bombay Stock Exchange (BSE)**.

    #### Key Pipeline Milestones:
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
    <div style="background: rgba(30, 41, 59, 0.55); border-left: 4px solid #38bdf8; padding: 14px 18px; border-radius: 8px; color: #cbd5e1; font-size: 0.85rem; margin-top: 16px;">
        <strong style="color: #38bdf8;">Methodology Note:</strong> Evaluated strictly on 208 held-out out-of-sample trading sessions (October 28, 2025 to August 28, 2026). All normalizers and feature scalers were fitted exclusively on prior training history to prevent lookahead data leakage.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
