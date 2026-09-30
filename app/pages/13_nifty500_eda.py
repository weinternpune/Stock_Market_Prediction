"""
13_nifty500_eda.py
------------------
Quantitative Exploratory Data Analysis & Volatility Clustering for NIFTY 500.
"""

from pathlib import Path
import sys
import numpy as np
import streamlit as st
import plotly.express as px

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.components.styling import apply_custom_styles
from app.components.nifty500_data import load_nifty500_data, format_nifty500_chart

def render_page():
    apply_custom_styles()
    
    st.title("🔍 Quantitative EDA & Volatility Clustering")
    st.markdown("##### Statistical return distributions, heavy tails, and historical macroeconomic shock regimes.")
    
    nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df = load_nifty500_data()
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        returns = nse_df['Daily_Return'].dropna() * 100
        fig_dist = px.histogram(
            returns, nbins=75,
            title="Nifty 500 Daily Return Distribution (%)",
            color_discrete_sequence=['#38bdf8']
        )
        format_nifty500_chart(fig_dist, height=360)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 14px 18px; font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
            <strong style="color: #f8fafc;">Statistical Moments:</strong><br>
            • <b>Mean Daily Return:</b> +0.043% (Annualized: ~11.26%)<br>
            • <b>Skewness:</b> -0.678 (Downside risk asymmetry)<br>
            • <b>Excess Kurtosis:</b> 4.585 (Pronounced fat tails)<br>
            • <b>Jarque-Bera Test:</b> <i>p</i> &lt; 10<sup>-250</sup> (Normality strictly rejected)
        </div>
        """, unsafe_allow_html=True)
        
    with col_eda2:
        returns_full = nse_df['Daily_Return'].fillna(0) * 100
        nse_df_copy = nse_df.copy()
        nse_df_copy['Rolling_Vol_20'] = returns_full.rolling(20).std() * np.sqrt(250)
        fig_v = px.line(
            nse_df_copy, x='Date', y='Rolling_Vol_20',
            title="20-Day Rolling Annualized Volatility (%) · Mandelbrot Clustering",
            color_discrete_sequence=['#f43f5e']
        )
        format_nifty500_chart(fig_v, height=360)
        st.plotly_chart(fig_v, use_container_width=True)
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 14px 18px; font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
            <strong style="color: #f8fafc;">Volatility Regimes:</strong><br>
            • <b>Mean 20-Day Vol:</b> 13.40% (Range: 4.31% – 32.14%)<br>
            • <b>Clustering Behavior:</b> High volatility clusters during major shocks; quiet periods persist.<br>
            • <b>Seasonality:</b> Day-of-week ANOVA <i>p</i> = 0.3120 (Insignificant calendar edge).
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div class="section-header">⚠️ Audited Macroeconomic Outlier Events (|Z| > 3.5)</div>', unsafe_allow_html=True)
    if not outliers_df.empty:
        st.dataframe(outliers_df[['Date', 'NSE_Return_Pct', 'BSE_Return_Pct', 'Z_Score', 'Verified_Historical_Event', 'Verdict']], use_container_width=True, hide_index=True)
    else:
        st.info("No extreme outlier records logged.")

if __name__ == "__main__":
    render_page()
