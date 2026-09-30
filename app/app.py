"""
app.py
------
Main Application Entry Point for NIFTY 50 Stock Price Prediction & Analytics.
Utilizes modern Streamlit st.Page and st.navigation architecture for seamless
multipage navigation across all 10 analytical sections.
"""

import sys
from pathlib import Path

# Add project root and app dir to sys.path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="NIFTY 50 Quant Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App directory paths
PAGES_DIR = APP_DIR / "pages"

# Define Dual-Universe Multipage Navigation
pages = {
    "NIFTY 50 (50 Constituent Companies)": [
        st.Page(str(PAGES_DIR / "01_overview.py"), title="Market Overview", icon="🏠", default=True),
        st.Page(str(PAGES_DIR / "02_stock_analysis.py"), title="Stock Analysis", icon="📈"),
        st.Page(str(PAGES_DIR / "03_stock_ranking.py"), title="Stock Ranking", icon="🏆"),
        st.Page(str(PAGES_DIR / "04_model_performance.py"), title="Model Scorecard", icon="🤖"),
        st.Page(str(PAGES_DIR / "05_sector_analysis.py"), title="Industry Analytics", icon="🏭"),
        st.Page(str(PAGES_DIR / "06_correlation_analysis.py"), title="Correlation Matrix", icon="🔗"),
        st.Page(str(PAGES_DIR / "07_backtesting.py"), title="Backtesting", icon="📊"),
        st.Page(str(PAGES_DIR / "08_future_predictions.py"), title="Future Forecasts", icon="🔮"),
        st.Page(str(PAGES_DIR / "09_data_quality.py"), title="Data Quality Audit", icon="🧹"),
    ],
    "NIFTY 500 & BSE 500 (Market Benchmark)": [
        st.Page(str(PAGES_DIR / "11_nifty500_overview.py"), title="Executive Overview", icon="📌"),
        st.Page(str(PAGES_DIR / "12_nifty500_explorer.py"), title="Dual-Exchange Explorer", icon="🌐"),
        st.Page(str(PAGES_DIR / "13_nifty500_eda.py"), title="Quantitative EDA & Volatility", icon="🔍"),
        st.Page(str(PAGES_DIR / "14_nifty500_features.py"), title="Technical Features", icon="📉"),
        st.Page(str(PAGES_DIR / "15_nifty500_models.py"), title="Model Scorecard", icon="🏅"),
        st.Page(str(PAGES_DIR / "16_nifty500_backtest.py"), title="Backtesting Visualizer", icon="🎯"),
        st.Page(str(PAGES_DIR / "17_nifty500_forecast.py"), title="30-Day Forecaster", icon="🚀"),
        st.Page(str(PAGES_DIR / "18_nifty500_deck.py"), title="Executive Presentation Deck", icon="🖥️"),
    ]
}

# Configure Navigation
pg = st.navigation(pages)

# Sleek Sidebar Header
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 4px 18px 4px; border-bottom: 1px solid rgba(148, 163, 184, 0.15); margin-bottom: 14px;">
        <div style="font-size: 1.18rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
            📈 NIFTY <span style="color: #38bdf8;">ANALYTICS PRO</span>
        </div>
        <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 2px;">
            Dual-Universe: 50 Constituents & 500 Benchmark
        </div>
        <div style="display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap;">
            <span style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 9999px; padding: 2px 8px; font-size: 0.68rem; font-weight: 700; color: #34d399;">
                50 CONSTITUENTS
            </span>
            <span style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 9999px; padding: 2px 8px; font-size: 0.68rem; font-weight: 700; color: #38bdf8;">
                NSE 500 + BSE 500
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Execute active page
pg.run()

