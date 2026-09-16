"""
06_correlation_analysis.py
--------------------------
Streamlit Dashboard Page 6: Cross-Stock Return Correlation Heatmap.
Displays empirical correlation matrix of historical daily returns across the NIFTY 50
constituents, with custom multi-select and sector filters.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

import sys
PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config.project_config import METRICS_DIR, PREDICTIONS_CSV
try:
    from app.components.styling import apply_custom_styles
    from app.components.charts import plot_correlation_heatmap
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.charts import plot_correlation_heatmap

@st.cache_data
def get_correlation_matrix():
    corr_path = METRICS_DIR / "correlation_matrix.csv"
    if not corr_path.exists():
        return None
    return pd.read_csv(corr_path, index_col=0)

@st.cache_data
def get_predictions():
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("🔗 Cross-Stock Correlation & Co-Movement Analysis")
    st.markdown("##### Empirical correlation structure of daily price changes across the NIFTY 50 universe.")
    
    corr_df = get_correlation_matrix()
    pred_df = get_predictions()
    
    if corr_df is None or pred_df is None:
        st.warning("⚠️ Correlation matrix not found. Please execute the EDA stage first.")
        return
        
    st.info("ℹ️ **Methodological Context**: Correlation measures pairwise linear co-movement in historical percentage returns. High positive correlation indicates stocks frequently rise and fall together. **Correlation does not imply causation.**")
    
    # Filter Controls
    st.markdown('<div class="section-header">Matrix Scope & Filtering</div>', unsafe_allow_html=True)
    f1, f2 = st.columns([2, 3])
    
    with f1:
        mode = st.radio("View Mode:", ["Full NIFTY 50 Matrix", "By Specific Industry", "Custom Stock Selection"], horizontal=True)
        
    stocks_to_display = list(corr_df.columns)
    
    if mode == "By Specific Industry":
        industries = sorted(pred_df['Industry'].unique().tolist())
        sel_ind = st.selectbox("Select Industry:", industries)
        stocks_to_display = pred_df[pred_df['Industry'] == sel_ind]['Symbol'].tolist()
        stocks_to_display = [s for s in stocks_to_display if s in corr_df.columns]
    elif mode == "Custom Stock Selection":
        default_picks = ['HDFCBANK', 'ICICIBANK', 'INFY', 'TCS', 'RELIANCE', 'TATAMOTORS', 'ITC', 'LT']
        default_picks = [p for p in default_picks if p in corr_df.columns]
        stocks_to_display = st.multiselect("Pick Stocks to Compare:", list(corr_df.columns), default=default_picks)
        
    if len(stocks_to_display) < 2:
        st.warning("Please select at least 2 stocks to compute a correlation matrix.")
        return
        
    sub_corr = corr_df.loc[stocks_to_display, stocks_to_display]
    
    st.markdown(f"**Showing {len(stocks_to_display)} stocks**")
    fig_corr = plot_correlation_heatmap(sub_corr, title=f"Daily Return Correlation Matrix ({len(stocks_to_display)} Stocks)")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Pairwise Table Highlights
    st.markdown('<div class="section-header">Strongest Co-Movement Pairs</div>', unsafe_allow_html=True)
    pairs = []
    cols = list(sub_corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = cols[i], cols[j]
            val = sub_corr.loc[s1, s2]
            pairs.append({"Stock 1": s1, "Stock 2": s2, "Correlation": round(float(val), 3)})
            
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        c_high, c_low = st.columns(2)
        with c_high:
            st.markdown("##### Highest Positive Correlations")
            st.dataframe(pairs_df.sort_values('Correlation', ascending=False).head(5), use_container_width=True, hide_index=True)
        with c_low:
            st.markdown("##### Lowest / Negative Correlations (Diversifiers)")
            st.dataframe(pairs_df.sort_values('Correlation', ascending=True).head(5), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    render_page()
