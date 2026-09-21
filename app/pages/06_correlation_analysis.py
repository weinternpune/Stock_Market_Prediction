"""
06_correlation_analysis.py
--------------------------
Streamlit Dashboard Page 6: Cross-Stock Correlation & Co-Movement Analysis.
Displays empirical correlation structure of historical daily returns across the NIFTY 50
universe, with industry filtering, custom stock selection, and pairwise co-movement analytics.
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
    from app.components.cards import metric_card
    from app.components.charts import plot_correlation_heatmap
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.cards import metric_card
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
    st.markdown("##### Empirical correlation structure of daily returns across the NIFTY 50 universe.")
    
    corr_df = get_correlation_matrix()
    pred_df = get_predictions()
    
    if corr_df is None or pred_df is None:
        st.warning("⚠️ Correlation matrix not found. Please execute the modeling/EDA pipeline first.")
        return
        
    # Methodological Context Box
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
        <div style="color: #38bdf8; font-weight: 600; font-size: 0.95rem; margin-bottom: 6px;">METHODOLOGICAL CONTEXT</div>
        <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
            Pearson correlation measures pairwise linear co-movement in historical daily returns.<br>
            Correlation ranges from <strong>-1.0</strong> to <strong>+1.0</strong>.<br>
            <strong>Correlation does not imply causation.</strong> Low or negative correlation pairs indicate potential portfolio diversification benefits rather than structural independence.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metadata KPI Row: Analysis Period, Correlation Method & Universe
    c_meta1, c_meta2, c_meta3 = st.columns(3)
    with c_meta1:
        metric_card("Analysis Period", "Sep 2021 – Sep 2026", subtext="15-Sep-2021 → 11-Sep-2026")
    with c_meta2:
        metric_card("Correlation Method", "Pearson", subtext="Pairwise Daily Percentage Returns")
    with c_meta3:
        metric_card("Analyzed Universe", f"{len(corr_df)} Stocks", subtext="Full NIFTY 50 Constituents")
        
    # Matrix Scope & Filtering
    st.markdown('<div class="section-header">Matrix Scope & Filtering</div>', unsafe_allow_html=True)
    f1, f2 = st.columns([2, 3])
    
    with f1:
        mode = st.radio(
            "Select Scope:",
            ["Full NIFTY 50 Matrix", "By Specific Industry", "Custom Stock Selection"],
            horizontal=True
        )
        
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
    
    # Daily Return Correlation Matrix Heatmap
    st.markdown('<div class="section-header">Daily Return Correlation Matrix</div>', unsafe_allow_html=True)
    
    fig_corr = plot_correlation_heatmap(
        sub_corr,
        title=f"Daily Return Correlation Matrix ({len(stocks_to_display)} Stocks)"
    )
    if len(stocks_to_display) > 25:
        fig_corr.update_layout(height=780)
    else:
        fig_corr.update_layout(height=520)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Download Correlation CSV
    csv_corr = sub_corr.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Correlation Matrix CSV",
        data=csv_corr,
        file_name="nifty50_correlation_matrix.csv",
        mime="text/csv"
    )
    
    # Strongest Co-Movement Pairs
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
            st.dataframe(
                pairs_df.sort_values('Correlation', ascending=False).head(5),
                use_container_width=True,
                hide_index=True
            )
        with c_low:
            st.markdown("##### Lowest Correlation Pairs")
            st.dataframe(
                pairs_df.sort_values('Correlation', ascending=True).head(5),
                use_container_width=True,
                hide_index=True
            )
            
    # Analytical Disclaimer Note Box
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 14px 18px; margin-top: 20px;">
        <span style="color: #94a3b8; font-size: 0.83rem; line-height: 1.5;">
            <strong>NOTE:</strong> Correlation describes historical linear co-movement and may change over time. It does not imply causation. Low correlation pairs indicate potential portfolio diversification benefits rather than structural independence.
        </span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
