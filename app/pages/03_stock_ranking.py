"""
03_stock_ranking.py
-------------------
Streamlit Dashboard Page 3: 50-Stock Ranking & Comparative League Table.
Ranks all constituents by 21-day predicted return with dynamic filters and CSV download.
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

from config.project_config import STOCK_RANKING_CSV
try:
    from app.components.styling import apply_custom_styles
except ModuleNotFoundError:
    from components.styling import apply_custom_styles

def get_ranking_data():
    if not STOCK_RANKING_CSV.exists():
        return None
    return pd.read_csv(STOCK_RANKING_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("🏆 NIFTY 50 Stock Forecast Ranking")
    st.markdown("##### All NIFTY 50 constituents ranked by model-forecasted 21-trading-day percentage return.")
    st.caption("Ranking basis: Forecasted return over the next 21 trading days. Model selection is based on lowest holdout RMSE.")
    
    df = get_ranking_data()
    if df is None:
        st.warning("⚠️ Ranking data not yet generated. Please execute the modeling pipeline first.")
        return
        
    # Interactive Filters
    st.markdown('<div class="section-header">Filter & Search</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    
    with f1:
        all_industries = ["All Industries"] + sorted(df['Industry'].unique().tolist())
        selected_ind = st.selectbox("Industry:", all_industries)
    with f2:
        all_models = ["All Models"] + sorted(df['Best_Model'].unique().tolist())
        selected_model = st.selectbox("Selected Model:", all_models)
    with f3:
        ret_filter = st.selectbox(
            "Forecast Direction:",
            ["All (50 Stocks)", "Positive (> 0%)", "Negative (< 0%)", "Neutral (0.00%)"]
        )
    with f4:
        top_n = st.selectbox("Display Limit:", ["All 50", "Top 10 Gainers", "Top 20 Gainers", "Bottom 10 Losers"])
        
    filtered_df = df.copy()
    if selected_ind != "All Industries":
        filtered_df = filtered_df[filtered_df['Industry'] == selected_ind]
    if selected_model != "All Models":
        filtered_df = filtered_df[filtered_df['Best_Model'] == selected_model]
    if ret_filter == "Positive (> 0%)":
        filtered_df = filtered_df[filtered_df['Expected_Return_Pct'] > 0]
    elif ret_filter == "Negative (< 0%)":
        filtered_df = filtered_df[filtered_df['Expected_Return_Pct'] < 0]
    elif ret_filter == "Neutral (0.00%)":
        filtered_df = filtered_df[filtered_df['Expected_Return_Pct'] == 0]
        
    if top_n == "Top 10 Gainers":
        filtered_df = filtered_df.head(10)
    elif top_n == "Top 20 Gainers":
        filtered_df = filtered_df.head(20)
    elif top_n == "Bottom 10 Losers":
        filtered_df = filtered_df.tail(10)
        
    st.markdown(f"**Displaying {len(filtered_df)} of {len(df)} stocks**")
    
    # Ranking Table
    st.markdown('<div class="section-header">Ranking Table</div>', unsafe_allow_html=True)
    
    display_df = filtered_df[[
        'Rank', 'Symbol', 'Company', 'Industry', 'Current_Price',
        'Predicted_21D_Price', 'Expected_Change', 'Expected_Return_Pct',
        'Best_Model', 'RMSE', 'MAE', 'MAPE'
    ]].copy()
    
    display_df.rename(columns={
        'Current_Price': 'Current Price',
        'Predicted_21D_Price': 'Forecasted Price (+21 Trading Days)',
        'Expected_Change': 'Forecasted Change',
        'Expected_Return_Pct': 'Forecasted Return (%)',
        'Best_Model': 'Selected Model'
    }, inplace=True)
    
    # Styled dataframe
    st.dataframe(
        display_df.style.format({
            'Current Price': '₹{:,.2f}',
            'Forecasted Price (+21 Trading Days)': '₹{:,.2f}',
            'Forecasted Change': '₹{:+,.2f}',
            'Forecasted Return (%)': '{:+.2f}%',
            'RMSE': '₹{:,.2f}',
            'MAE': '₹{:,.2f}',
            'MAPE': '{:.2f}%'
        }).background_gradient(subset=['Forecasted Return (%)'], cmap='RdYlGn', vmin=-15, vmax=15),
        use_container_width=True,
        hide_index=True
    )
    
    # Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Stock Ranking CSV",
        data=csv_bytes,
        file_name="nifty50_stock_ranking.csv",
        mime="text/csv"
    )
    
    # Note / Disclaimer
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 14px 18px; margin-top: 18px;">
        <span style="color: #94a3b8; font-size: 0.83rem; line-height: 1.5;">
            <strong>NOTE:</strong> Forecasts are model-generated estimates based on historical data and are subject to model error and market uncertainty.
        </span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
