"""
08_future_predictions.py
------------------------
Streamlit Dashboard Page 8: 21-Trading-Day Future Forecasts & Uncertainty Intervals.
Model-generated price forecasts for the next 21 trading days from the latest available
market close, with empirical 90% uncertainty bounds and full prediction CSV export.
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

from config.project_config import PREDICTIONS_CSV
try:
    from app.components.styling import apply_custom_styles
    from app.components.cards import metric_card
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.cards import metric_card

@st.cache_data
def get_predictions():
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("🔮 21-Trading-Day Future Forecasts")
    st.markdown("##### Model-generated price forecasts for the next 21 trading days from the latest available market close.")
    
    df = get_predictions()
    if df is None:
        st.warning("⚠️ Future predictions not found. Please execute the modeling pipeline first.")
        return
        
    try:
        dt_val = pd.to_datetime(df['Latest_Date'].max()).strftime('%d %b %Y')
    except Exception:
        dt_val = str(df['Latest_Date'].max())
        
    st.caption(f"**Data As Of:** {dt_val} | **Forecast Horizon:** 21 Trading Days | **Model Selection:** Lowest Holdout RMSE")
    
    # -------------------------------------------------------------
    # FORECAST SUMMARY
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Forecast Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    
    n_total = len(df)
    n_up = (df['Expected_Return_Pct'] > 0).sum()
    n_down = (df['Expected_Return_Pct'] < 0).sum()
    n_neu = (df['Expected_Return_Pct'] == 0).sum()
    avg_ret = df['Expected_Return_Pct'].mean()
    
    with k1:
        metric_card("Stocks Forecasted", f"{n_total}", subtext="NIFTY 50 Universe")
    with k2:
        metric_card("Positive Forecasts", f"{n_up} Stocks", delta=f"{(n_up / n_total) * 100:.1f}%", is_positive=True)
    with k3:
        metric_card("Negative Forecasts", f"{n_down} Stocks", delta=f"{(n_down / n_total) * 100:.1f}%", is_positive=False)
    with k4:
        neu_label = "1 Stock" if n_neu == 1 else f"{n_neu} Stocks"
        metric_card("Neutral Forecasts", neu_label, delta=f"{(n_neu / n_total) * 100:.1f}%", is_positive=None, subtext="Flat (0.00% Return)")
    with k5:
        metric_card("Average Forecasted 21-Day Return", f"{avg_ret:+.2f}%", is_positive=(avg_ret >= 0), subtext="Equal-weighted universe")
        
    # -------------------------------------------------------------
    # FILTER FORECAST UNIVERSE
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Filter Forecast Universe</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    
    with f1:
        scope = st.selectbox("Scope:", [
            "All (50 Stocks)",
            "5 Highest Forecasted Returns",
            "10 Highest Forecasted Returns",
            "Forecasted Positive Return Only",
            "Forecasted Negative Return Only",
            "Forecasted Neutral Return Only (0.00%)"
        ])
    with f2:
        all_industries = ["All Industries"] + sorted(df['Industry'].unique().tolist())
        sel_ind = st.selectbox("Industry:", all_industries)
    with f3:
        all_models = ["All Models"] + sorted(df['Best_Model'].unique().tolist())
        sel_model = st.selectbox("Forecasting Model:", all_models)
        
    filtered = df.copy()
    if sel_ind != "All Industries":
        filtered = filtered[filtered['Industry'] == sel_ind]
    if sel_model != "All Models":
        filtered = filtered[filtered['Best_Model'] == sel_model]
        
    if scope == "5 Highest Forecasted Returns":
        filtered = filtered.sort_values('Expected_Return_Pct', ascending=False).head(5)
    elif scope == "10 Highest Forecasted Returns":
        filtered = filtered.sort_values('Expected_Return_Pct', ascending=False).head(10)
    elif scope == "Forecasted Positive Return Only":
        filtered = filtered[filtered['Expected_Return_Pct'] > 0].sort_values('Expected_Return_Pct', ascending=False)
    elif scope == "Forecasted Negative Return Only":
        filtered = filtered[filtered['Expected_Return_Pct'] < 0].sort_values('Expected_Return_Pct', ascending=True)
    elif scope == "Forecasted Neutral Return Only (0.00%)":
        filtered = filtered[filtered['Expected_Return_Pct'] == 0].sort_values('Symbol')
    else:
        filtered = filtered.sort_values('Expected_Return_Pct', ascending=False)
        
    # Showing Counter
    st.markdown(f"**Showing {len(filtered)} of {len(df)} NIFTY 50 Stocks**")
    
    # Clean Display Table
    display_df = filtered[[
        'Symbol', 'Company', 'Industry', 'Current_Price', 'Predicted_21D_Price',
        'Expected_Change', 'Expected_Return_Pct', 'Lower_Bound_90', 'Upper_Bound_90',
        'Best_Model', 'RMSE', 'MAPE'
    ]].copy()
    
    display_df.rename(columns={
        'Current_Price': 'Latest Close (₹)',
        'Predicted_21D_Price': '21-Trading-Day Forecast (₹)',
        'Expected_Change': 'Expected Price Change (₹)',
        'Expected_Return_Pct': 'Expected 21-Day Return (%)',
        'Lower_Bound_90': '90% Lower Bound (₹)',
        'Upper_Bound_90': '90% Upper Bound (₹)',
        'Best_Model': 'Selected Model',
        'RMSE': 'RMSE (₹)',
        'MAPE': 'MAPE (%)'
    }, inplace=True)
    
    st.dataframe(
        display_df.style.format({
            'Latest Close (₹)': '₹{:,.2f}',
            '21-Trading-Day Forecast (₹)': '₹{:,.2f}',
            'Expected Price Change (₹)': '₹{:+,.2f}',
            'Expected 21-Day Return (%)': '{:+.2f}%',
            '90% Lower Bound (₹)': '₹{:,.2f}',
            '90% Upper Bound (₹)': '₹{:,.2f}',
            'RMSE (₹)': '₹{:,.2f}',
            'MAPE (%)': '{:.2f}%'
        }).background_gradient(subset=['Expected 21-Day Return (%)'], cmap='RdYlGn', vmin=-12, vmax=12),
        use_container_width=True,
        hide_index=True
    )
    
    # Download Button
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Future Predictions CSV",
        data=csv_bytes,
        file_name="nifty50_future_predictions.csv",
        mime="text/csv"
    )
    
    # Analytical Disclaimer Note Box
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 14px 18px; margin-top: 18px;">
        <span style="color: #94a3b8; font-size: 0.83rem; line-height: 1.5;">
            <strong>NOTE:</strong> Forecasts are model-generated estimates based on historical data and empirical holdout residual uncertainty bounds. They are subject to model error and market uncertainty, and do not represent guaranteed future prices or investment advice.
        </span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
