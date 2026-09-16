"""
08_future_predictions.py
------------------------
Streamlit Dashboard Page 8: 21-Day Future Forecasts & Uncertainty Intervals.
Dedicated forecasting view with model confidence bounds, expected return chips,
and one-click full prediction CSV export.
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
    st.markdown("##### Forward projections from the latest observed market close with empirical 90% uncertainty intervals.")
    
    df = get_predictions()
    if df is None:
        st.warning("⚠️ Future predictions not found. Please execute the modeling pipeline first.")
        return
        
    # Summary KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Total Forecasts", f"{len(df)} Stocks", subtext="NIFTY 50 Universe")
    with k2:
        n_up = (df['Expected_Return_Pct'] > 0).sum()
        metric_card("Bullish Projections", f"{n_up} Stocks", delta=f"{(n_up/len(df))*100:.1f}%", is_positive=True)
    with k3:
        n_down = (df['Expected_Return_Pct'] < 0).sum()
        metric_card("Bearish Projections", f"{n_down} Stocks", delta=f"{(n_down/len(df))*100:.1f}%", is_positive=False)
    with k4:
        avg_ret = df['Expected_Return_Pct'].mean()
        metric_card("Average Expected Return", f"{avg_ret:+.2f}%", is_positive=(avg_ret >= 0))
        
    # Interactive Filters
    st.markdown('<div class="section-header">Filter Forecast Universe</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    
    with f1:
        scope = st.selectbox("Scope:", ["All 50 Stocks", "Top 5 Gainers", "Top 10 Gainers", "Positive Return Only", "Negative Return Only"])
    with f2:
        all_industries = ["All Industries"] + sorted(df['Industry'].unique().tolist())
        sel_ind = st.selectbox("Industry:", all_industries)
    with f3:
        all_models = ["All Models"] + sorted(df['Best_Model'].unique().tolist())
        sel_model = st.selectbox("Best Model Architecture:", all_models)
        
    filtered = df.copy()
    if sel_ind != "All Industries":
        filtered = filtered[filtered['Industry'] == sel_ind]
    if sel_model != "All Models":
        filtered = filtered[filtered['Best_Model'] == sel_model]
        
    if scope == "Top 5 Gainers":
        filtered = filtered.sort_values('Expected_Return_Pct', ascending=False).head(5)
    elif scope == "Top 10 Gainers":
        filtered = filtered.sort_values('Expected_Return_Pct', ascending=False).head(10)
    elif scope == "Positive Return Only":
        filtered = filtered[filtered['Expected_Return_Pct'] > 0].sort_values('Expected_Return_Pct', ascending=False)
    elif scope == "Negative Return Only":
        filtered = filtered[filtered['Expected_Return_Pct'] < 0].sort_values('Expected_Return_Pct', ascending=True)
    else:
        filtered = filtered.sort_values('Expected_Return_Pct', ascending=False)
        
    # Dataframe with Uncertainty Bounds
    st.markdown(f"**Showing {len(filtered)} Stocks**")
    
    display_cols = [
        'Symbol', 'Company', 'Industry', 'Current_Price', 'Predicted_21D_Price',
        'Expected_Change', 'Expected_Return_Pct', 'Lower_Bound_90', 'Upper_Bound_90',
        'Best_Model', 'RMSE', 'MAPE'
    ]
    
    st.dataframe(
        filtered[display_cols].style.format({
            'Current_Price': '₹{:,.2f}',
            'Predicted_21D_Price': '₹{:,.2f}',
            'Expected_Change': '₹{:+,.2f}',
            'Expected_Return_Pct': '{:+.2f}%',
            'Lower_Bound_90': '₹{:,.2f}',
            'Upper_Bound_90': '₹{:,.2f}',
            'RMSE': '₹{:,.2f}',
            'MAPE': '{:.2f}%'
        }).background_gradient(subset=['Expected_Return_Pct'], cmap='RdYlGn', vmin=-12, vmax=12),
        use_container_width=True,
        hide_index=True
    )
    
    # Download Button
    csv_bytes = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Future Predictions CSV",
        data=csv_bytes,
        file_name="nifty50_future_predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    render_page()
