"""
17_nifty500_forecast.py
-----------------------
Future Horizon Forecaster: T+1 to T+30 Projections for NIFTY 500.
"""

from pathlib import Path
import sys
import streamlit as st
import plotly.graph_objects as go

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.components.styling import apply_custom_styles
from app.components.cards import metric_card
from app.components.nifty500_data import load_nifty500_data, format_nifty500_chart

def render_page():
    apply_custom_styles()
    
    st.title("🚀 NIFTY 500: 30-Day Forward Forecaster")
    st.markdown("##### Recursive forward price projections with expanding volatility-based 95% confidence intervals.")
    
    nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df = load_nifty500_data()
    
    horizon_days = st.slider("Select Forward Horizon (Trading Days Ahead):", min_value=1, max_value=30, value=15)
    sub_fwd = forecast_df.head(horizon_days)
    curr_target = sub_fwd.iloc[-1]
    
    # Modern KPI cards
    c_f1, c_f2, c_f3 = st.columns(3)
    with c_f1:
        base_p = sub_fwd['Naive_Persistence'].iloc[0]
        metric_card("Base Close (31-Aug-2026)", f"₹{base_p:,.2f}", subtext="Origin Price P(0)")
    with c_f2:
        rf_p = curr_target['Random_Forest']
        rf_chg = ((rf_p - base_p) / base_p) * 100
        metric_card(f"Random Forest ({curr_target['Step']})", f"₹{rf_p:,.2f}", delta=f"{rf_chg:+.2f}%", is_positive=(rf_chg >= 0))
    with c_f3:
        xgb_p = curr_target['XGBoost']
        xgb_chg = ((xgb_p - base_p) / base_p) * 100
        metric_card(f"XGBoost ({curr_target['Step']})", f"₹{xgb_p:,.2f}", delta=f"{xgb_chg:+.2f}%", is_positive=(xgb_chg >= 0))
        
    fig_f = go.Figure()
    hist_tail = nse_df.tail(60)
    fig_f.add_trace(go.Scatter(
        x=hist_tail['Date'], y=hist_tail['Close'],
        name="Historical Actual", line=dict(color='#f8fafc', width=2.2)
    ))
    
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
        name="Random Forest", line=dict(color='#a855f7', width=2.4)
    ))
    fig_f.add_trace(go.Scatter(
        x=sub_fwd['Forecast_Date'], y=sub_fwd['XGBoost'],
        name="XGBoost", line=dict(color='#f43f5e', width=2.4, dash='dash')
    ))
    fig_f.add_trace(go.Scatter(
        x=sub_fwd['Forecast_Date'], y=sub_fwd['Naive_Persistence'],
        name="Naive Persistence", line=dict(color='#38bdf8', width=1.5, dash='dot')
    ))
    
    format_nifty500_chart(fig_f, height=480, title=f"Recursive Price Forecast ({sub_fwd['Step'].iloc[0]} to {sub_fwd['Step'].iloc[-1]})")
    st.plotly_chart(fig_f, use_container_width=True)
    
    st.markdown('<div class="section-header">Forecast Data Table</div>', unsafe_allow_html=True)
    st.dataframe(sub_fwd.style.format({
        'Naive_Persistence': '₹{:,.2f}',
        'Random_Forest': '₹{:,.2f}',
        'XGBoost': '₹{:,.2f}',
        'Lower_95_CI': '₹{:,.2f}',
        'Upper_95_CI': '₹{:,.2f}'
    }), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    render_page()
