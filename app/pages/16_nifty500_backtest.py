"""
16_nifty500_backtest.py
-----------------------
Actual vs. Predicted Backtesting Visualizer for NIFTY 500 Models.
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
from app.components.nifty500_data import load_nifty500_data, format_nifty500_chart

def render_page():
    apply_custom_styles()
    
    st.title("🔮 Actual vs. Predicted Backtesting Visualizer")
    st.markdown("##### Tracking model performance and residual errors across 208 out-of-sample trading days.")
    
    nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df = load_nifty500_data()
    
    avail_models = [c for c in ["Naive_Persistence", "Moving_Average_5D", "ARIMA_1_1_1", "Random_Forest", "XGBoost", "LSTM"] if c in preds_df.columns]
    
    models_selected = st.multiselect(
        "Select Model Overlays:",
        options=avail_models,
        default=[m for m in ["Naive_Persistence", "Random_Forest", "LSTM"] if m in avail_models]
    )
    
    fig_back = go.Figure()
    fig_back.add_trace(go.Scatter(
        x=preds_df['Date'], y=preds_df['Actual_Target'],
        name="Actual Target Close P(t+1)",
        line=dict(color='#f8fafc', width=2.8)
    ))
    
    palette = {
        "Naive_Persistence": "#38bdf8",
        "Moving_Average_5D": "#34d399",
        "ARIMA_1_1_1": "#fbbf24",
        "Random_Forest": "#a855f7",
        "XGBoost": "#f43f5e",
        "LSTM": "#fb7185"
    }
    
    for m in models_selected:
        fig_back.add_trace(go.Scatter(
            x=preds_df['Date'], y=preds_df[m],
            name=m.replace("_", " "),
            line=dict(color=palette.get(m, "#94a3b8"), width=1.8, dash='dot')
        ))
        
    format_nifty500_chart(fig_back, height=480, title="Out-of-Sample Backtesting Overlays")
    st.plotly_chart(fig_back, use_container_width=True)
    
    if models_selected:
        st.markdown('<div class="section-header">Prediction Residuals (Actual - Predicted)</div>', unsafe_allow_html=True)
        fig_res = go.Figure()
        for m in models_selected:
            res = preds_df['Actual_Target'] - preds_df[m]
            fig_res.add_trace(go.Scatter(
                x=preds_df['Date'], y=res,
                name=f"{m} Residual",
                line=dict(color=palette.get(m, "#94a3b8"), width=1)
            ))
        fig_res.add_hline(y=0, line_dash="solid", line_color="#94a3b8")
        format_nifty500_chart(fig_res, height=280, title="Residual Errors Over Time")
        st.plotly_chart(fig_res, use_container_width=True)

if __name__ == "__main__":
    render_page()
