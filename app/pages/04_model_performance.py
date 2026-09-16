"""
04_model_performance.py
-----------------------
Streamlit Dashboard Page 4: Cross-Model Comparative Performance & Architecture Audit.
Compares Baseline, ARIMA, XGBoost, and PyTorch LSTM across MAE, RMSE, MAPE, and win counts.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys
PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config.project_config import MODEL_COMPARISON_CSV, PREDICTIONS_CSV, METRICS_DIR
try:
    from app.components.styling import apply_custom_styles
    from app.components.charts import plot_model_win_counts, CHART_THEME
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.charts import plot_model_win_counts, CHART_THEME

from src.utilities.io_helpers import load_json

@st.cache_data
def get_data():
    if not MODEL_COMPARISON_CSV.exists() or not PREDICTIONS_CSV.exists():
        return None, None
    return pd.read_csv(MODEL_COMPARISON_CSV), pd.read_csv(PREDICTIONS_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("🤖 Model Performance & Evaluation Scorecard")
    st.markdown("##### Rigorous empirical comparison of statistical, machine-learning, and deep-learning architectures.")
    
    comp_df, pred_df = get_data()
    if comp_df is None or pred_df is None:
        st.warning("⚠️ Model comparison metrics not found. Please run the modeling pipeline first.")
        return
        
    summary = load_json(METRICS_DIR / "master_model_metrics.json")
    
    # Win Counts Bar Chart
    win_counts = pred_df['Best_Model'].value_counts().to_dict()
    st.markdown('<div class="section-header">Architecture Win Breakdown</div>', unsafe_allow_html=True)
    
    col_chart, col_stats = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(plot_model_win_counts(win_counts), use_container_width=True)
    with col_stats:
        st.markdown("##### Overall Architecture Averages")
        model_avgs = comp_df.groupby('Model')[['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']].mean().round(2).reset_index()
        st.dataframe(model_avgs, use_container_width=True, hide_index=True)
        st.caption("Lower RMSE/MAE/MAPE indicates superior out-of-sample fit. Directional accuracy reflects sign correctness.")
        
    # Error Distributions (Box Plots)
    st.markdown('<div class="section-header">Error Distribution Comparison (Holdout Test Set)</div>', unsafe_allow_html=True)
    
    tab_mape, tab_rmse, tab_mae = st.tabs(["MAPE (%) Distribution", "RMSE (₹) Distribution", "MAE (₹) Distribution"])
    
    with tab_mape:
        fig_box = px.box(
            comp_df, x="Model", y="MAPE", color="Model",
            title="Mean Absolute Percentage Error (MAPE %) by Architecture",
            template=CHART_THEME
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with tab_rmse:
        fig_rmse = px.box(
            comp_df, x="Model", y="RMSE", color="Model",
            title="Root Mean Squared Error (RMSE ₹) by Architecture",
            template=CHART_THEME
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    with tab_mae:
        fig_mae = px.box(
            comp_df, x="Model", y="MAE", color="Model",
            title="Mean Absolute Error (MAE ₹) by Architecture",
            template=CHART_THEME
        )
        st.plotly_chart(fig_mae, use_container_width=True)
        
    # Full Stock-by-Stock Model Scorecard
    st.markdown('<div class="section-header">Stock-by-Stock Multi-Model Scorecard</div>', unsafe_allow_html=True)
    
    # Pivot table: Stock x Model MAE
    pivot_rmse = comp_df.pivot(index=['Symbol', 'Company'], columns='Model', values='RMSE').round(2).reset_index()
    pivot_rmse['Best_Model'] = pred_df.set_index('Symbol').loc[pivot_rmse['Symbol']]['Best_Model'].values
    
    st.dataframe(pivot_rmse, use_container_width=True, hide_index=True)
    
    # Download Comparison CSV
    csv_bytes = comp_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Complete Model Comparison CSV",
        data=csv_bytes,
        file_name="nifty50_model_comparison.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    render_page()
