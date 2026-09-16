"""
07_backtesting.py
-----------------
Streamlit Dashboard Page 7: Historical Backtesting & Out-of-Sample Performance Audit.
Plots predicted vs. actual prices over the held-out test sessions, with residual
histograms, prediction error curves, and directional fidelity meters.
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

from config.project_config import BACKTEST_PREDICTIONS_CSV, MODEL_COMPARISON_CSV
try:
    from app.components.styling import apply_custom_styles
    from app.components.cards import metric_card
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.cards import metric_card

from app.components.charts import plot_backtest_actual_vs_predicted, plot_residual_distribution
from src.evaluation.evaluator import compute_evaluation_metrics

@st.cache_data
def get_backtest_data():
    if not BACKTEST_PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(BACKTEST_PREDICTIONS_CSV)

@st.cache_data
def get_comparison_data():
    if not MODEL_COMPARISON_CSV.exists():
        return None
    return pd.read_csv(MODEL_COMPARISON_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("📊 Historical Backtesting & Out-of-Sample Validation")
    st.markdown("##### Rigorous inspection of how models performed historically on unseen holdout test data.")
    
    backtest_df = get_backtest_data()
    comp_df = get_comparison_data()
    
    if backtest_df is None or comp_df is None:
        st.warning("⚠️ Backtesting predictions not found. Please execute the modeling pipeline first.")
        return
        
    symbols = sorted(backtest_df['Symbol'].unique())
    models = sorted(backtest_df['Model'].unique())
    
    # Selection Controls
    st.markdown('<div class="section-header">Backtest Parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_sym = st.selectbox("Select Stock:", symbols, index=0)
    with c2:
        sel_model = st.selectbox("Select Model Architecture:", models, index=0)
    with c3:
        st.info("ℹ️ Test sessions represent unseen, strictly held-out forward market data.")
        
    sub_df = backtest_df[(backtest_df['Symbol'] == sel_sym) & (backtest_df['Model'] == sel_model)].sort_values('Date')
    
    if sub_df.empty:
        st.warning("No backtesting data found for this combination.")
        return
        
    y_true = sub_df['Actual_Future_Close'].values
    y_pred = sub_df['Predicted_Future_Close'].values
    eval_metrics = compute_evaluation_metrics(y_true, y_pred)
    
    # Backtest Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Test Sessions", f"{len(sub_df)} Days", subtext="Out-of-sample window")
    with m2:
        metric_card("Root Mean Sq Error (RMSE)", f"₹{eval_metrics['rmse']:,.2f}", subtext="Penalty for large errors")
    with m3:
        metric_card("Mean Abs Error (MAE)", f"₹{eval_metrics['mae']:,.2f}", subtext="Average deviation")
    with m4:
        metric_card("Mean Abs % Error (MAPE)", f"{eval_metrics['mape']:.2f}%", subtext="Normalized percentage error")
        
    # Main Backtest Plot
    st.markdown('<div class="section-header">Actual vs. Predicted 21-Day Forward Prices</div>', unsafe_allow_html=True)
    fig_backtest = plot_backtest_actual_vs_predicted(backtest_df, sel_sym, sel_model)
    st.plotly_chart(fig_backtest, use_container_width=True)
    
    # Residual Error Distribution
    st.markdown('<div class="section-header">Residual Error Distribution</div>', unsafe_allow_html=True)
    fig_res = plot_residual_distribution(backtest_df, sel_sym, sel_model)
    st.plotly_chart(fig_res, use_container_width=True)
    
    # Granular Backtest Data Table
    st.markdown('<div class="section-header">Granular Test Session Log</div>', unsafe_allow_html=True)
    st.dataframe(
        sub_df[['Date', 'Actual_Future_Close', 'Predicted_Future_Close', 'Error', 'Abs_Error', 'Pct_Error']].style.format({
            'Actual_Future_Close': '₹{:,.2f}',
            'Predicted_Future_Close': '₹{:,.2f}',
            'Error': '₹{:+,.2f}',
            'Abs_Error': '₹{:,.2f}',
            'Pct_Error': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

if __name__ == "__main__":
    render_page()
