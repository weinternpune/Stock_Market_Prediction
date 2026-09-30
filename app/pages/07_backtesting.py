"""
07_backtesting.py
-----------------
Streamlit Dashboard Page 7: Historical Backtesting & Out-of-Sample Validation.
Evaluates model forecasts on historical holdout data using rolling out-of-sample backtesting,
with actual vs. forecasted forward price curves, residual distributions, and granular results.
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
    from app.components.charts import plot_backtest_actual_vs_predicted, plot_residual_distribution
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.cards import metric_card
    from components.charts import plot_backtest_actual_vs_predicted, plot_residual_distribution

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
    st.markdown("##### Evaluation of model forecasts on historical holdout data using rolling out-of-sample backtesting.")
    
    backtest_df = get_backtest_data()
    comp_df = get_comparison_data()
    
    if backtest_df is None or comp_df is None:
        st.warning("⚠️ Backtesting predictions not found. Please execute the modeling pipeline first.")
        return
        
    symbols = sorted(backtest_df['Symbol'].unique())
    models = sorted(backtest_df['Model'].unique())
    
    # -------------------------------------------------------------
    # BACKTEST PARAMETERS
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Backtest Parameters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        def_sym_idx = symbols.index("ADANIENT") if "ADANIENT" in symbols else 0
        sel_sym = st.selectbox("Select Stock:", symbols, index=def_sym_idx)
    with c2:
        def_mod_idx = models.index("ARIMA") if "ARIMA" in models else 0
        sel_model = st.selectbox("Select Forecasting Model:", models, index=def_mod_idx)
        
    sub_df = backtest_df[(backtest_df['Symbol'] == sel_sym) & (backtest_df['Model'] == sel_model)].sort_values('Date')
    
    if sub_df.empty:
        st.warning("No backtesting data found for this combination.")
        return
        
    try:
        start_date = pd.to_datetime(sub_df['Date'].min()).strftime('%d-%b-%Y')
        end_date = pd.to_datetime(sub_df['Date'].max()).strftime('%d-%b-%Y')
    except Exception:
        start_date = str(sub_df['Date'].min())
        end_date = str(sub_df['Date'].max())
        
    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        metric_card("Forecast Horizon", "21 Trading Days", subtext="Forward Projection Window")
    with cp2:
        metric_card("Backtest Period", f"{start_date} – {end_date}", subtext="Historical Holdout Window")
    with cp3:
        metric_card("Validation Method", "Rolling-Origin / Walk-Forward", subtext="Chronological Time Series Split")
        
    # Methodological Context Callout
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 14px 18px; margin: 18px 0;">
        <span style="color: #94a3b8; font-size: 0.83rem; line-height: 1.5;">
            <strong>Methodological Notice on Rolling-Origin Validation:</strong><br>
            • <strong>Rolling Forecast Origins</strong>: The model generates 21-trading-day forward projections across 120 successive daily trading sessions.<br>
            • <strong>Overlapping Evaluation Windows</strong>: Consecutive 21-day projection horizons naturally overlap in time. Metrics reflect the cumulative performance of rolling-origin forecasts rather than 120 isolated independent experiments.<br>
            • <strong>Model Selection Criterion</strong>: Production forecasting architectures are objectively selected by lowest holdout test RMSE across this validation period.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # -------------------------------------------------------------
    # TEST SUMMARY
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Test Summary</div>', unsafe_allow_html=True)
    y_true = sub_df['Actual_Future_Close'].values
    y_pred = sub_df['Predicted_Future_Close'].values
    eval_metrics = compute_evaluation_metrics(y_true, y_pred)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Test Observations", f"{len(sub_df)}", subtext="Rolling Forecast Origins")
    with m2:
        metric_card("Root Mean Squared Error (RMSE)", f"₹{eval_metrics['rmse']:,.2f}", subtext="Penalty for large deviations")
    with m3:
        metric_card("Mean Absolute Error (MAE)", f"₹{eval_metrics['mae']:,.2f}", subtext="Average absolute deviation")
    with m4:
        metric_card("Mean Absolute Percentage Error (MAPE)", f"{eval_metrics['mape']:.2f}%", subtext="Scale-free error rate")
        
    # -------------------------------------------------------------
    # ACTUAL VS FORECASTED 21-TRADING-DAY FORWARD CLOSE
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Actual vs. Forecasted 21-Trading-Day Forward Close</div>', unsafe_allow_html=True)
    fig_backtest = plot_backtest_actual_vs_predicted(backtest_df, sel_sym, sel_model)
    st.plotly_chart(fig_backtest, use_container_width=True)
    
    # -------------------------------------------------------------
    # FORECAST ERROR DISTRIBUTION
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Forecast Error Distribution</div>', unsafe_allow_html=True)
    fig_res = plot_residual_distribution(backtest_df, sel_sym, sel_model)
    st.plotly_chart(fig_res, use_container_width=True)
    
    # -------------------------------------------------------------
    # DETAILED BACKTEST RESULTS
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Detailed Backtest Results</div>', unsafe_allow_html=True)
    
    col_filter, col_info = st.columns([2, 3])
    with col_filter:
        disp_limit = st.selectbox(
            "Display Limit:",
            ["All 120 Observations", "First 10 Observations", "First 20 Observations", "Last 20 Observations"],
            index=0
        )
        
    sub_table = sub_df[['Date', 'Actual_Future_Close', 'Predicted_Future_Close', 'Error', 'Abs_Error', 'Pct_Error']].copy()
    sub_table.rename(columns={
        'Date': 'Origin Date',
        'Actual_Future_Close': 'Actual Future Close',
        'Predicted_Future_Close': 'Forecasted Future Close',
        'Error': 'Residual (Actual − Forecast)',
        'Abs_Error': 'Absolute Error (₹)',
        'Pct_Error': 'Absolute Percentage Error (%)'
    }, inplace=True)
    
    if disp_limit == "First 10 Observations":
        display_df = sub_table.head(10)
    elif disp_limit == "First 20 Observations":
        display_df = sub_table.head(20)
    elif disp_limit == "Last 20 Observations":
        display_df = sub_table.tail(20)
    else:
        display_df = sub_table
        
    st.markdown(f"**Displaying {len(display_df)} of {len(sub_table)} backtest observations**")
    
    st.dataframe(
        display_df.style.format({
            'Actual Future Close': '₹{:,.2f}',
            'Forecasted Future Close': '₹{:,.2f}',
            'Residual (Actual − Forecast)': '₹{:+,.2f}',
            'Absolute Error (₹)': '₹{:,.2f}',
            'Absolute Percentage Error (%)': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Download Full Backtest CSV
    csv_backtest = sub_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Backtest CSV",
        data=csv_backtest,
        file_name=f"{sel_sym.lower()}_{sel_model.lower()}_backtest_results.csv",
        mime="text/csv"
    )
    
    # Bottom Note Box
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 14px 18px; margin-top: 18px;">
        <span style="color: #94a3b8; font-size: 0.83rem; line-height: 1.5;">
            <strong>NOTE:</strong> Model forecasts on historical holdout data are retrospective statistical evaluations. Past holdout performance does not guarantee future prediction accuracy under changing market conditions.
        </span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
