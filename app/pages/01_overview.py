"""
01_overview.py
--------------
Streamlit Dashboard Page 1: Overview & Market Sentiment.
Displays top-level market KPIs, model win distribution, Top 5 gainers/losers,
and neutral automated market summary.
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

from config.project_config import PREDICTIONS_CSV, METRICS_DIR
try:
    from app.components.styling import apply_custom_styles
    from app.components.cards import metric_card, render_stock_pill
    from app.components.charts import plot_model_win_counts
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.cards import metric_card, render_stock_pill
    from components.charts import plot_model_win_counts

from src.utilities.io_helpers import load_json

def render_page():
    apply_custom_styles()
    
    st.title("🏠 NIFTY 50 Stock Prediction & Analytics")
    st.markdown("##### Historical analysis, machine-learning forecasts, model evaluation and comparative analytics.")
    
    # Load cached prediction results
    if not PREDICTIONS_CSV.exists():
        st.warning("⚠️ Prediction results not yet generated. Please execute the modeling pipeline first.")
        return
        
    df = pd.read_csv(PREDICTIONS_CSV)
    summary = load_json(METRICS_DIR / "master_model_metrics.json")
    
    # KPI Row 1: Dataset & Coverage
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Constituents", f"{len(df)} Stocks", subtext="NIFTY 50 Universe")
    with c2:
        metric_card("Historical Data Period", "5 Years", subtext="2021-09-15 → 2026-09-11")
    with c3:
        metric_card("Data As Of", str(df['Latest_Date'].max()), subtext="Market Close")
    with c4:
        metric_card("Forecast Horizon", "21 Trading Days", subtext="~1 Month Ahead")
    with c5:
        best_overall = max(summary.get("model_win_counts", {"ARIMA": 1}).items(), key=lambda x: x[1])[0]
        metric_card("Model Selection Summary", best_overall, subtext="Model with Lowest Holdout RMSE")
        
    st.markdown('<div class="section-header">Market Forecast Summary</div>', unsafe_allow_html=True)
    
    # KPI Row 2: Prediction Summary (Accounting for all 50 constituent stocks with Positive, Negative, and Neutral)
    m1, m2, m3, m4, m5 = st.columns(5)
    avg_ret = df['Expected_Return_Pct'].mean()
    med_ret = df['Expected_Return_Pct'].median()
    n_pos = (df['Expected_Return_Pct'] > 0).sum()
    n_neg = (df['Expected_Return_Pct'] < 0).sum()
    n_neu = (df['Expected_Return_Pct'] == 0).sum()
    
    with m1:
        metric_card("Forecasted Positive Return", f"{n_pos} / {len(df)}", delta=f"{(n_pos/len(df))*100:.1f}%", is_positive=True)
    with m2:
        metric_card("Forecasted Negative Return", f"{n_neg} / {len(df)}", delta=f"{(n_neg/len(df))*100:.1f}%", is_positive=False)
    with m3:
        metric_card("Forecasted Neutral Return", f"{n_neu} / {len(df)}", delta=f"{(n_neu/len(df))*100:.1f}%", is_positive=None, subtext="Flat (0.00% Return)")
    with m4:
        metric_card("Mean Forecast Return", f"{avg_ret:+.2f}%", is_positive=(avg_ret >= 0))
    with m5:
        metric_card("Median Forecast Return", f"{med_ret:+.2f}%", is_positive=(med_ret >= 0))
        
    # Automated Neutral Summary
    st.info(
        f"📊 **FORECAST SUMMARY**\n\n"
        f"**{len(df)}/{len(df)} stocks evaluated** — "
        f"**{n_pos} positive** | **{n_neg} negative** | **{n_neu} neutral**\n\n"
        f"Forecasts are model-generated estimates based on historical data and are subject to model error and market uncertainty. "
        f"Median forecasted return: **{med_ret:+.2f}%** (Mean: **{avg_ret:+.2f}%**)."
    )
    
    st.markdown('<div class="section-header">Highest & Lowest Forecasted Returns</div>', unsafe_allow_html=True)
    col_gainers, col_losers = st.columns(2)
    
    sorted_df = df.sort_values(by="Expected_Return_Pct", ascending=False).reset_index(drop=True)
    
    with col_gainers:
        st.markdown("##### 🚀 5 Highest Forecasted Returns")
        for rank, row in sorted_df.head(5).iterrows():
            render_stock_pill(
                rank=rank + 1,
                symbol=row['Symbol'],
                company=row['Company'],
                expected_return=row['Expected_Return_Pct'],
                current_p=row['Current_Price'],
                pred_p=row['Predicted_21D_Price'],
                model=row['Best_Model']
            )
            
    with col_losers:
        st.markdown("##### 🔻 5 Lowest Forecasted Returns")
        bottom_5 = sorted_df.tail(5).iloc[::-1].reset_index(drop=True)
        for rank, row in bottom_5.iterrows():
            render_stock_pill(
                rank=len(df) - rank,
                symbol=row['Symbol'],
                company=row['Company'],
                expected_return=row['Expected_Return_Pct'],
                current_p=row['Current_Price'],
                pred_p=row['Predicted_21D_Price'],
                model=row['Best_Model']
            )
            
    st.markdown('<div class="section-header">Model Selection by Holdout RMSE</div>', unsafe_allow_html=True)
    win_counts = summary.get("model_win_counts", df['Best_Model'].value_counts().to_dict())
    st.plotly_chart(plot_model_win_counts(win_counts), use_container_width=True)
    
    # Methodology & Data Disclaimer Box
    st.markdown('<div class="section-header">Methodology & Framework</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 12px; padding: 18px 22px; margin-top: 6px; margin-bottom: 20px;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #f1f5f9; margin-bottom: 8px;">
            📌 Methodology
        </div>
        <ul style="margin: 0 0 12px 18px; padding: 0; color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
            <li><strong>21 trading-day forecast:</strong> Forecasts are generated using historical daily stock data over a forward 21-session horizon (~1 calendar month).</li>
            <li><strong>Time-aware holdout evaluation:</strong> Evaluated using chronologically isolated holdout test partitions to prevent future data leakage.</li>
            <li><strong>Lowest holdout RMSE used for model selection:</strong> The selected model for each stock is chosen strictly based on the lowest holdout test RMSE across competing architectures.</li>
        </ul>
        <div style="font-size: 0.82rem; color: #94a3b8; border-top: 1px solid rgba(148, 163, 184, 0.15); padding-top: 8px;">
            <strong>Note:</strong> Forecasts are statistical/model estimates and are not guaranteed future prices or returns.
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
