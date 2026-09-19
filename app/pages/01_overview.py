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
    st.markdown("##### Historical analysis, machine-learning forecasts and comparative analytics for NIFTY 50 constituent stocks.")
    
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
        metric_card("Historical Span", "5 Years", subtext="2021 – 2026 Daily")
    with c3:
        metric_card("Latest Date", str(df['Latest_Date'].max()), subtext="Market Close")
    with c4:
        metric_card("Horizon", "21 Days", subtext="Trading Sessions (~1 Mo)")
    with c5:
        best_overall = max(summary.get("model_win_counts", {"ARIMA": 1}).items(), key=lambda x: x[1])[0]
        metric_card("Top Model", best_overall, subtext="By Lowest Test RMSE")
        
    st.markdown('<div class="section-header">Market Outlook Summary</div>', unsafe_allow_html=True)
    
    # KPI Row 2: Prediction Summary (Accounting for all 50 constituent stocks with Positive, Negative, and Neutral)
    m1, m2, m3, m4, m5 = st.columns(5)
    avg_ret = df['Expected_Return_Pct'].mean()
    med_ret = df['Expected_Return_Pct'].median()
    n_pos = (df['Expected_Return_Pct'] > 0).sum()
    n_neg = (df['Expected_Return_Pct'] < 0).sum()
    n_neu = (df['Expected_Return_Pct'] == 0).sum()
    
    with m1:
        metric_card("Predicted Up (+ve)", f"{n_pos} / {len(df)}", delta=f"{(n_pos/len(df))*100:.1f}%", is_positive=True)
    with m2:
        metric_card("Predicted Down (-ve)", f"{n_neg} / {len(df)}", delta=f"{(n_neg/len(df))*100:.1f}%", is_positive=False)
    with m3:
        metric_card("Predicted Neutral", f"{n_neu} / {len(df)}", delta=f"{(n_neu/len(df))*100:.1f}%", is_positive=None, subtext="Flat (0.00% Return)")
    with m4:
        metric_card("Mean Forecast Return", f"{avg_ret:+.2f}%", is_positive=(avg_ret >= 0))
    with m5:
        metric_card("Median Forecast Return", f"{med_ret:+.2f}%", is_positive=(med_ret >= 0))
        
    # Automated Neutral Summary
    st.info(
        f"📊 **Automated Market Synthesis**: Out of {len(df)} evaluated NIFTY 50 stocks, **{n_pos} stocks** have predicted positive returns "
        f"(bullish, > 0.00%), **{n_neg} stocks** are projected lower (bearish, < 0.00%), and **{n_neu} stock{'s' if n_neu != 1 else ''}** "
        f"are projected neutral / flat (0.00% return). Exactly 100% of the constituent universe is accounted for. "
        f"The median expected return is **{med_ret:+.2f}%**. These projections represent model central tendencies subject to historical error bounds."
    )
    
    st.markdown('<div class="section-header">Extreme Expected Return Outliers</div>', unsafe_allow_html=True)
    col_gainers, col_losers = st.columns(2)
    
    sorted_df = df.sort_values(by="Expected_Return_Pct", ascending=False).reset_index(drop=True)
    
    with col_gainers:
        st.markdown("##### 🚀 Top 5 Predicted Gainers (21-Day)")
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
        st.markdown("##### 🔻 Bottom 5 Predicted Performers (21-Day)")
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
            
    st.markdown('<div class="section-header">Model Architecture Distribution</div>', unsafe_allow_html=True)
    win_counts = summary.get("model_win_counts", df['Best_Model'].value_counts().to_dict())
    st.plotly_chart(plot_model_win_counts(win_counts), use_container_width=True)

if __name__ == "__main__":
    render_page()
