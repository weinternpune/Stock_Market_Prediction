"""
05_sector_analysis.py
---------------------
Streamlit Dashboard Page 5: NIFTY 50 Industry Comparative Analytics.
Aggregates constituent performance across industries: forecast returns,
constituent comparisons, and model-based analytics for NIFTY 50 stocks.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

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
    from app.components.charts import CHART_THEME
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.cards import metric_card
    from components.charts import CHART_THEME

def get_predictions():
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("🏭 NIFTY 50 Industry Comparative Analytics")
    st.markdown("##### Industry-level forecast returns, constituent comparisons, and model-based analytics for NIFTY 50 stocks.")
    st.caption("Model selection criterion: Lowest holdout RMSE. Industry averages are equal-weighted across constituent stocks.")
    
    df = get_predictions()
    if df is None:
        st.warning("⚠️ Prediction data not found. Please run the modeling pipeline first.")
        return
        
    # Industry Aggregation (Equal-weighted across constituent stocks)
    sector_agg = df.groupby('Industry').agg(
        Companies=('Symbol', 'count'),
        Avg_Return=('Expected_Return_Pct', 'mean'),
        Median_Return=('Expected_Return_Pct', 'median'),
        Positive_Count=('Expected_Return_Pct', lambda x: (x > 0).sum()),
        Best_Stock=('Symbol', lambda s: df.loc[s.index].sort_values('Expected_Return_Pct', ascending=False)['Symbol'].iloc[0]),
        Best_Return=('Expected_Return_Pct', 'max'),
        Worst_Stock=('Symbol', lambda s: df.loc[s.index].sort_values('Expected_Return_Pct', ascending=True)['Symbol'].iloc[0]),
        Worst_Return=('Expected_Return_Pct', 'min')
    ).reset_index()
    
    sector_agg['Avg_Return'] = sector_agg['Avg_Return'].round(2)
    sector_agg['Median_Return'] = sector_agg['Median_Return'].round(2)
    sector_agg['Best_Return'] = sector_agg['Best_Return'].round(2)
    sector_agg['Worst_Return'] = sector_agg['Worst_Return'].round(2)
    
    # -------------------------------------------------------------
    # 1. INDUSTRY FORECAST RETURNS
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Industry Forecast Returns</div>', unsafe_allow_html=True)
    
    fig_sector_bar = px.bar(
        sector_agg.sort_values('Avg_Return', ascending=True),
        x='Avg_Return',
        y='Industry',
        orientation='h',
        color='Avg_Return',
        color_continuous_scale='RdYlGn',
        title="Average Forecasted 21-Trading-Day Return (%) by Industry",
        labels={'Avg_Return': 'Avg Forecasted Return (%)', 'Industry': 'Industry'},
        template=CHART_THEME
    )
    fig_sector_bar.update_layout(height=460, margin=dict(l=40, r=40, t=50, b=40))
    fig_sector_bar.update_traces(
        hovertemplate="<b>%{y}</b><br>Avg Forecasted Return: %{x:+.2f}%<extra></extra>"
    )
    st.plotly_chart(fig_sector_bar, use_container_width=True)
    
    # -------------------------------------------------------------
    # 2. INDUSTRY FORECAST SUMMARY
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Industry Forecast Summary</div>', unsafe_allow_html=True)
    
    summary_table = pd.DataFrame({
        'Industry': sector_agg['Industry'],
        'Companies': sector_agg['Companies'],
        'Avg Forecasted Return': sector_agg['Avg_Return'],
        'Median Forecasted Return': sector_agg['Median_Return'],
        'Positive Forecasts': sector_agg.apply(lambda r: f"{r['Positive_Count']} / {r['Companies']}", axis=1),
        'Highest Forecasted Return': sector_agg.apply(lambda r: f"{r['Best_Stock']} ({r['Best_Return']:+.2f}%)", axis=1),
        'Lowest Forecasted Return': sector_agg.apply(lambda r: f"{r['Worst_Stock']} ({r['Worst_Return']:+.2f}%)", axis=1)
    }).sort_values('Avg Forecasted Return', ascending=False)
    
    st.dataframe(
        summary_table.style.format({
            'Avg Forecasted Return': '{:+.2f}%',
            'Median Forecasted Return': '{:+.2f}%'
        }).background_gradient(subset=['Avg Forecasted Return'], cmap='RdYlGn', vmin=-5, vmax=10),
        use_container_width=True,
        hide_index=True
    )
    
    csv_industry = summary_table.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Industry Forecast Summary CSV",
        data=csv_industry,
        file_name="nifty50_industry_forecast_summary.csv",
        mime="text/csv"
    )
    
    # -------------------------------------------------------------
    # 3. INDUSTRY CONSTITUENT DRILL-DOWN
    # -------------------------------------------------------------
    st.markdown('<div class="section-header">Industry Constituent Drill-Down</div>', unsafe_allow_html=True)
    
    industries = sorted(df['Industry'].unique().tolist())
    default_idx = industries.index("Automobile and Auto Components") if "Automobile and Auto Components" in industries else 0
    selected_ind = st.selectbox("Select Industry:", industries, index=default_idx)
    
    ind_stocks = df[df['Industry'] == selected_ind].sort_values('Expected_Return_Pct', ascending=False)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Companies", f"{len(ind_stocks)} Companies", subtext="NIFTY 50 Universe")
    with c2:
        avg_ret = ind_stocks['Expected_Return_Pct'].mean()
        metric_card("Avg Forecasted Return", f"{avg_ret:+.2f}%", is_positive=(avg_ret >= 0), subtext="21-Trading-Day Horizon")
    with c3:
        best_sym = ind_stocks.iloc[0]['Symbol']
        best_ret = ind_stocks.iloc[0]['Expected_Return_Pct']
        metric_card("Highest Forecasted Return", f"{best_sym} ({best_ret:+.2f}%)", is_positive=(best_ret >= 0), subtext="Selected Model Forecast")
        
    st.markdown(f"##### Constituents in {selected_ind}:")
    display_sub = ind_stocks[[
        'Symbol', 'Company', 'Current_Price', 'Predicted_21D_Price',
        'Expected_Return_Pct', 'Best_Model', 'RMSE', 'MAPE'
    ]].copy()
    
    display_sub.rename(columns={
        'Current_Price': 'Current Price',
        'Predicted_21D_Price': 'Forecasted Price (+21 Trading Days)',
        'Expected_Return_Pct': 'Forecasted Return (%)',
        'Best_Model': 'Selected Model',
        'RMSE': 'RMSE',
        'MAPE': 'MAPE'
    }, inplace=True)
    
    st.dataframe(
        display_sub.style.format({
            'Current Price': '₹{:,.2f}',
            'Forecasted Price (+21 Trading Days)': '₹{:,.2f}',
            'Forecasted Return (%)': '{:+.2f}%',
            'RMSE': '₹{:,.2f}',
            'MAPE': '{:.2f}%'
        }).background_gradient(subset=['Forecasted Return (%)'], cmap='RdYlGn', vmin=-10, vmax=15),
        use_container_width=True,
        hide_index=True
    )
    
    csv_drilldown = display_sub.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download {selected_ind} Constituents CSV",
        data=csv_drilldown,
        file_name=f"{selected_ind.lower().replace(' ', '_')}_constituents.csv",
        mime="text/csv"
    )
    
    # Methodology & Disclaimer Note Box
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 10px; padding: 14px 18px; margin-top: 18px;">
        <span style="color: #94a3b8; font-size: 0.83rem; line-height: 1.5;">
            <strong>NOTE:</strong> Model selection criterion: Lowest holdout RMSE across competing architectures. Industry averages are equal-weighted across constituent stocks. Forecasts are model-generated estimates based on historical data and are subject to model error and market uncertainty.
        </span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
