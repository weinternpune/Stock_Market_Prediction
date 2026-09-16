"""
05_sector_analysis.py
---------------------
Streamlit Dashboard Page 5: Sector & Industry Performance Analysis.
Aggregates constituent performance across industries: return expectations,
historical volatility, constituent counts, and sector champion rankings.
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

@st.cache_data
def get_predictions():
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("🏭 Sector & Industry Comparative Analytics")
    st.markdown("##### Sector-wide return projections, volatility profiles, and intra-industry constituent leadership.")
    
    df = get_predictions()
    if df is None:
        st.warning("⚠️ Prediction data not found. Please run the modeling pipeline first.")
        return
        
    # Sector Aggregation
    sector_agg = df.groupby('Industry').agg(
        Company_Count=('Symbol', 'count'),
        Avg_Expected_Return=('Expected_Return_Pct', 'mean'),
        Median_Expected_Return=('Expected_Return_Pct', 'median'),
        Positive_Count=('Expected_Return_Pct', lambda x: (x > 0).sum()),
        Best_Stock=('Symbol', lambda s: df.loc[s.index].sort_values('Expected_Return_Pct', ascending=False)['Symbol'].iloc[0]),
        Best_Stock_Return=('Expected_Return_Pct', 'max'),
        Worst_Stock=('Symbol', lambda s: df.loc[s.index].sort_values('Expected_Return_Pct', ascending=True)['Symbol'].iloc[0]),
        Worst_Stock_Return=('Expected_Return_Pct', 'min')
    ).reset_index()
    
    sector_agg['Avg_Expected_Return'] = sector_agg['Avg_Expected_Return'].round(2)
    sector_agg['Best_Stock_Return'] = sector_agg['Best_Stock_Return'].round(2)
    sector_agg['Worst_Stock_Return'] = sector_agg['Worst_Stock_Return'].round(2)
    
    st.markdown('<div class="section-header">Sector Return Expectations</div>', unsafe_allow_html=True)
    
    fig_sector_bar = px.bar(
        sector_agg.sort_values('Avg_Expected_Return', ascending=True),
        x='Avg_Expected_Return',
        y='Industry',
        orientation='h',
        color='Avg_Expected_Return',
        color_continuous_scale='RdYlGn',
        title="Average Projected 21-Day Return (%) by Industry",
        template=CHART_THEME
    )
    fig_sector_bar.update_layout(height=450, margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig_sector_bar, use_container_width=True)
    
    # Sector League Table
    st.markdown('<div class="section-header">Industry Aggregation Table</div>', unsafe_allow_html=True)
    st.dataframe(
        sector_agg.style.format({
            'Avg_Expected_Return': '{:+.2f}%',
            'Median_Expected_Return': '{:+.2f}%',
            'Best_Stock_Return': '{:+.2f}%',
            'Worst_Stock_Return': '{:+.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Interactive Industry Drill-Down
    st.markdown('<div class="section-header">Industry Constituent Drill-Down</div>', unsafe_allow_html=True)
    
    industries = sorted(df['Industry'].unique().tolist())
    selected_ind = st.selectbox("Select Industry to Inspect:", industries, index=0)
    
    ind_stocks = df[df['Industry'] == selected_ind].sort_values('Expected_Return_Pct', ascending=False)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Industry Stocks", f"{len(ind_stocks)} Companies")
    with c2:
        metric_card("Avg Forecast Return", f"{ind_stocks['Expected_Return_Pct'].mean():+.2f}%", is_positive=(ind_stocks['Expected_Return_Pct'].mean() >= 0))
    with c3:
        best_sym = ind_stocks.iloc[0]['Symbol']
        best_ret = ind_stocks.iloc[0]['Expected_Return_Pct']
        metric_card("Top Performer", f"{best_sym} ({best_ret:+.2f}%)", is_positive=(best_ret >= 0))
        
    st.markdown(f"##### Constituents in {selected_ind}:")
    display_sub = ind_stocks[['Symbol', 'Company', 'Current_Price', 'Predicted_21D_Price', 'Expected_Return_Pct', 'Best_Model', 'RMSE', 'MAPE']]
    st.dataframe(
        display_sub.style.format({
            'Current_Price': '₹{:,.2f}',
            'Predicted_21D_Price': '₹{:,.2f}',
            'Expected_Return_Pct': '{:+.2f}%',
            'RMSE': '₹{:,.2f}',
            'MAPE': '{:.2f}%'
        }),
        use_container_width=True,
        hide_index=True
    )

if __name__ == "__main__":
    render_page()
