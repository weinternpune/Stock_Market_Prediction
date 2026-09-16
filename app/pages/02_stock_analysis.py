"""
02_stock_analysis.py
--------------------
Streamlit Dashboard Page 2: Individual Stock Deep Dive.
Provides company information, interactive candlestick charts with SMAs and Bollinger Bands,
volume, technical indicators (MACD, RSI), 21-day forecast KPI cards, and cross-model comparison.
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

from config.project_config import (
    PROCESSED_DATA_PARQUET,
    PREDICTIONS_CSV,
    MODEL_COMPARISON_CSV,
    MODELS_DIR
)
try:
    from app.components.styling import apply_custom_styles
    from app.components.cards import metric_card
    from app.components.charts import plot_price_and_indicators
except ModuleNotFoundError:
    from components.styling import apply_custom_styles
    from components.cards import metric_card
    from components.charts import plot_price_and_indicators

from src.utilities.io_helpers import load_dataframe, load_json

@st.cache_data
def get_stock_data():
    if not PROCESSED_DATA_PARQUET.exists():
        return None
    df = load_dataframe(PROCESSED_DATA_PARQUET)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def get_predictions_data():
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)

@st.cache_data
def get_comparison_data():
    if not MODEL_COMPARISON_CSV.exists():
        return None
    return pd.read_csv(MODEL_COMPARISON_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("📈 Individual Stock Analysis & Forecasting")
    st.markdown("##### Comprehensive technical profile, momentum indicators, and 21-day model projections.")
    
    pred_df = get_predictions_data()
    full_df = get_stock_data()
    comp_df = get_comparison_data()
    
    if pred_df is None or full_df is None:
        st.warning("⚠️ Prediction and feature datasets not ready. Please run the modeling pipeline first.")
        return
        
    symbols = sorted(pred_df['Symbol'].unique())
    
    col_sel, col_empty = st.columns([2, 3])
    with col_sel:
        selected_symbol = st.selectbox("Select NIFTY 50 Stock:", symbols, index=0)
        
    stock_pred = pred_df[pred_df['Symbol'] == selected_symbol].iloc[0]
    stock_hist = full_df[full_df['Symbol'] == selected_symbol].sort_values('Date')
    
    # Company Header Banner (Theme-adaptive styling)
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.6); padding: 18px 22px; border-radius: 12px; border: 1px solid rgba(148, 163, 184, 0.18); margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: #f8fafc; font-weight: 800;">{stock_pred['Company']} ({selected_symbol})</h3>
                <div style="margin-top: 8px;">
                    <span class="badge-sector">Sector: {stock_pred['Sector']}</span>
                    <span class="badge-sector" style="margin-left: 6px;">Industry: {stock_pred['Industry']}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Latest Observed Close ({stock_pred['Latest_Date']})</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #f8fafc; line-height: 1.2;">₹{stock_pred['Current_Price']:,.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction Section KPI Cards
    st.markdown('<div class="section-header">21-Trading-Day Forecast Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    
    is_pos = stock_pred['Expected_Return_Pct'] >= 0
    delta_str = f"{stock_pred['Expected_Return_Pct']:+.2f}% (₹{stock_pred['Expected_Change']:+,.2f})"
    
    with k1:
        metric_card("Latest Market Close", f"₹{stock_pred['Current_Price']:,.2f}", subtext=f"As of {stock_pred['Latest_Date']}")
    with k2:
        metric_card(
            "Projected Close (+21D)",
            f"₹{stock_pred['Predicted_21D_Price']:,.2f}",
            delta=delta_str,
            is_positive=is_pos,
            subtext=f"Selected Model: {stock_pred['Best_Model']}"
        )
    with k3:
        metric_card(
            "90% Prediction Interval",
            f"₹{stock_pred['Lower_Bound_90']:,.1f} – ₹{stock_pred['Upper_Bound_90']:,.1f}",
            subtext="Based on empirical test residual std"
        )
    with k4:
        metric_card(
            "Model Accuracy (RMSE)",
            f"₹{stock_pred['RMSE']:,.2f}",
            subtext=f"Test MAE: ₹{stock_pred['MAE']:,.2f} (MAPE: {stock_pred['MAPE']:.2f}%)"
        )
        
    # Technical Chart
    st.markdown('<div class="section-header">Historical Price Action & Technical Indicators</div>', unsafe_allow_html=True)
    fig_tech = plot_price_and_indicators(stock_hist, selected_symbol)
    st.plotly_chart(fig_tech, use_container_width=True)
    
    # 4-Model Comparison Table for this stock
    st.markdown('<div class="section-header">Model Performance & Prediction Breakdown</div>', unsafe_allow_html=True)
    
    stock_models = comp_df[comp_df['Symbol'] == selected_symbol].copy()
    if not stock_models.empty:
        # Load stock artifact for predicted future prices per model
        stock_artifact_path = MODELS_DIR / selected_symbol / "metrics.json"
        future_dict = {}
        if stock_artifact_path.exists():
            art = load_json(stock_artifact_path)
            future_dict = art.get("future_forecasts", {})
            
        stock_models['Predicted_21D_Price'] = stock_models['Model'].map(lambda m: f"₹{future_dict.get(m, stock_pred['Current_Price']):,.2f}")
        stock_models['Exp_Return_Pct'] = stock_models['Model'].map(
            lambda m: f"{((future_dict.get(m, stock_pred['Current_Price']) - stock_pred['Current_Price']) / stock_pred['Current_Price']) * 100:+.2f}%"
        )
        
        display_cols = ['Model', 'Predicted_21D_Price', 'Exp_Return_Pct', 'RMSE', 'MAE', 'MAPE', 'Directional_Accuracy', 'Is_Best_Model']
        st.dataframe(
            stock_models[display_cols].style.highlight_max(subset=['Is_Best_Model'], color='#dcfce7'),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Model comparison details loading...")

if __name__ == "__main__":
    render_page()
