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

def get_predictions_data():
    if not PREDICTIONS_CSV.exists():
        return None
    return pd.read_csv(PREDICTIONS_CSV)

def get_comparison_data():
    if not MODEL_COMPARISON_CSV.exists():
        return None
    return pd.read_csv(MODEL_COMPARISON_CSV)

def render_page():
    apply_custom_styles()
    
    st.title("📈 Individual Stock Analysis & Forecasting")
    st.markdown("##### Comprehensive technical profile, momentum indicators, and 21-trading-day model forecasts.")
    
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
    
    # Format date as '11 Sep 2026'
    try:
        dt_val = pd.to_datetime(stock_pred['Latest_Date']).strftime('%d %b %Y')
    except Exception:
        dt_val = str(stock_pred['Latest_Date'])
        
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
                <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Data As Of: {dt_val}</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #f8fafc; line-height: 1.2;">₹{stock_pred['Current_Price']:,.2f}</div>
                <div style="font-size: 0.75rem; color: #64748b;">Latest Observed Close</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction Section KPI Cards
    st.markdown('<div class="section-header">21-Trading-Day Forecast</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    
    is_pos = stock_pred['Expected_Return_Pct'] > 0
    is_neu = stock_pred['Expected_Return_Pct'] == 0
    delta_flag = None if is_neu else is_pos
    
    with k1:
        metric_card(
            "Forecasted Close",
            f"₹{stock_pred['Predicted_21D_Price']:,.2f}",
            delta=f"{stock_pred['Expected_Return_Pct']:+.2f}%",
            is_positive=delta_flag,
            subtext=f"Selected Model: {stock_pred['Best_Model']}"
        )
    with k2:
        metric_card(
            "Forecasted Return",
            f"{stock_pred['Expected_Return_Pct']:+.2f}%",
            delta=f"₹{stock_pred['Expected_Change']:+,.2f}",
            is_positive=delta_flag,
            subtext="21-Trading-Day Horizon"
        )
    with k3:
        metric_card(
            "Estimated 90% Forecast Range",
            f"₹{int(round(stock_pred['Lower_Bound_90'])):,} – ₹{int(round(stock_pred['Upper_Bound_90'])):,}",
            subtext="Empirical Holdout Residual Bounds"
        )
    with k4:
        metric_card(
            "Holdout RMSE",
            f"₹{stock_pred['RMSE']:,.2f}",
            subtext=f"Holdout MAE: ₹{stock_pred['MAE']:,.2f} (MAPE: {stock_pred['MAPE']:.2f}%)"
        )
        
    # Technical Chart
    st.markdown('<div class="section-header">Historical Price Action & Technical Indicators</div>', unsafe_allow_html=True)
    fig_tech = plot_price_and_indicators(stock_hist, selected_symbol)
    st.plotly_chart(fig_tech, use_container_width=True)
    
    # 4-Model Comparison Table for this stock
    st.markdown('<div class="section-header">Model Performance & Forecast Comparison</div>', unsafe_allow_html=True)
    
    stock_models = comp_df[comp_df['Symbol'] == selected_symbol].copy()
    if not stock_models.empty:
        # Load stock artifact for predicted future prices per model
        stock_artifact_path = MODELS_DIR / selected_symbol / "metrics.json"
        future_dict = {}
        if stock_artifact_path.exists():
            art = load_json(stock_artifact_path)
            future_dict = art.get("future_forecasts", {})
            
        model_order = {"Baseline": 0, "ARIMA": 1, "XGBoost": 2, "LSTM": 3}
        stock_models['sort_idx'] = stock_models['Model'].map(lambda m: model_order.get(m, 99))
        stock_models = stock_models.sort_values('sort_idx').drop(columns=['sort_idx'])
        
        curr_p = stock_pred['Current_Price']
        
        stock_models['Forecasted Price (+21 Trading Days)'] = stock_models['Model'].map(
            lambda m: f"₹{future_dict.get(m, curr_p):,.2f}"
        )
        
        def _calc_ret(m):
            f_price = future_dict.get(m, curr_p)
            ret = ((f_price - curr_p) / curr_p) * 100.0
            return f"{ret:+.2f}%"
        stock_models['Forecasted Return (%)'] = stock_models['Model'].map(_calc_ret)
        
        stock_models['Holdout RMSE (₹)'] = stock_models['RMSE'].map(lambda x: f"₹{x:,.2f}")
        stock_models['Holdout MAE (₹)'] = stock_models['MAE'].map(lambda x: f"₹{x:,.2f}")
        stock_models['Holdout MAPE (%)'] = stock_models['MAPE'].map(lambda x: f"{x:.2f}%")
        
        # Baseline directional accuracy is N/A because baseline assumes zero price change
        stock_models['Directional Accuracy (%)'] = stock_models.apply(
            lambda row: "N/A" if row['Model'] == "Baseline" else f"{row['Directional_Accuracy']:.2f}%",
            axis=1
        )
        
        stock_models['Selected Model'] = stock_models['Model'].map(
            lambda m: "⭐ Selected" if m == stock_pred['Best_Model'] else "-"
        )
        
        display_cols = [
            'Model',
            'Forecasted Price (+21 Trading Days)',
            'Forecasted Return (%)',
            'Holdout RMSE (₹)',
            'Holdout MAE (₹)',
            'Holdout MAPE (%)',
            'Directional Accuracy (%)',
            'Selected Model'
        ]
        
        st.dataframe(
            stock_models[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Model Selection Rule & Analytical Context
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.55); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 12px; padding: 18px 22px; margin-top: 10px; margin-bottom: 24px;">
            <div style="font-size: 0.95rem; font-weight: 700; color: #f1f5f9; margin-bottom: 8px;">
                📌 Model Selection Rule: Lowest Holdout RMSE
            </div>
            <p style="margin: 0 0 10px 0; color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
                The production forecasting model for <strong>{stock_pred['Company']}</strong> is objectively selected by comparing 
                out-of-sample holdout test Root Mean Squared Error (RMSE). <strong>{stock_pred['Best_Model']}</strong> achieved the lowest 
                holdout error of <strong>₹{stock_pred['RMSE']:,.2f}</strong> among competing active models.
            </p>
            <div style="font-size: 0.84rem; color: #94a3b8; border-top: 1px solid rgba(148, 163, 184, 0.15); padding-top: 10px; line-height: 1.5;">
                💡 <strong>Conceptual Analytical Distinction:</strong><br>
                • <strong>Technical Indicators</strong> (OHLC Candlesticks, SMAs, Bollinger Bands, Volume, MACD, RSI) describe <em>historical market behavior</em>.<br>
                • <strong>Machine-Learning & Statistical Forecasting</strong> (Baseline, ARIMA, XGBoost, LSTM) describe an <em>independent forecasting experiment</em> evaluating multi-horizon future price estimates. Technical indicators provide historical market context, while the forecasting models independently generate and evaluate future-price estimates.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Model comparison details loading...")

if __name__ == "__main__":
    render_page()
