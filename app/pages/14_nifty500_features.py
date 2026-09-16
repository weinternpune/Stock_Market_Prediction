"""
14_nifty500_features.py
-----------------------
Technical Indicators & Feature Importance Analysis for NIFTY 500.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.components.styling import apply_custom_styles
from app.components.nifty500_data import load_nifty500_data, format_nifty500_chart, FEATURES_DIR

def render_page():
    apply_custom_styles()
    
    st.title("📈 Technical Indicators & Feature Importances")
    st.markdown("##### 15+ quantitative indicators: Moving Averages, Wilder RSI, MACD, Bollinger Bands, and Lags.")
    
    nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df = load_nifty500_data()
    
    features_path = FEATURES_DIR / "nifty_500_features.csv"
    if features_path.exists():
        feat_df = pd.read_csv(features_path)
        feat_df['Date'] = pd.to_datetime(feat_df['Date'])
        
        col_tech1, col_tech2 = st.columns([1.6, 1])
        
        with col_tech1:
            sub_f = feat_df.tail(250)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=sub_f['Date'], y=sub_f['RSI_14'], name="14-Day RSI", line=dict(color='#a855f7', width=1.8)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f43f5e", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)")
            format_nifty500_chart(fig_rsi, height=260, title="14-Day Relative Strength Index (RSI)")
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=sub_f['Date'], y=sub_f['Close'], name="Close", line=dict(color='#f8fafc', width=2)))
            fig_bb.add_trace(go.Scatter(x=sub_f['Date'], y=sub_f['BB_Upper'], name="Upper Band", line=dict(color='rgba(244, 63, 94, 0.4)')))
            fig_bb.add_trace(go.Scatter(x=sub_f['Date'], y=sub_f['BB_Lower'], name="Lower Band", line=dict(color='rgba(16, 185, 129, 0.4)'), fill='tonexty', fillcolor='rgba(148, 163, 184, 0.05)'))
            format_nifty500_chart(fig_bb, height=300, title="Bollinger Bands (20-Day, ±2σ)")
            st.plotly_chart(fig_bb, use_container_width=True)
            
        with col_tech2:
            if not feat_imp.empty:
                fig_imp = px.bar(
                    feat_imp.head(10),
                    x='RF_Importance', y='Feature', orientation='h',
                    title="Top 10 Feature Importances (Random Forest)",
                    color='RF_Importance', color_continuous_scale='Blues'
                )
                format_nifty500_chart(fig_imp, height=440)
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance data not available.")
    else:
        st.warning("nifty_500_features.csv not found.")

if __name__ == "__main__":
    render_page()
