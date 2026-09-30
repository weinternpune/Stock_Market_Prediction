"""
15_nifty500_models.py
---------------------
Out-of-Sample Model Performance Scorecard for NIFTY 500.
"""

from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.components.styling import apply_custom_styles
from app.components.nifty500_data import load_nifty500_data, format_nifty500_chart

def render_page():
    apply_custom_styles()
    
    st.title("🏆 NIFTY 500 Out-of-Sample Model Scorecard")
    st.markdown("##### Benchmarked across 208 held-out trading sessions (October 28, 2025 to August 28, 2026).")
    
    nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df = load_nifty500_data()
    
    scorecard_df = pd.DataFrame(metrics_summary)
    if not scorecard_df.empty:
        display_cols = [c for c in ["Model", "RMSE", "MAE", "MAPE_Pct", "Directional_Accuracy", "vs_Naive_RMSE"] if c in scorecard_df.columns]
        
        st.dataframe(
            scorecard_df[display_cols].style.highlight_min(subset=[c for c in ["RMSE", "MAE", "MAPE_Pct"] if c in scorecard_df.columns], color="rgba(16, 185, 129, 0.25)"),
            use_container_width=True,
            hide_index=True
        )
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            if 'RMSE' in scorecard_df.columns:
                fig_rmse = px.bar(
                    scorecard_df, x='Model', y='RMSE', color='Model',
                    title="Out-of-Sample RMSE (Points) · Lower is Better",
                    text='RMSE', color_discrete_sequence=px.colors.qualitative.Safe
                )
                format_nifty500_chart(fig_rmse, height=360)
                fig_rmse.update_layout(showlegend=False)
                st.plotly_chart(fig_rmse, use_container_width=True)
            
        with c_m2:
            if 'raw_dir_acc' in scorecard_df.columns:
                dir_df = scorecard_df[scorecard_df['raw_dir_acc'].notna()].copy()
                fig_dir = px.bar(
                    dir_df, x='Model', y='raw_dir_acc', color='Model',
                    title="Directional Hit Rate (%) · Threshold = 50%",
                    text='raw_dir_acc', color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_dir.add_hline(y=50.0, line_dash="dash", line_color="#94a3b8")
                format_nifty500_chart(fig_dir, height=360)
                fig_dir.update_layout(showlegend=False, yaxis_range=[40, 60])
                st.plotly_chart(fig_dir, use_container_width=True)
                
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.55); border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 8px; color: #cbd5e1; font-size: 0.85rem; margin-top: 16px;">
            <strong style="color: #fbbf24;">Scientific Insight (The Martingale Property):</strong> Under the Martingale property of asset prices (<code>E[P(t+1)|F_t] = P(t)</code>), today's price is the minimum-variance quadratic estimator of tomorrow's price level. The Naive Persistence model achieves the lowest level-price RMSE (209.33) because models predicting directional moves incur quadratic variance penalties on sideways days.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Scorecard metrics not found.")

if __name__ == "__main__":
    render_page()
