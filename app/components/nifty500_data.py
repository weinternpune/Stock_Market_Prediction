"""
nifty500_data.py
----------------
Cached data loader and chart utilities for NIFTY 500 and BSE 500 benchmark analytics.
"""

from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"

@st.cache_data
def load_nifty500_data():
    """Loads and formats all NIFTY 500 and BSE 500 processed datasets and models."""
    nse_df = pd.read_csv(PROCESSED_DIR / "NIFTY500_clean.csv")
    bse_df = pd.read_csv(PROCESSED_DIR / "BSE500_clean.csv")
    preds_df = pd.read_csv(PROCESSED_DIR / "test_predictions.csv")
    forecast_df = pd.read_csv(PROCESSED_DIR / "future_forecast_t30.csv")
    
    nse_df['Date'] = pd.to_datetime(nse_df['Date'])
    bse_df['Date'] = pd.to_datetime(bse_df['Date'])
    preds_df['Date'] = pd.to_datetime(preds_df['Date'])
    forecast_df['Forecast_Date'] = pd.to_datetime(forecast_df['Forecast_Date'])
    
    metrics_path = MODELS_DIR / "metrics_summary.json"
    metrics_summary = []
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)
            
    imp_path = MODELS_DIR / "feature_importances.csv"
    feat_imp = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
    
    outliers_path = MODELS_DIR / "outlier_investigation.csv"
    outliers_df = pd.read_csv(outliers_path) if outliers_path.exists() else pd.DataFrame()
    
    return nse_df, bse_df, preds_df, forecast_df, metrics_summary, feat_imp, outliers_df

def format_nifty500_chart(fig: go.Figure, height: int = 420, title: str = "") -> go.Figure:
    """Applies theme-adaptive transparent styling to Plotly charts."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#E2E8F0")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Plus Jakarta Sans, sans-serif', color='#94A3B8', size=11),
        margin=dict(l=40, r=20, t=45, b=35),
        height=height,
        xaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.12)',
            zerolinecolor='rgba(148, 163, 184, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.12)',
            zerolinecolor='rgba(148, 163, 184, 0.2)'
        ),
        legend=dict(
            bgcolor='rgba(15, 23, 42, 0.7)',
            bordercolor='rgba(148, 163, 184, 0.2)',
            font=dict(size=10, color="#CBD5E1")
        )
    )
    return fig
