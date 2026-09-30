"""
charts.py
---------
Interactive Plotly charts for financial price action, indicators, residuals,
model evaluation comparisons, and cross-stock correlation heatmaps.
Engineered with transparent backgrounds and modern fintech color schemes.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CHART_THEME = "plotly_dark"
GRID_COLOR = "rgba(148, 163, 184, 0.12)"
FONT_COLOR = "#94a3b8"

def apply_clean_layout(fig: go.Figure, height: int = 450, title: str = None) -> go.Figure:
    """Applies modern transparent fintech styling to Plotly figures."""
    layout_dict = dict(
        template=CHART_THEME,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        font=dict(family="Plus Jakarta Sans, sans-serif", color=FONT_COLOR, size=11),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    )
    if title:
        layout_dict["title"] = dict(text=title, font=dict(size=14, color="#f1f5f9", family="Plus Jakarta Sans"))
    fig.update_layout(**layout_dict)
    return fig

def plot_price_and_indicators(stock_df: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Creates a multi-panel financial chart:
    Panel 1: Candlestick + SMA 20, 50, 200 + Bollinger Bands
    Panel 2: Volume bar chart
    Panel 3: MACD (MACD line, Signal, Histogram)
    Panel 4: RSI (14) with 70/30 overbought/oversold threshold lines
    """
    df = stock_df.sort_values('Date').copy()
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        subplot_titles=(
            f"{symbol} Price & Moving Averages",
            "Trading Volume",
            "MACD (12, 26, 9)",
            "RSI (14)"
        )
    )
    
    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC",
            increasing_line_color='#10b981',
            decreasing_line_color='#f43f5e'
        ),
        row=1, col=1
    )
    
    # SMAs
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='#60a5fa', width=1.5), name="SMA 20"), row=1, col=1)
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], line=dict(color='#fbbf24', width=1.5), name="SMA 50"), row=1, col=1)
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_200'], line=dict(color='#c084fc', width=1.5), name="SMA 200"), row=1, col=1)
        
    # Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], line=dict(color='rgba(148,163,184,0.3)', dash='dot'), name="BB Upper", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], line=dict(color='rgba(148,163,184,0.3)', dash='dot'), fill='tonexty', fillcolor='rgba(148,163,184,0.05)', name="BB Band", showlegend=False), row=1, col=1)
        
    # 2. Volume
    colors = ['#10b981' if c >= o else '#f43f5e' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="Volume"),
        row=2, col=1
    )
    if 'Volume_SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume_SMA_20'], line=dict(color='#94a3b8', width=1.2), name="Vol SMA 20"), row=2, col=1)
        
    # 3. MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], line=dict(color='#38bdf8', width=1.5), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], line=dict(color='#f43f5e', width=1.2), name="Signal"), row=3, col=1)
        hist_colors = ['#10b981' if h >= 0 else '#f43f5e' for h in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], marker_color=hist_colors, name="Hist"), row=3, col=1)
        
    # 4. RSI
    if 'RSI_14' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI_14'], line=dict(color='#a855f7', width=1.5), name="RSI (14)"), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(244, 63, 94, 0.6)", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.6)", row=4, col=1)
        
    fig.update_layout(
        template=CHART_THEME,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=720,
        margin=dict(l=40, r=40, t=40, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig

def plot_backtest_actual_vs_predicted(backtest_df: pd.DataFrame, symbol: str, model_name: str) -> go.Figure:
    """Plots actual vs forecasted future prices over the rolling holdout test period."""
    sub = backtest_df[(backtest_df['Symbol'] == symbol) & (backtest_df['Model'] == model_name)].sort_values('Date')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub['Date'], y=sub['Actual_Future_Close'],
        mode='lines+markers', name="Actual Future Close",
        line=dict(color='#f8fafc', width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=sub['Date'], y=sub['Predicted_Future_Close'],
        mode='lines+markers', name=f"Forecasted Future Close ({model_name})",
        line=dict(color='#38bdf8', width=2.0, dash='dash')
    ))
    
    apply_clean_layout(fig, height=430, title=f"Actual vs. Forecasted 21-Trading-Day Forward Close — {symbol} ({model_name})")
    fig.update_layout(
        xaxis_title="Origin Date",
        yaxis_title="Price (₹)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_residual_distribution(backtest_df: pd.DataFrame, symbol: str, model_name: str) -> go.Figure:
    """Plots residual distribution (Actual - Forecasted) for test predictions."""
    sub = backtest_df[(backtest_df['Symbol'] == symbol) & (backtest_df['Model'] == model_name)]
    errors = sub['Error'].dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=25,
        marker_color='#6366f1', opacity=0.8,
        name="Residual (₹)"
    ))
    fig.add_vline(x=0, line_dash="solid", line_color="#f43f5e", line_width=1.5)
    
    apply_clean_layout(fig, height=320, title=f"Forecast Residual Distribution (Residual = Actual − Forecast) — {symbol}")
    fig.update_layout(xaxis_title="Residual (Actual − Forecast) (₹)", yaxis_title="Frequency")
    return fig

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "NIFTY 50 Daily Returns Correlation") -> go.Figure:
    """Interactive correlation heatmap with custom diverging color palette."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmin=-0.2, zmax=1.0,
        colorbar=dict(title="Corr")
    ))
    apply_clean_layout(fig, height=650, title=title)
    return fig

def plot_model_win_counts(win_counts: dict) -> go.Figure:
    """Bar chart of model win counts across all 50 stocks."""
    models = list(win_counts.keys())
    counts = list(win_counts.values())
    
    colors = {
        "XGBoost": "#38bdf8",
        "LSTM": "#a855f7",
        "ARIMA": "#10b981",
        "Baseline": "#64748b"
    }
    bar_colors = [colors.get(m, "#3b82f6") for m in models]
    
    fig = go.Figure(go.Bar(
        x=models, y=counts,
        text=counts, textposition='outside',
        marker=dict(color=bar_colors, opacity=0.9)
    ))
    apply_clean_layout(fig, height=350, title="Model Selection by Holdout RMSE")
    fig.update_layout(xaxis_title="Architecture", yaxis_title="Stocks with Lowest Holdout RMSE")
    return fig
