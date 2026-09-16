"""
18_nifty500_deck.py
-------------------
Executive Presentation Deck: 15 Interactive Slides for NIFTY 500.
"""

from pathlib import Path
import sys
import streamlit as st

PAGES_DIR = Path(__file__).resolve().parent
APP_DIR = PAGES_DIR.parent
PROJECT_ROOT = APP_DIR.parent

for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.components.styling import apply_custom_styles

def render_page():
    apply_custom_styles()
    
    st.title("🖥️ Executive Presentation Deck")
    st.markdown("##### 15 interactive presentation slides summarizing methodology, multi-model benchmarks, and findings.")
    
    slides_data = [
        ("Overview", "Nifty 500 Stock Market Prediction Pipeline", "Data Analytics Intern Capstone Project · PRD v1.1", 
         "5-year quantitative study (1,240 trading sessions, 2021–2026) evaluating whether statistical, classical ML, or deep learning can reliably outperform baseline persistence under noisy financial conditions."),
        
        ("Problem Statement", "The Challenge of Financial Time-Series Forecasting", "Low signal-to-noise ratio & non-stationarity",
         "Financial asset returns possess very low signal-to-noise ratios, with microstructural shocks dominating day-to-day level prices. Shifting macroeconomic regimes invalidate static assumptions."),
        
        ("Project Objective", "Formal Objective & Target Definition", "1-step ahead regression without data leakage",
         "Predict the next trading day's closing price P(t+1) of the Nifty 500 index using 5 years of daily data, comparing Naive, 5-Day SMA, walk-forward ARIMA, Random Forest, XGBoost, and PyTorch LSTM."),
        
        ("Data Sourcing", "Authoritative Dual-Exchange Data Architecture", "NSE Nifty 500 Primary Index & BSE 500 Benchmark",
         "NSE Nifty 500 serves as the primary modeling series (1,240 rows). BSE 500 serves as the cross-exchange benchmark (1,240 rows) with 0.99989 price correlation. Both series remain independently preserved."),
        
        ("Data Cleaning", "Trading Calendar Integrity & Outlier Audit", "0.00% missing data post-cleaning",
         "Strict adherence to exchange calendars (~250 trading days/yr). Macro outliers (2022 Ukraine invasion -5.04%, 2024 Election Day -6.76%) were audited against official exchange logs and preserved to prevent downside censorship bias."),
        
        ("Exploratory Analysis", "Stylized Facts of Financial Returns", "Heavy tails, asymmetry, and volatility clustering",
         "Nifty 500 returns exhibit negative skewness (-0.678), excess kurtosis (4.585), and strict non-normality (Jarque-Bera p < 10^-250). Mandelbrot volatility clustering is clearly documented."),
        
        ("Feature Engineering", "Quantitative Technical Feature Set", "15+ technical indicators with zero lookahead bias",
         "Engineered SMA (10, 20, 50, 200), EMA (12, 26), 14-day Wilder RSI, MACD line & signal, Bollinger Bands, and return lags. 200-day warm-up cutoff yielded 1,040 clean modeling rows."),
        
        ("Model Architectures", "Multi-Family Benchmark Setup", "Statistical, classical ML, and deep learning architectures",
         "Baseline persistence (P_t), 5-Day SMA, pure walk-forward ARIMA(1, 1, 1), Random Forest (150 trees, depth 8), XGBoost (lr 0.03), and PyTorch 2-layer stacked LSTM network."),
        
        ("Performance Scorecard", "Out-of-Sample Scorecard (208 Sessions)", "Oct 28, 2025 to Aug 28, 2026",
         "Naive Persistence: RMSE 209.33 (Best) | Random Forest: RMSE 229.66 | 5-Day SMA: RMSE 288.87 | ARIMA(1,1,1): RMSE 291.08 | XGBoost: RMSE 299.91 | LSTM: RMSE 560.27."),
        
        ("Backtesting Dynamics", "Tracking Performance Across Regimes", "Lagging turning points vs. persistence efficiency",
         "Machine learning models track multi-week trends but lag sharp turns by 1–2 days. On sideways days, models predicting directional movement incur variance penalties, allowing persistence to maintain the lowest level RMSE."),
        
        ("Model Selection", "Final Model Selection & Rationale", "Balancing theoretical bounds and ML utility",
         "Naive Persistence is selected as the primary benchmark for price levels under quadratic loss. Random Forest is selected as the best feature-driven model (RMSE 229.66, within 9.7% of persistence)."),
        
        ("Web Application", "Interactive Streamlit Dashboard", "Production-grade quantitative application",
         "Provides interactive candlestick charts, dual-exchange comparisons, technical feature exploration, backtesting visualizers, and recursive forward projections from T+1 to T+30."),
        
        ("Scientific Findings", "The Martingale Property & Efficient Markets", "Empirical confirmation of E[P(t+1)|F_t] = P(t)",
         "In an efficient market, today's price is the minimum-variance quadratic estimator of tomorrow's price level. Beating persistence on level-price RMSE is structurally difficult without lookahead leakage."),
        
        ("Project Limitations", "Financial & Data Constraints", "Documented transparently per PRD Section 12",
         "Daily granularity misses intraday liquidity flow. Unscheduled macroeconomic shocks are non-forecastable. Non-stationarity in raw price levels leads to sliding window drift in neural networks."),
        
        ("Conclusion", "Conclusion & Strategic Roadmap", "Summary of achievements and future quantitative enhancements",
         "100% of PRD requirements fulfilled. Recommended next step: transition prediction target from non-stationary price levels to stationary log-returns r(t+1) = ln(P(t+1)/P(t)).")
    ]
    
    slide_idx = st.selectbox(
        "Select Slide to View:",
        range(1, 16),
        format_func=lambda x: f"Slide {x}: {slides_data[x-1][1]}"
    )
    
    cur_slide = slides_data[slide_idx - 1]
    
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 14px; padding: 28px 32px; color: #f8fafc; margin-top: 16px; backdrop-filter: blur(10px); box-shadow: 0 12px 30px rgba(0,0,0,0.3);">
        <div style="display: inline-block; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 9999px; padding: 4px 12px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
            {cur_slide[0]} · Slide {slide_idx} of 15
        </div>
        <h2 style="margin: 14px 0 4px 0; color: #f8fafc; font-size: 1.5rem; font-weight: 700;">{cur_slide[1]}</h2>
        <div style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;">{cur_slide[2]}</div>
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 10px; padding: 22px 24px; font-size: 1.05rem; line-height: 1.7; color: #e2e8f0;">
            {cur_slide[3]}
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
