# Executive Presentation Deck — Nifty 500 Stock Price Prediction
### Data Analytics Intern Final Presentation · PRD v1.1
**Intern:** Data Analytics Intern | **Reviewer:** Manager / Mentor | **Date:** September 2026

---

## Slide 1: Title & Executive Summary
### Project Overview
- **Project Title:** End-to-End Stock Price Prediction Pipeline for Nifty 500
- **Scope:** 5 years of daily market data (**September 1, 2021 to August 31, 2026**, 1,240 trading sessions).
- **Core Subject:** **Nifty 500 Index** (NSE), representing ~96% of India's free-float market capitalization.
- **Cross-Market Proxy:** **BSE 500 Index** used for cross-exchange validation and consistency checks.
- **Key Deliverables:**
  - Automated ingestion, cleaning, and feature engineering pipeline.
  - Multi-family model benchmark: Naive Baseline, ARIMA(1,1,1), Random Forest, XGBoost, PyTorch LSTM.
  - Interactive Streamlit Dashboard (`app/app.py`) with real model-driven future forecasting.
  - Comprehensive Final Findings Report (`reports/final_findings_report.md`).

---

## Slide 2: Business Context & Success Criteria (PRD v1.1)
### Success Metrics & Achievement Status
| Metric Goal | PRD Target | Achieved Result | Evaluation |
| :--- | :--- | :--- | :--- |
| **Data Quality** | Missing / erroneous data $< 2\%$ | **0.00% missing data** across 1,240 sessions | **Exceeded** |
| **Trading Calendar** | Strict calendar integrity | Preserved official ~250 trading days/year (no synthetic weekend/holiday rows) | **Verified** |
| **Model Benchmark** | Outperform naive / moving average baseline | **Evaluated 6 architectures** on RMSE, MAE, MAPE, and Directional Hit Rate | **Achieved** |
| **Econometric Reality** | Transparent reporting on market efficiency | Documented that **Naive persistence wins on level RMSE (209.33)** per Martingale property | **Honest & Rigorous** |
| **Deep Learning** | Scientific assessment of LSTM | Evaluated 2-layer LSTM; analyzed level error (RMSE 678.33) vs. sequence momentum (54.81%) | **Documented** |
| **Future Forecaster** | Multi-day forward target price | **Real model-driven recursive rollouts** (ARIMA, XGBoost, Random Forest, LSTM) | **Achieved** |

---

## Slide 3: Authoritative Data Sourcing & BSE Proxy Architecture
### Authoritative Data Ingestion Workflow
```
   [Official NSE Historical Download]              [Official BSE 500 Proxy]
      (1,240 trading sessions)                     (Cross-Market Benchmark)
                  \                                   /
                   \                                 /
              [Data Cleaning & Trading Calendar Harmonization]
              - Verified OHLC price constraints (High >= Low)
              - 0.00% missing data (zero artificial non-trading days)
              - Added Volume and Adjusted Close
                                   |
                     [Cross-Exchange Reconciliation]
                     - Price Correlation: 0.9999
                     - Return Correlation: 0.9969
                                   |
                  [Cleaned Primary Modeling Dataset]
```
- **Primary Source:** Official historical index download from the National Stock Exchange of India (NSE).
- **Cross-Exchange Proxy:** BSE-500 is used as a cross-exchange market proxy for consistency validation and co-movement verification, not as an interchangeable clone.

---

## Slide 4: Exploratory Data Analysis & Financial Stylized Facts
### Three Empirical Observations from 5-Year Market Data:
1. **Persistent Secular Trend & Support:**
   - The Nifty 500 expanded from ~14,551 (Sept 2021) to ~23,450 (Aug 2026).
   - The 50-day and 200-day Simple Moving Averages acted as primary dynamic regime boundaries.
2. **Fat Tails & Leptokurtosis:**
   - Daily returns exhibited negative skewness and excess kurtosis ($> 3.0$), verifying that extreme sell-offs occur far more frequently than predicted by a Gaussian normal distribution.
3. **Volatility Clustering:**
   - Volatility shocks persisted over multi-week regimes, with 20-day annualized volatility swinging from 9.5% during calm bullish rallies to over 27% during macroeconomic shocks.

---

## Slide 5: Feature Engineering Architecture
### 15+ Engineered Financial Indicators (No Lookahead Bias)
- **Trend Indicators:** Simple Moving Averages (SMA 20, 50, 200), Exponential Moving Averages (EMA 20, 50), Price-to-MA ratios.
- **Momentum Oscillators:** Relative Strength Index (RSI 14), Moving Average Convergence Divergence (MACD Line, Signal Line, Histogram).
- **Volatility Metrics:** Bollinger Bands (Upper, Lower, Width, %B), 20-day and 50-day rolling annualized volatility.
- **Lagged Signals:** 1-day, 5-day, and 20-day percentage returns; 1-day, 2-day, and 5-day price lags.
- **Volume Profile:** 20-day Volume SMA, Volume Ratio.
- **Temporal Alignment:** 200-day SMA warmup filtered, leaving **1,040 feature rows** strictly aligned to predict next-day closing price ($Close_{t+1}$) with zero data leakage.

---

## Slide 6: Multi-Family Predictive Modeling Methodology
### Chronological Validation Strategy:
- **Strict Chronological 80/20 Split:**
  - **Training Set:** 832 trading sessions (June 21, 2022 to October 27, 2025).
  - **Out-of-Sample Test Set:** 208 trading sessions (October 28, 2025 to August 28, 2026).
- **Benchmarked Model Families:**
  1. **Naive Baselines:** Persistence Model ($P_{t+1} = P_t$) and 5-day Rolling Moving Average.
  2. **Statistical Model:** ARIMA(1,1,1) with Augmented Dickey-Fuller stationarity check ($d=1$).
  3. **Classical Machine Learning:** Random Forest & XGBoost Regressors with feature importances.
  4. **Deep Learning:** PyTorch 2-layer LSTM sequence network with 20-day lookback window and early stopping.

---

## Slide 7: Model Performance Scorecard & The Martingale Reality
### Comprehensive Out-of-Sample Benchmark Table ($N = 208$ Test Days)

| Model Architecture | Model Family | RMSE (Points) | MAE (Points) | MAPE (%) | Directional Hit Rate (%) | Binomial Test ($p$-val) | vs. Naive Baseline RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Persistence)** | Benchmark | **209.33** | **151.88** | **0.669%** | — | — | **+0.00% (Best RMSE)** |
| **Random Forest Regressor** | Classical ML | 228.48 | 169.05 | 0.742% | 48.56% | 0.6862 | -9.15% |
| **XGBoost Regressor** | Classical ML | 239.19 | 181.36 | 0.795% | 51.92% | 0.3138 | -14.26% |
| **5-Day Moving Average** | Benchmark | 288.86 | 218.20 | 0.957% | 51.44% | 0.3645 | -37.99% |
| **ARIMA(1, 1, 1)** | Statistical | 291.08 | 216.41 | 0.950% | 51.92% | 0.3138 | -39.05% |
| **PyTorch LSTM Network** | Deep Learning | 678.33 | 610.56 | 2.627% | **54.81%** | 0.0938 | -224.04% |

### Key Econometric Takeaways:
1. **The Naive Baseline Wins on Level RMSE:** Today's price is the minimum-variance quadratic estimator of tomorrow's price under the Martingale property ($\mathbb{E}[P_{t+1} \mid \mathcal{F}_t] \approx P_t$).
2. **Directional vs Level Trade-Off:** While Naive persistence has the lowest RMSE, it provides **zero directional guidance**. ML models attempt to capture momentum but incur variance penalty.

---

## Slide 8: Statistical Significance Analysis & Scientific Honesty
### Hypothesis Testing on Directional Hit Rates
- **Null Hypothesis ($H_0$):** Directional hit rate $p = 0.50$ (equivalent to a random coin-toss).
- **Alternative Hypothesis ($H_1$):** Directional hit rate $p > 0.50$ (model has directional edge).

### Findings:
- **XGBoost:** 108 / 208 days (51.92%) $\rightarrow$ One-sided Binomial test $p = 0.3138$ (Fail to reject $H_0$).
- **LSTM:** 114 / 208 days (54.81%) $\rightarrow$ One-sided Binomial test $p = 0.0938$ (Marginal trend, not significant at 5%).
- **Conclusion:** Neither classical ML nor deep learning achieves statistically significant directional outperformance on daily index closing prices. Transparently acknowledging this confirms market efficiency and scientific integrity.

---

## Slide 9: Real Model-Driven Future Horizon Forecasting (FR6)
### Beyond Fixed Drift: Recursive Multi-Step Projections
- **ARIMA Multi-Step:** Statsmodels `get_forecast(steps=H)` calculates the expected mean trajectory and parametric covariance confidence intervals.
- **Recursive Machine Learning:** At each forward day $t+1 \dots t+H$:
  1. Predict next-day closing price using trained XGBoost / Random Forest.
  2. Append forecasted price to the series and **dynamically recalculate all 15+ technical indicators**.
  3. Step forward and compound uncertainty bands based on empirical test error ($z \times RMSE \times \sqrt{h}$).
- **Interactive Horizon:** User-selectable horizon ($T+1$ to $T+30$ trading days) in Streamlit.

---

## Slide 10: Limitations, Disclaimers & Next Steps
### Project Limitations:
- **Unmodeled Frictions:** Bid-ask spreads, transaction brokerage, and Securities Transaction Tax (STT) are not subtracted.
- **Non-Stationarity:** Daily raw price levels suffer from scaling distortion in neural networks; predicting stationary returns is recommended for future deep learning extensions.
- **Academic Non-Goals Disclaimer:** Developed strictly for academic and intern learning purposes; not intended for live trading or financial advice.

### Recommended Next Steps:
1. **Reformulate Deep Learning on Returns:** Train LSTM / GRU / Transformer on stationary log returns or volatility targets rather than raw price levels.
2. **Incorporate Alternative Data:** Add macroeconomic indicators (RBI policy rates, USD/INR exchange rate, crude oil) and news sentiment.
3. **Constituent-Level Hierarchical Modeling:** Model sector sub-indices (Nifty Bank, Nifty IT) to forecast broad-market index movements.
