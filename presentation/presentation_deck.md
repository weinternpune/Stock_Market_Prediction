# Stock Market Prediction — Nifty 500
## Final Findings Presentation Deck · Data Analytics Intern Project
**Author:** Data Analytics Intern | **Project:** Nifty 500 Predictive Modeling Pipeline  
**Status:** PRD Functionality Implemented; Naive-Baseline Performance Target Not Achieved | **Date:** September 2026  

---

## Slide 1: Executive Title & Project Overview
### Nifty 500 Stock Price Prediction System
- **Context:** The Nifty 500 represents ~96% of India's free-float equity market capitalization on the National Stock Exchange (NSE).
- **Core Objective:** Build, evaluate, and benchmark a predictive modeling system for the Nifty 500 index using 5 years of daily market data (September 1, 2021 to August 31, 2026; 1,240 trading sessions).
- **Project Status:** **PRD functionality implemented; naive-baseline performance target not achieved.**
- **Key Deliverables Delivered:**
  1. Authoritative official NSE data ingestion & BSE 500 cross-market proxy reconciliation.
  2. Exploratory Data Analysis & stylized financial facts (fat tails, volatility regimes).
  3. 15+ engineered technical & momentum features (zero lookahead leakage).
  4. Multi-family modeling: Baselines, Statistical ARIMA, Classical ML, and PyTorch LSTM.
  5. 5-Fold Time-Series Cross-Validation and leak-proof walk-forward evaluation.
  6. Real model-driven recursive multi-step forecasting ($T+1$ to $T+30$ days).
  7. 8-page interactive Streamlit dashboard & comprehensive findings report.

---

## Slide 2: Authoritative Data Sourcing & BSE Proxy Architecture
### Authoritative Data Ingestion (PRD Section 5):
- **Primary Source:** Sourced directly from the official NSE historical download (`data/NIFTY_500_Historical_PR_01-09-2021 to 31-08-2026.csv`, 1,240 trading sessions).
- **Minimum Required Schema Fully Met:** `Date, Open, High, Low, Close, Volume, Adjusted Close` (0 nulls).
- **Zero Third-Party Scraping Dependency:** Local pipeline operates with complete self-contained file integrity.

### BSE 500 as Cross-Exchange Market Proxy:
- **Role:** BSE 500 is used as a cross-exchange broad-market reference proxy for consistency validation and co-movement analysis across India's two premier exchanges.
- **Clarification:** BSE 500 is **not** an authoritative duplicate of Nifty 500, but a separate broad-market proxy.
- **Co-Movement Across 1,229 Common Days:**
  - **Price Correlation:** **0.9999**
  - **Daily Return Correlation:** **0.9969**
  - **Defensible Conclusion:** Both indices share strong common broad-market dynamics, confirming high data fidelity.

---

## Slide 3: Trading Calendar Integrity & Outlier Investigation
### Calendar & Quality Verification:
- **Zero Synthetic Dates:** Preserved the official exchange trading calendar (~250 sessions/year). Zero artificial rows were synthesized for weekends or holidays.
- **Data Quality Benchmark Exceeded:** Post-cleaning missing data is **0.00%** (PRD Target: $< 2\%$).

### Outlier Investigation & Event Validation (|Z| > 5):
Rather than arbitrary deletion or truncation, extreme daily return outliers were investigated and verified against official exchange event records:
- **2022-02-24 (-5.04%, Z = -5.59):** Outbreak of the Russia-Ukraine geopolitical crisis.
- **2024-06-04 (-6.76%, Z = -7.48):** 2024 Indian General Election Results counting day market shock.
- **Retention Rationale:** Both represent verified legitimate historical macroeconomic events. Retained in dataset to preserve real-world fat-tail distributions and avoid downside risk censorship bias.

---

## Slide 4: Exploratory Data Analysis & Financial Stylized Facts
### Empirical Observations:
1. **Long-Term Secular Expansion:** The Nifty 500 expanded from ₹14,551 (Sept 2021) to ₹23,450 (Aug 2026).
2. **Dynamic Support Boundaries:** 50-day and 200-day Simple Moving Averages acted as primary structural support zones during corrections.
3. **Leptokurtosis (Fat Tails):** Negative skewness (-0.38) and excess kurtosis (> 3.0) confirm that severe market drawdowns occur far more frequently than modeled by Gaussian normal distributions.
4. **Volatility Clustering:** Volatility shocks persist over extended multi-week regimes, with 20-day annualized volatility varying from 9.5% to 27.4%.

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

## Slide 6: Multi-Family Predictive Modeling & Validation Strategy
### Validation Rigor:
- **Strict Chronological 80/20 Holdout Split:**
  - **Training Set:** 832 trading sessions (June 21, 2022 to October 27, 2025).
  - **Out-of-Sample Test Set:** 208 trading sessions (October 28, 2025 to August 28, 2026).
- **5-Fold Time-Series Cross-Validation:**
  - Evaluated expanding window cross-validation across historical data to evaluate rolling stability:
  - **Random Forest:** CV Mean RMSE: **1094.54** (±779.47) | MAE: 849.42
  - **XGBoost:** CV Mean RMSE: **1067.78** (±828.08) | MAE: 832.33
- **Leak-Proof ARIMA Walk-Forward:**
  - Replaced static full-series concatenation with pure iterative 1-step rolling extend (`model.extend()`). Zero lookahead leakage.

---

## Slide 7: Model Performance Scorecard & The Martingale Reality
### Out-of-Sample Benchmark Table ($N = 208$ Test Days)

| Model Architecture | Model Family | RMSE (Points) | MAE (Points) | MAPE (%) | Directional Hit Rate | Binomial Test ($p$-val) | vs. Naive Baseline RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Persistence)** | Benchmark | **209.33** | **151.88** | **0.669%** | **N/A** | **N/A** | **+0.00% (Best RMSE)** |
| **Random Forest Regressor** | Classical ML | 228.48 | 169.05 | 0.742% | 48.56% | 0.6862 | -9.15% |
| **XGBoost Regressor** | Classical ML | 239.19 | 181.36 | 0.795% | 51.92% | 0.3138 | -14.26% |
| **5-Day Moving Average** | Benchmark | 288.86 | 218.20 | 0.957% | 51.44% | 0.3645 | -37.99% |
| **ARIMA(1, 1, 1) Walk-Forward** | Statistical | 291.08 | 216.41 | 0.950% | 51.92% | 0.3138 | -39.05% |
| **PyTorch LSTM Network** | Deep Learning | 563.46 | 425.56 | 1.877% | **56.73%** | 0.0305 | -169.17% |

### Key Econometric Takeaways:
1. **The Naive Baseline Wins on Level RMSE:** Today's price is the minimum-variance quadratic estimator of tomorrow's price under the Martingale property ($\mathbb{E}[P_{t+1} \mid \mathcal{F}_t] \approx P_t$).
2. **Directional Accuracy for Naive:** Appropriately reported as **N/A** because a persistence model predicts zero price movement, not a directional signal.
3. **PRD Performance Target Status:** **Naive baseline performance target not achieved**, confirming the noise and efficiency of daily equity markets.

---

## Slide 8: Statistical Significance Analysis & Scientific Honesty
### Hypothesis Testing on Directional Hit Rates
- **Null Hypothesis ($H_0$):** Directional hit rate $p = 0.50$ (equivalent to a random coin-toss).
- **Alternative Hypothesis ($H_1$):** Directional hit rate $p > 0.50$ (model has directional edge).

### Findings:
- **XGBoost:** 108 / 208 days (51.92%) $\rightarrow$ One-sided Binomial test $p = 0.3138$ (Fail to reject $H_0$).
- **LSTM:** 118 / 208 days (56.73%) $\rightarrow$ One-sided Binomial test $p = 0.0305$ (Captures sequential trend momentum, but incurs severe level-price error drift of RMSE 563.46).
- **Scientific Integrity:** Transparent reporting confirms that predicting daily closing levels remains extremely difficult under semi-strong market efficiency.

---

## Slide 9: Real Model-Driven Future Horizon Forecasting (FR6)
### Beyond Fixed Drift: Recursive Multi-Step Projections
- **ARIMA Multi-Step:** Statsmodels `get_forecast(steps=H)` calculates the expected mean trajectory and parametric covariance confidence intervals.
- **Recursive Machine Learning:** At each forward day $t+1 \dots t+H$:
  1. Predict next-day closing price using trained XGBoost / Random Forest.
  2. Append forecasted price to the series and **dynamically recalculate all 15+ technical indicators**.
  3. Step forward and compound uncertainty bands based on empirical test error ($z \times RMSE \times \sqrt{h}$).
- **Explicit Forecast Horizon Design:**
  - **Backtesting Target:** Next-day closing price ($T+1$) for rigorous sequential testing.
  - **Future Horizon:** User-selectable from $T+1$ to $T+30$ trading days in the interactive dashboard.

---

## Slide 10: Limitations, Disclaimers & Next Steps
### Project Limitations:
- **Unmodeled Frictions:** Bid-ask spreads, transaction brokerage, and Securities Transaction Tax (STT) are not subtracted.
- **Non-Stationarity:** Daily raw price levels suffer from error drift in neural networks; predicting stationary returns is recommended for future deep learning extensions.
- **Academic Non-Goals Disclaimer:** Developed strictly for academic and intern learning purposes; not intended for live trading or financial advice.

### Recommended Next Steps:
1. **Reformulate Deep Learning on Stationary Returns:** Predict log returns ($\ln(P_{t+1}/P_t)$) rather than raw price levels.
2. **Incorporate Macroeconomic Features:** Ingest RBI policy rates, USD/INR forex rates, and crude oil prices.
3. **Sectoral Hierarchical Modeling:** Build constituent-level sub-models for major sectors (Nifty Bank, Nifty IT, Nifty Auto).
