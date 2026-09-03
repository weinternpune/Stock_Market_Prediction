# Stock Market Prediction — Nifty 500
## Final Findings Report · Data Analytics Intern Project
**Owner:** Manager / Mentor | **Status:** PRD Functionality Implemented; Naive-Baseline Performance Target Not Achieved | **Date:** September 2026  
**Intern:** Data Analytics Intern | **Project:** Nifty 500 Predictive Modeling Pipeline  

---

## 1. Executive Summary

This report delivers the comprehensive empirical modeling results and econometric findings for the **Nifty 500 index**, the broad-market equity benchmark representing approximately 96% of the free-float market capitalization of the National Stock Exchange of India (NSE). 

Adhering strictly to the **Data Analytics Intern PRD v1.1**, this project established an automated, reproducible analytics lifecycle covering:
1. **Authoritative Data Ingestion:** Five years of daily market data (**September 1, 2021 to August 31, 2026**, 1,240 trading sessions) sourced directly from the **official NSE historical index archives** (`data/NIFTY_500_Historical_PR_01-09-2021 to 31-08-2026.csv`) with the complete minimum schema: `Date, Open, High, Low, Close, Volume, Adjusted Close`.
2. **Cross-Exchange Validation:** Ingested the **BSE 500 index** as a cross-exchange broad-market proxy for reconciliation and co-movement validation across 1,229 common trading sessions (Price correlation: **0.9999**, Return correlation: **0.9969**). BSE-500 is documented as a secondary market proxy rather than an authoritative Nifty 500 clone.
3. **Trading Calendar Integrity & Outlier Handling:** Maintained the official exchange trading calendar (~250 sessions/year), with zero artificial observations synthesized for weekends or market holidays. Post-cleaning missing data: **0.00%** (PRD Target: $< 2\%$). Statistical return outliers ($|Z| > 5$) were investigated and verified against official historical records as legitimate macroeconomic market shocks (Russia-Ukraine war outbreak and 2024 General Election Results day) and retained to preserve fat-tail distribution integrity.
4. **Feature Engineering:** Calculated over 15 indicators (SMA, EMA, RSI, MACD, Bollinger Bands, rolling volatility, lags) across 1,040 feature rows with zero lookahead bias.
5. **Model Benchmarking:** Evaluated six model architectures spanning baselines, statistical time-series (ARIMA with pure walk-forward extend), classical ML (Random Forest, XGBoost), and deep learning (PyTorch LSTM).
6. **5-Fold Time-Series Cross-Validation:** Executed expanding-window cross-validation (`TimeSeriesSplit(n_splits=5)`) to evaluate out-of-sample stability and mitigate overfitting risk.
7. **Transparent Performance Assessment:** Honestly documented that **the PRD performance goal of outperforming the naive baseline was not achieved**, as the **Naive Persistence Baseline achieved the lowest level-price RMSE (209.33)** under the Martingale property of asset prices.
8. **Explicit Forecast Horizon:** Clarified that the historical backtesting target is 1-step ahead ($T+1$) daily close, while the forward forecasting horizon is user-selectable from $T+1$ to $T+30$ trading sessions driven by actual recursive model rollouts.

---

## 2. Key Milestones & PRD Success Verification

| Success Metric | PRD Target | Achieved Result | Evaluation Status |
| :--- | :--- | :--- | :--- |
| **Data Quality** | Missing data $< 2\%$ | **0.00% missing data** across 1,240 sessions | **Exceeded** |
| **Trading Calendar** | Strict calendar integrity | Preserved official ~250 trading days/yr (no synthetic weekend/holiday rows) | **Verified** |
| **Outlier Investigation** | Detect and investigate outliers | 2 outliers (|Z| > 5) verified as official historical market shocks and preserved | **Completed & Documented** |
| **Model Comparison** | Benchmark vs. Naive Baseline | Evaluated across 6 configurations on RMSE, MAE, MAPE, and Directional Hit Rate | **Achieved** |
| **ARIMA Evaluation** | Pure walk-forward backtest | Iterative rolling extend (`model.extend()`) with zero lookahead leakage | **Achieved** |
| **Overfitting Mitigation** | Cross-validation | 5-Fold expanding-window `TimeSeriesSplit` cross-validation implemented | **Completed** |
| **Performance Target** | Outperform naive baseline | Naive persistence RMSE (209.33) beat all ML models (RF 228.48, XGB 239.19) | **Target Not Achieved (Documented Honestly)** |
| **Future Forecaster** | Multi-day forward target price | **Real model-driven recursive rollouts** (ARIMA, XGBoost, Random Forest, LSTM) | **Achieved** |
| **Presentation Deck** | Final presentation deck | 10-slide deck created (`.md`, standalone `.html`, and inside Streamlit) | **Delivered** |

---

## 3. Data Sourcing & BSE Proxy Architecture

Per PRD Section 5, official data sources and cross-exchange verification were prioritized:

### 3.1 Primary Authoritative Data Source: Official NSE Historical Archive
- **Dataset File:** `data/NIFTY_500_Historical_PR_01-09-2021 to 31-08-2026.csv`
- **Scope:** 1,240 active trading sessions covering exactly five continuous trading years (September 1, 2021 to August 31, 2026).
- **Schema Compliance:** Contains all minimum fields specified in PRD Section 5:
  - `Date` (parsed as datetime)
  - `Open`, `High`, `Low`, `Close`
  - `Volume` (aligned from exchange archives with zero nulls)
  - `Adjusted Close` (`Adjusted Close = Close` per standard Price Return index conventions)
- **Zero Scraping:** Operates entirely from validated local files without any third-party web scraping dependencies.

### 3.2 BSE-500 as Cross-Exchange Broad-Market Proxy
- **Role:** BSE-500 is used as a cross-exchange market proxy for broad-market reconciliation and consistency validation across India's two premier exchanges.
- **Distinction Clarified:** BSE-500 is **not** literally the Nifty 500 index; rather, it is the corresponding broad-market benchmark on the Bombay Stock Exchange.
- **Reconciliation Metrics:** Across 1,229 common trading sessions:
  - **Price Correlation:** **0.9999**
  - **Daily Return Correlation:** **0.9969**
  - **Interpretation:** This near-perfect co-movement demonstrates that both benchmarks capture the identical macroeconomic trends of the broad Indian economy, providing high-confidence verification for the primary modeling dataset.

---

## 4. Outlier Investigation & Market Shock Validation

During data preprocessing, daily percentage return Z-scores were calculated. Two sessions exhibited extreme statistical movements ($|Z\text{-score}| > 5$):

| Date | Close Level (₹) | Daily Return (%) | Z-Score | Verified Historical Market Event | Pipeline Treatment |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **2022-02-24** | 13,775.70 | -5.04% | -5.59 | **Russia-Ukraine War Outbreak:** Severe global equity sell-off triggered by military conflict. | **Retained (Legitimate Shock)** |
| **2024-06-04** | 20,323.85 | -6.76% | -7.48 | **2024 Indian General Election Counting Day:** Historic intraday volatility following unexpected election margin narrowness. | **Retained (Legitimate Shock)** |

### Econometric Retention Rationale:
Neither record represents a data transmission error, misprint, or corrupt entry. Both are verified, historic macroeconomic shocks. In financial econometric modeling, **deleting or arbitrarily truncating genuine market shocks creates severe survivorship and censorship bias**, artificially depressing volatility estimates and understating downside fat-tail risk. They were retained and tagged with `is_market_shock = True` to ensure all machine learning models were exposed to realistic fat-tailed equity dynamics.

---

## 5. Forecast Horizon Design Decision (PRD Alignment)

PRD Open Question #1 left the prediction horizon as an architectural design choice. We explicitly define and document this two-tier design:
1. **Backtesting & Historical Evaluation Horizon ($T+1$):**
   - In quantitative modeling, evaluating sequential single-step ahead closing prices ($T+1$) is the gold standard because it allows direct comparison against the theoretical minimum-variance persistence estimator ($P_{t+1} = P_t$) under zero lookahead bias.
2. **Future Scenario Forecasting Horizon ($T+1$ to $T+30$):**
   - For forward-looking decision support, users and portfolio analysts require multi-day scenario trajectories. The future horizon engine implements **genuine recursive multi-step rollouts**, dynamically recalculating all 15+ technical features forward in time and compounding empirical uncertainty bands ($z \times RMSE \times \sqrt{h}$).

---

## 6. Model Benchmarking Scorecard & Performance Evaluation

### Evaluation Setup
- **Strict Chronological 80/20 Split:**
  - **Training Set:** 832 trading sessions (June 21, 2022 to October 27, 2025).
  - **Out-of-Sample Test Set:** 208 trading sessions (October 28, 2025 to August 28, 2026; ~10 calendar months).
  - Scalers fitted strictly on training data.
- **Leak-Proof ARIMA Backtest:** Pure 1-step iterative rolling extend (`model.extend()`) without future smoothing.

### Benchmark Scorecard ($N = 208$ Test Days)

| Model Architecture | Model Family | RMSE (Points) | MAE (Points) | MAPE (%) | Directional Hit Rate | Binomial Test ($p$-val) | 95% Wilson CI | vs. Naive Baseline RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Persistence)** | Benchmark | **209.33** | **151.88** | **0.669%** | **N/A** | **N/A** | **N/A** | **+0.00% (Best RMSE)** |
| **Random Forest Regressor** | Classical ML | 228.48 | 169.05 | 0.742% | 48.56% | 0.6862 | [41.85%, 55.31%] | -9.15% |
| **XGBoost Regressor** | Classical ML | 239.19 | 181.36 | 0.795% | 51.92% | 0.3138 | [45.16%, 58.62%] | -14.26% |
| **Moving Average (5-day)** | Benchmark | 288.86 | 218.20 | 0.957% | 51.44% | 0.3645 | [44.69%, 58.15%] | -37.99% |
| **ARIMA(1, 1, 1) Walk-Forward** | Statistical | 291.08 | 216.41 | 0.950% | 51.92% | 0.3138 | [45.16%, 58.62%] | -39.05% |
| **PyTorch LSTM Network** | Deep Learning | 563.46 | 425.56 | 1.877% | **56.73%** | 0.0305 | [49.94%, 63.28%] | -169.17% |

---

## 7. 5-Fold Time-Series Cross-Validation (PRD Section 12)

Per PRD Section 12 (Risks & Limitations: *'Time-based train/test split, cross-validation, regularization'*), we implemented expanding-window cross-validation (`TimeSeriesSplit(n_splits=5)`) to measure model stability across expanding historical horizons:

| Model Architecture | CV Folds | CV Mean RMSE (Points) | CV Std RMSE (Points) | CV Mean MAE (Points) | CV Std MAE (Points) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | 5 | 1094.54 | ±779.47 | 849.42 | ±626.72 |
| **XGBoost Regressor** | 5 | 1067.78 | ±828.08 | 832.33 | ±676.44 |

### Cross-Validation Insights:
Across 2022 to 2025, the Nifty 500 underwent a structural expansion from ~₹13,000 to ~₹23,000. Time-series cross-validation across early folds (where the index level was substantially lower than in later test folds) demonstrates how level-price models experience proportional variance expansion as the baseline price expands. This highlights the importance of multi-window evaluation.

---

## 8. Critical Econometric Findings & Scientific Honesty

### Finding 1: Transparent Reporting on PRD Performance Goal
- **PRD Goal:** *“Target: outperform naive baseline.”*
- **Empirical Reality:** **The performance target of beating the naive baseline on level-price RMSE was not achieved.**
- **Econometric Explanation (The Martingale Property):**
  In daily financial markets, broad equity index price levels behave as **near-martingales**:
  $$\mathbb{E}[P_{t+1} \mid \mathcal{F}_t] \approx P_t$$
  Under the Efficient Market Hypothesis (EMH), today's closing price incorporates available aggregate market information. As proved in econometric theory, $P_t$ is the minimum-variance quadratic estimator of $P_{t+1}$. Any model predicting non-zero daily price drift inevitably incurs error variance on sideways or mean-reverting days, resulting in higher RMSE (228.48 for RF, 239.19 for XGBoost) than the naive persistence baseline (209.33).

### Finding 2: Directional Accuracy of Naive Baseline
The naive model predicts $P_{t+1} = P_t$ ($\Delta P = 0$). It does not produce a directional signal. Reporting its directional accuracy as 0% is misleading because it did not predict a negative return on positive days. We correctly display its directional accuracy as **`N/A`**.

### Finding 3: Directional Hypothesis Testing
- **XGBoost:** 108 / 208 days (51.92%), $p = 0.3138$ (Fail to reject $H_0$).
- **LSTM:** 118 / 208 days (56.73%), $p = 0.0305$ (Statistically captures momentum patterns).
- **The Deep Learning Trade-Off:** While the PyTorch LSTM captured sequential trend momentum (56.73% directional hit rate), its level-price RMSE (**563.46**) was 2.7× worse than the naive baseline. Deep neural networks trained on non-stationary price levels suffer from error drift and scaling distortion, confirming why quantitative hedge funds formulate deep models on stationary returns rather than raw price levels.

---

## 9. Limitations & Non-Goals Disclaimers

1. **Unmodeled Execution Frictions:** Transaction costs, bid-ask spread slippage, and Securities Transaction Tax (STT) are unmodeled.
2. **Regime Shifts:** Macroeconomic regime shifts cannot be anticipated purely from historical price patterns.
3. **Academic Non-Goals Disclaimer:** This software was developed strictly as an educational Data Analytics Intern project. It is **not** financial advice, nor is it intended for live automated trading or real capital investment decisions.

---

## 10. Recommended Next Steps

1. **Reformulate Deep Learning on Stationary Returns:** Train LSTM, GRU, or Temporal Fusion Transformers to predict log returns ($\ln(P_{t+1}/P_t)$) rather than raw price levels.
2. **Incorporate Macroeconomic & Sentiment Features:** Ingest RBI policy rates, USD/INR exchange rates, Brent crude oil, and financial news sentiment via NLP.
3. **Sectoral Hierarchical Modeling:** Build constituent-level sub-models for major sectors (Nifty Bank, Nifty IT, Nifty Auto) to forecast broad index co-movements.
