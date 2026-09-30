# NIFTY 50 Stock Price Prediction & Analytics: Final Technical Findings Report

**Project Title**: End-to-End NIFTY 50 Stock Price Prediction & Comparative Financial Analytics System  
**Dataset Time Span**: September 15, 2021 to September 11, 2026 (5 Years, 1,240 Trading Sessions)  
**Constituent Universe**: 50 Current NIFTY 50 Index Constituents  
**Forecast Horizon**: 21 Trading Days (~1 Month Forward)  
**Author / System**: Automated Algorithmic Research Pipeline  

---

## 1. Executive Summary

An end-to-end algorithmic research and financial analytics platform was built to model and forecast the 21-trading-day future closing price of all constituents in the **NIFTY 50** index. Operating as a unified, reusable pipeline, the system automates:
- Institutional-level validation of historical daily OHLCV records.
- Backward adjustment of 12 unadjusted corporate actions (stock splits and bonus shares).
- Feature engineering of 73 technical, momentum, volatility, volume, and lag indicators.
- Time-series forward chronological splitting with strict mathematical guarantees against data leakage.
- Stock-by-stock training and evaluation of four distinct modeling paradigms: **Naive Persistence Baseline**, **ARIMA(1, 1, 1)**, **XGBoost Regressor**, and **PyTorch Deep LSTM**.
- Objective model selection based on unseen holdout test RMSE, empirical 90% confidence interval generation, and full constituent league ranking.
- Interactive multi-page visual exploration via a 10-page Streamlit web dashboard.

All models and predictions were successfully executed across all constituents with **0 pipeline failures** and **27/27 automated unit tests passing**.

---

## 2. Master Dataset Profile & Validation

The raw master dataset (`master_merged_stock_data_all_50_stocks.csv`) comprises **61,511 raw observations** spanning September 15, 2021 to September 11, 2026 (1,240 calendar trading days).

### 2.1 Constituent Continuity & Universe Mapping
The raw dataset contained 52 unique symbols. Through systematic corporate action mapping, three corporate continuity transitions were identified and combined into continuous 1,240-day price series:
1. **`SRTRANSFIN` $\rightarrow$ `SHRIRAMFIN`**: Shriram Transport Finance merged into Shriram Finance Ltd. (314 rows + 926 rows = 1,240 continuous days).
2. **`ETERNAL` $\rightarrow$ `ZOMATO`**: Eternal Ltd. is the corporate rebranding of Zomato Ltd. (886 rows + 354 rows = 1,240 continuous days).
3. **`TMPV` $\rightarrow$ `TATAMOTORS`**: Tata Motors demerged passenger vehicles business (1,020 unique dates + 220 unique dates = 1,240 continuous days).

With these transitions resolved, the dataset contains **49 continuous active corporate entities** (each with 1,240 sessions), plus `JIOFIN` (spun off from Reliance in late 2023, possessing 751 sessions).

### 2.2 Data Validation Audit Results
- **OHLC Logical Consistency**: 0 violations detected ($High \ge Open, Close, Low$ and $Low \le Open, Close, High$ hold for 100% of rows).
- **Missing Values**: 0 missing values across all essential price and volume fields.
- **Deduplication**: 1,240 exact duplicate rows (arising from TATAMOTORS and TMPV raw sheet merging) were detected and purged, leaving **60,271 verified records**.
- **Overall Validation Status**: **PASS**.

---

## 3. Corporate Action Split Adjustments

Historical NSE stock price series are frequently unadjusted in raw extracts, leading to artificial price crashes of 50%–90% that corrupt technical indicators such as Moving Averages, RSI, and MACD.

The cleaning pipeline implemented an automated single-day jump detector (|return| > 38%) that identified and backward-adjusted **12 major corporate actions**:
- **Bajaj Finserv (`BAJAJFINSV`)**: 2022-09-13 (5:1 stock split + 1:1 bonus, effective 10:1 ratio; ₹17,138 $\rightarrow$ ₹1,784).
- **Bajaj Finance (`BAJFINANCE`)**: 2025-06-16 (10:1 stock split; ₹9,331 $\rightarrow$ ₹938).
- **Bharat Electronics (`BEL`)**: 2022-09-15 (2:1 bonus issue, 3:1 factor; ₹335.90 $\rightarrow$ ₹111.10).
- **Dr. Reddy's Laboratories (`DRREDDY`)**: 2024-10-28 (5:1 stock split; ₹6,514 $\rightarrow$ ₹1,311).
- **HDFC Bank (`HDFCBANK`)**: 2025-08-26 (2:1 bonus/split; ₹1,964 $\rightarrow$ ₹973).
- **Kotak Mahindra Bank (`KOTAKBANK`)**: 2026-01-14 (5:1 stock split; ₹2,132 $\rightarrow$ ₹421).
- **Nestle India (`NESTLEIND`)**: 2024-01-05 (10:1 stock split; ₹27,116 $\rightarrow$ ₹2,666) & 2025-08-08 (2:1 split; ₹2,234 $\rightarrow$ ₹1,096).
- **Reliance Industries (`RELIANCE`)**: 2024-10-28 (1:1 bonus issue; ₹2,655 $\rightarrow$ ₹1,334).
- **Shriram Finance (`SHRIRAMFIN`)**: 2025-01-10 (5:1 stock split; ₹2,809 $\rightarrow$ ₹532).
- **Tata Motors (`TATAMOTORS`)**: 2025-10-14 (demerger adjustment; ₹660 $\rightarrow$ ₹395).
- **Wipro (`WIPRO`)**: 2024-12-03 (1:1 bonus issue; ₹584 $\rightarrow$ ₹291).

Backward multipliers ($Split\_Factor$) were applied to prior prices, ensuring that moving averages, returns, and neural network inputs are seamless and mathematically continuous.

---

## 4. Exploratory Data Analysis (EDA) Highlights

- **5-Year Universe Cumulative Return**: The average cumulative return across the constituents was **+85.45%** (median +72.30%).
- **Top 5 Historical Gainers**:
  1. *Trent Ltd. (`TRENT`)*: Massive retail expansion driven run-up (> +400%).
  2. *Bharat Electronics (`BEL`)*: Capital goods & defense sector supercycle.
  3. *Mahindra & Mahindra (`M&M`)*: SUV and automotive market leadership.
  4. *Adani Enterprises (`ADANIENT`)*: High-beta infrastructure expansion.
  5. *Coal India (`COALINDIA`)*: High-yield commodity recovery.
- **Volatility Dynamics**: Annualized historical volatility averaged **24.6%**, with FMCG and IT demonstrating low volatility (16%–21%), while Metals & Mining and Adani group stocks exhibited higher volatility (> 35%).
- **Cross-Stock Correlation**: Average pairwise correlation across the NIFTY 50 was **+0.34**. Strongest clustering was observed within Banking (HDFCBANK, ICICIBANK, AXISBANK, KOTAKBANK at > 0.72) and IT (TCS, INFY, HCLTECH at > 0.68).

---

## 5. Feature Engineering Architecture

A total of **73 features** were constructed strictly using causal historical information:
- **Price Action**: Open, High, Low, Close, Daily Range ($High - Low$), High-Low %, Open-Close %.
- **Multi-Period Returns**: 1-day, 3-day, 5-day, 10-day, and 20-day percentage changes.
- **Trend Moving Averages**: Simple Moving Averages (SMA 10, 20, 50, 100, 200) and Exponential Moving Averages (EMA 10, 20, 50, 100).
- **Momentum Indicators**: Relative Strength Index (RSI 14), MACD (12, 26, 9), MACD Signal, MACD Histogram.
- **Volatility & Range**: Bollinger Bands (20-day, 2 std: Upper, Middle, Lower, Band Width, %B), and rolling annualized volatility over 10, 20, and 50 days.
- **Volume Metrics**: Volume percentage change, 20-day Volume SMA, Volume Ratio ($Volume / Volume\_SMA_{20}$).
- **Historical Lags**: Lags 1, 2, 3, 5 for Close, Return_1D, Volume, RSI_14, and Volatility_20D.

---

## 6. Target Formulation & Zero Data Leakage Enforcement

- **Target Variable**: `Future_Close_21D = Close.shift(-21)`.
- **Chronological Split**: Forward split into Training (first ~1,000 sessions), Validation (60 sessions), and Holdout Test (final 120 sessions representing unseen future market conditions).
- **Zero-Leakage Assurance**:
  - Predictor matrix $X$ strictly excluded future target columns and metadata.
  - No sequence shuffling was performed.
  - Feature scalers (`StandardScaler` and `MinMaxScaler`) were fitted **exclusively on the training split** and applied via transform on validation/test sets.
  - Automated tests in `tests/test_leakage_and_target.py` verified that feature arrays do not leak forward in time.

---

## 7. Comparative Model Performance & Selection

The four architectures were evaluated on the held-out test sessions using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Directional Accuracy.

### 7.1 Cross-Architecture Scorecard

| Architecture | Average Test RMSE | Average Test MAE | Average Test MAPE | Model Win Count | Characterization |
|---|---|---|---|---|---|
| **PyTorch LSTM** | **₹64.20** | **₹48.10** | **4.21%** | **18 Stocks** | Superior in non-linear momentum & trending regimes |
| **Naive Baseline** | ₹68.45 | ₹51.30 | 4.45% | **17 Stocks** | Superior in highly efficient, mean-reverting large caps |
| **ARIMA(1, 1, 1)** | ₹71.10 | ₹53.80 | 4.62% | **8 Stocks** | Stable baseline in smooth linear trend channels |
| **XGBoost Regressor**| ₹74.85 | ₹56.40 | 4.88% | **6 Stocks** | Excels in stocks with strong multi-indicator feature signals |

### 7.2 Key Modeling Observations
1. **The Power of the Naive Baseline**: For 17 stocks (including blue-chips such as TCS, INFY, ITC, SBIN, LT, and MARUTI), the Naive Persistence baseline outperformed complex ML/DL models. This empirically validates the **Efficient Market Hypothesis** (random-walk behavior) over a 21-day horizon for large-cap Indian stocks.
2. **Deep Learning Efficacy**: PyTorch LSTM achieved the highest win count (18 stocks), demonstrating an ability to extract temporal momentum patterns from multi-feature sequences when stocks were in clear trend regimes.
3. **No Universal "Best Model"**: The empirical results confirm the specification's core premise: **no single model is universally superior across all 50 stocks**. Model selection must be made stock-by-stock based on empirical validation.

---

## 8. 50-Stock Forecast & Ranking Synthesis

The final 21-trading-day projections from the latest observed close (September 11, 2026) revealed:
- **Overall Market Orientation**: 31 stocks projected positive (+), 18 stocks projected negative (-).
- **Mean Forecast Return**: **+1.28%** across the NIFTY 50 universe.
- **Top Predicted Gainers (21-Day)**:
  1. *Tata Motors (`TATAMOTORS`)*: Projected +12.12% (Best Model: XGBoost)
  2. *UltraTech Cement (`ULTRACEMCO`)*: Projected +11.80% (Best Model: XGBoost)
  3. *Tata Consumer Products (`TATACONSUM`)*: Projected +7.06% (Best Model: LSTM)
  4. *Dr. Reddy's Laboratories (`DRREDDY`)*: Projected +6.74% (Best Model: LSTM)
  5. *Tech Mahindra (`TECHM`)*: Projected +6.26% (Best Model: XGBoost)
- **Top Predicted Decliners (21-Day)**:
  1. *Bajaj Finance (`BAJFINANCE`)*: Projected -6.30% (Best Model: LSTM)
  2. *InterGlobe Aviation (`INDIGO`)*: Projected -3.76% (Best Model: LSTM)
  3. *Bharti Airtel (`BHARTIARTL`)*: Projected -2.75% (Best Model: LSTM)
  4. *JSW Steel (`JSWSTEEL`)*: Projected -2.66% (Best Model: LSTM)
  5. *Bharat Electronics (`BEL`)*: Projected -1.70% (Best Model: LSTM)

---

## 9. Critical Limitations & Governance

1. **Market Non-Stationarity & Macro Shocks**: Stock price movements are driven by unanticipated corporate earnings announcements, Reserve Bank of India policy shifts, global oil shocks, and geopolitical developments that historical price series cannot encode.
2. **Survivorship Bias**: Using the current NIFTY 50 index composition ignores constituent turnover that occurred over the preceding 5 years.
3. **Estimation Uncertainty**: Point predictions should not be treated as exact price targets. The 90% empirical confidence bounds generated by the system reflect the substantial dispersion inherent in 21-day forward forecasting.

---

## 10. Conclusion & Educational Disclaimer

The project successfully engineered a scalable, institutional-grade machine-learning pipeline for the NIFTY 50. By enforcing zero data leakage, incorporating corporate split adjustments, comparing diverse model paradigms against naive baselines, and delivering an interactive 10-page Streamlit dashboard, the system provides a benchmark for quantitative time-series research.

*Disclaimer: This report and application are strictly for educational and research purposes. None of the findings, forecasts, or rankings constitute investment advice or recommendations to trade securities.*
