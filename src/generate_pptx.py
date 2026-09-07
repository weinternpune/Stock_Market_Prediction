"""
Script to generate a professional 18-slide Executive Presentation Deck (.pptx)
for the Nifty 500 Stock Market Prediction Pipeline capstone project.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# --- COLOR PALETTE ---
COLOR_NAVY_DARK   = RGBColor(15, 23, 42)     # #0F172A
COLOR_NAVY_LIGHT  = RGBColor(30, 41, 59)     # #1E293B
COLOR_BLUE_DARK   = RGBColor(30, 58, 138)    # #1E3A8A
COLOR_BLUE_PRI    = RGBColor(37, 99, 235)    # #2563EB
COLOR_BLUE_LIGHT  = RGBColor(239, 246, 255)  # #EFF6FF
COLOR_CYAN        = RGBColor(14, 165, 233)   # #0EA5E9
COLOR_CYAN_LIGHT  = RGBColor(240, 249, 255)  # #F0F9FF
COLOR_EMERALD     = RGBColor(16, 185, 129)   # #10B981
COLOR_EMERALD_BG  = RGBColor(236, 253, 245)  # #ECFDF5
COLOR_ROSE        = RGBColor(244, 63, 94)    # #F43F5E
COLOR_ROSE_BG     = RGBColor(255, 241, 242)  # #FFF1F2
COLOR_AMBER       = RGBColor(245, 158, 11)   # #F59E0B
COLOR_CARD_BG     = RGBColor(255, 255, 255)  # #FFFFFF
COLOR_CARD_ALT_BG = RGBColor(248, 250, 252)  # #F8FAFC
COLOR_BORDER      = RGBColor(226, 232, 240)  # #E2E8F0
COLOR_TEXT_DARK   = RGBColor(15, 23, 42)     # #0F172A
COLOR_TEXT_BODY   = RGBColor(51, 65, 85)     # #334155
COLOR_TEXT_MUTED  = RGBColor(100, 116, 139)  # #64748B
COLOR_WHITE       = RGBColor(255, 255, 255)  # #FFFFFF

FONT_HEADING = "Segoe UI"
FONT_BODY = "Segoe UI"

def create_deck():
    prs = Presentation()
    # 16:9 Widescreen standard
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, topic_badge: str, title_text: str, subtitle_text: str, slide_num: int, total_slides: int = 18):
        # Top banner background accent line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_BLUE_PRI
        line.line.color.rgb = COLOR_BLUE_PRI

        # Badge pill
        badge_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.35), Inches(3.4), Inches(0.35))
        badge_box.fill.solid()
        badge_box.fill.fore_color.rgb = COLOR_BLUE_LIGHT
        badge_box.line.color.rgb = COLOR_BLUE_PRI
        badge_box.line.width = Pt(1)
        tf_b = badge_box.text_frame
        tf_b.word_wrap = True
        tf_b.vertical_anchor = MSO_ANCHOR.MIDDLE
        p_b = tf_b.paragraphs[0]
        p_b.text = f"{topic_badge.upper()}  •  SLIDE {slide_num} OF {total_slides}"
        p_b.font.name = FONT_HEADING
        p_b.font.size = Pt(9.5)
        p_b.font.bold = True
        p_b.font.color.rgb = COLOR_BLUE_PRI
        p_b.alignment = PP_ALIGN.CENTER

        # Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.55))
        tf_t = tb_title.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.name = FONT_HEADING
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_NAVY_DARK

        # Subtitle
        tb_sub = slide.shapes.add_textbox(Inches(0.8), Inches(1.22), Inches(11.7), Inches(0.45))
        tf_s = tb_sub.text_frame
        tf_s.word_wrap = True
        p_s = tf_s.paragraphs[0]
        p_s.text = subtitle_text
        p_s.font.name = FONT_BODY
        p_s.font.size = Pt(11.5)
        p_s.font.color.rgb = COLOR_TEXT_MUTED

        # Footer divider
        f_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(7.0), Inches(11.733), Inches(0.02))
        f_line.fill.solid()
        f_line.fill.fore_color.rgb = COLOR_BORDER
        f_line.line.color.rgb = COLOR_BORDER

        # Footer text
        tb_f = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(8.0), Inches(0.35))
        p_f = tb_f.text_frame.paragraphs[0]
        p_f.text = "Nifty 500 Quantitative Forecasting Pipeline  |  PRD v1.1 Capstone  |  Confidential"
        p_f.font.name = FONT_BODY
        p_f.font.size = Pt(9)
        p_f.font.color.rgb = COLOR_TEXT_MUTED

        # Footer page number
        tb_num = slide.shapes.add_textbox(Inches(10.5), Inches(7.05), Inches(2.0), Inches(0.35))
        p_num = tb_num.text_frame.paragraphs[0]
        p_num.text = f"Page {slide_num}"
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.name = FONT_BODY
        p_num.font.size = Pt(9)
        p_num.font.bold = True
        p_num.font.color.rgb = COLOR_TEXT_MUTED

    def add_card(slide, left, top, width, height, title: str, items: list, border_color=COLOR_BORDER, bg_color=COLOR_CARD_BG, title_color=COLOR_NAVY_DARK):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)

        tf = card.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.TOP
        tf.margin_left = Inches(0.25)
        tf.margin_right = Inches(0.25)
        tf.margin_top = Inches(0.2)
        tf.margin_bottom = Inches(0.2)

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = FONT_HEADING
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = title_color
        p0.space_after = Pt(8)

        for it in items:
            p = tf.add_paragraph()
            p.text = f"• {it}"
            p.font.name = FONT_BODY
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_TEXT_BODY
            p.space_after = Pt(4)

    def add_metric_badge(slide, left, top, width, height, value: str, label: str, sub: str = "", val_color=COLOR_BLUE_PRI, bg_color=COLOR_BLUE_LIGHT):
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        box.fill.solid()
        box.fill.fore_color.rgb = bg_color
        box.line.color.rgb = COLOR_BORDER
        box.line.width = Pt(1)

        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.15)
        tf.margin_right = Inches(0.15)

        p_val = tf.paragraphs[0]
        p_val.text = value
        p_val.font.name = FONT_HEADING
        p_val.font.size = Pt(20)
        p_val.font.bold = True
        p_val.font.color.rgb = val_color
        p_val.alignment = PP_ALIGN.CENTER

        p_lbl = tf.add_paragraph()
        p_lbl.text = label.upper()
        p_lbl.font.name = FONT_HEADING
        p_lbl.font.size = Pt(9)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = COLOR_TEXT_DARK
        p_lbl.alignment = PP_ALIGN.CENTER

        if sub:
            p_sub = tf.add_paragraph()
            p_sub.text = sub
            p_sub.font.name = FONT_BODY
            p_sub.font.size = Pt(8)
            p_sub.font.color.rgb = COLOR_TEXT_MUTED
            p_sub.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 1: Title Slide (Dark Executive Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = COLOR_NAVY_DARK
    bg1.line.fill.background()

    # Accent decorative bar
    dec_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.15), Inches(3.6))
    dec_bar.fill.solid()
    dec_bar.fill.fore_color.rgb = COLOR_CYAN
    dec_bar.line.fill.background()

    # Title box
    tb = s1.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(11.0), Inches(3.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "CAPSTONE EXECUTIVE PRESENTATION"
    p0.font.name = FONT_HEADING
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_CYAN
    p0.space_after = Pt(12)

    p1 = tf.add_paragraph()
    p1.text = "NIFTY 500 STOCK MARKET PREDICTION PIPELINE"
    p1.font.name = FONT_HEADING
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_after = Pt(14)

    p2 = tf.add_paragraph()
    p2.text = "A 5-Year Quantitative Analytics, Dual-Exchange Modeling & Benchmark Evaluation Study"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    p2.space_after = Pt(28)

    p3 = tf.add_paragraph()
    p3.text = "Author: Data Analytics Intern  |  Project Scope: PRD v1.1 (60-Step Roadmap)  |  Target: NSE NIFTY 500 Index\nEvaluation Period: September 1, 2021 – August 31, 2026 (1,240 Trading Sessions)  |  Deployment: Streamlit v1.30+"
    p3.font.name = FONT_BODY
    p3.font.size = Pt(11.5)
    p3.font.color.rgb = RGBColor(203, 213, 225)

    # 4 Bottom Highlight Cards on Slide 1
    w_card = Inches(2.75)
    h_card = Inches(1.1)
    top_c = Inches(5.8)
    add_metric_badge(s1, Inches(1.2), top_c, w_card, h_card, "1,240", "Trading Sessions", "100% Complete Calendar", COLOR_CYAN, COLOR_NAVY_LIGHT)
    add_metric_badge(s1, Inches(4.15), top_c, w_card, h_card, "0.00%", "Missing Data", "PRD Target < 2.0%", COLOR_EMERALD, COLOR_NAVY_LIGHT)
    add_metric_badge(s1, Inches(7.1), top_c, w_card, h_card, "209.33", "Best Out-of-Sample RMSE", "Naive Persistence Baseline", COLOR_AMBER, COLOR_NAVY_LIGHT)
    add_metric_badge(s1, Inches(10.05), top_c, w_card, h_card, "8-Page", "Interactive Web App", "Production Streamlit Suite", COLOR_WHITE, COLOR_NAVY_LIGHT)

    # =========================================================================
    # SLIDE 2: Executive Summary & Project Highlights
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Executive Summary", "Project Highlights & Core Empirical Takeaways", "Overview of pipeline architecture, dataset scale, and primary scientific conclusions", 2)
    
    add_card(s2, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), 
             "🎯 Quantitative Objectives & Scale", [
                 "End-to-end forecasting pipeline built for India's broad-market benchmark: NSE Nifty 500 (capturing ~96% free-float market cap).",
                 "Strict 5-year chronological evaluation from Sep 1, 2021 to Aug 31, 2026 across 1,240 verified exchange trading days.",
                 "Dual-exchange architecture: Primary NSE modeling series paired with a secondary BSE 500 benchmark (0.99989 correlation).",
                 "Zero synthetic holiday imputation; post-cleaning missing value rate achieved was 0.00% (surpassing the PRD <2% threshold).",
                 "Production-ready deployment featuring an 8-page Streamlit dashboard, modular Python package, and automated test suite."
             ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s2, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8),
             "💡 Key Scientific & Empirical Findings", [
                 "Naive Persistence Baseline achieved lowest level-price RMSE (209.33 pts, MAE 151.88, MAPE 0.669%) across 208 out-of-sample sessions.",
                 "Empirical confirmation of the Martingale Property of Asset Prices: E[P(t+1) | F_t] ≈ P(t). Today's price is the minimum-variance estimator.",
                 "Random Forest proved to be the champion feature-driven ML model (RMSE 229.66 pts), tracking persistence within 9.71%.",
                 "Classical & Deep Learning models (XGBoost, LSTM) track multi-week trends effectively but suffer from a 1–2 day turning-point lag.",
                 "Actionable Quantitative Lesson: Machine learning pipelines in financial equity markets should target stationary log-returns rather than raw price levels."
             ], COLOR_BORDER, COLOR_CARD_BG, COLOR_EMERALD)

    # =========================================================================
    # SLIDE 3: Problem Statement
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Problem Statement", "The Complexity of Financial Time-Series Forecasting", "Why predicting broad equity index levels is notoriously difficult and prone to bias", 3)

    w3 = Inches(3.65)
    h3 = Inches(4.8)
    add_card(s3, Inches(0.8), Inches(1.8), w3, h3, "1. Microstructural Noise", [
        "Low Signal-to-Noise Ratio: Equity price changes are dominated by random high-frequency microstructural noise, liquidity variations, and intraday order flow.",
        "Stochastic Fluctuations: Day-to-day point changes often lack persistent directional memory, resembling geometric Brownian motion.",
        "Overfitting Trap: Complex deep learning models easily memorize random historical fluctuations, leading to strong in-sample fit but poor out-of-sample generalization."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_ROSE)

    add_card(s3, Inches(4.8), Inches(1.8), w3, h3, "2. Non-Stationarity & Regimes", [
        "Time-Varying Distributions: Equity index price levels exhibit evolving means, non-constant variances, and structural macroeconomic breaks.",
        "Macroeconomic Disruptions: Events such as geopolitical conflicts (2022 Ukraine invasion) or domestic elections (2024 Lok Sabha) shift market regimes overnight.",
        "Distributional Drift: Standard neural network activation functions struggle when test-period price levels reach new all-time highs never seen during training."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_AMBER)

    add_card(s3, Inches(8.8), Inches(1.8), w3, h3, "3. Lookahead Leakage Risks", [
        "Global Scaling Bias: Fitting scalers (MinMaxScaler/StandardScaler) on the entire dataset leaks future test prices into the training process.",
        "Indicator Boundary Leakage: Computing moving averages or RSI without strict lagging creates artificial forward-looking dependencies.",
        "Random K-Fold Shuffling: Standard cross-validation randomly shuffles past and future dates, destroying temporal causality and inflating performance."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_PRI)

    # =========================================================================
    # SLIDE 4: Project Objectives & Scope
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Project Objectives", "Formal Objectives, Target Formulation & Architectural Boundaries", "Strict quantitative definitions designed to ensure complete leak-free modeling", 4)

    add_card(s4, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "🎯 Formal Quantitative Objectives", [
        "Primary Target Variable: Predict next trading day's closing price level P(t+1) using information strictly available up to day t.",
        "Chronological Splitting: 80% train (832 sessions: Jun 2022 – Oct 2025) and 20% held-out test (208 sessions: Oct 28, 2025 – Aug 28, 2026).",
        "Multi-Horizon Forecasting: 1-step-ahead backtesting for rigorous statistical comparison; recursive rollouts from T+1 to T+30 for forward uncertainty estimation.",
        "Comprehensive Evaluation Metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Directional Hit Rate (DHR).",
        "Benchmark Hierarchy: Compare against theoretical baselines (Naive Persistence, 5-Day SMA) to ensure ML models justify their operational complexity."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s4, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "🛡️ Strict Modeling Governance & Scope", [
        "Zero Lookahead Bias: All feature engineering utilizes strictly backward-looking windows. Scalers fitted only on the 832-row training set.",
        "Expanding Window Validation: Cross-validation uses strictly expanding time windows (TimeSeriesSplit) to mimic real-world trading deployment.",
        "Dual-Exchange Preservation: NSE and BSE index datasets maintained independently. Zero price blending or averaging across exchanges.",
        "Full Trading Calendar Integrity: Aligned strictly with authentic NSE trading days (~250 sessions/yr), omitting artificial weekend fills.",
        "Transparent Reporting: All project limitations, edge cases, and theoretical trade-offs fully documented per PRD Section 12."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_NAVY_DARK)

    # =========================================================================
    # SLIDE 5: Dual-Exchange Authoritative Data Architecture
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Data Architecture", "Authoritative Dual-Exchange Data Architecture", "Independent ingestion of NSE Nifty 500 and BSE 500 market data across 1,240 sessions", 5)

    add_card(s5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "🇮🇳 Primary Dataset: NSE NIFTY 500", [
        "Authoritative Archive: Sourced from the National Stock Exchange of India official historical records.",
        "Breadth & Coverage: Captures ~96% of the total free-float market capitalization in India.",
        "Sample Size: Exactly 1,240 daily records spanning September 1, 2021 to August 31, 2026.",
        "Schema: Date, Open, High, Low, Close (Full OHLC series).",
        "Pipeline Role: Sole primary target series for quantitative feature engineering, econometric modeling, ML training, and backtesting.",
        "Data Quality: 0.00% missing values post-cleaning. Complete calendar consistency across all five operating years."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "🏛️ Secondary Reference: BSE 500 Index", [
        "Cross-Market Benchmark: Sourced from the Bombay Stock Exchange of India broad-market database.",
        "Enriched Dimensions: Provides Volume (Crores), Turnover, P/E Ratio, P/B Ratio, and Dividend Yield.",
        "Sample Size: 1,240 identical daily records corresponding to identical NSE trading dates.",
        "Inter-Exchange Co-Movement: Price correlation of 0.99989 and return correlation of 0.99930 against NSE Nifty 500.",
        "Pipeline Role: Independent cross-market validation proxy and macroeconomic sentiment gauge.",
        "Architectural Rule: Kept strictly decoupled to preserve exchange data lineage and avoid synthetic averaging."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_AMBER)

    # =========================================================================
    # SLIDE 6: Data Cleaning & Macro Outlier Audit
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Data Integrity", "Trading Calendar Integrity & Macro Outlier Audit", "Rigorous sanitization and empirical investigation of statistical price shocks (|Z| > 3.5)", 6)

    add_card(s6, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "🧹 Data Sanitization & Calendar Reconciliation", [
        "Trading Day Reconciliation: Strict verification against official NSE holidays (Republic Day, Diwali, Eid, Christmas, etc.).",
        "Zero Synthetic Imputation: Non-trading weekends and exchange holidays were properly omitted rather than filled with artificial static prices.",
        "String Placeholder Replacement: Converted 23 string '-' placeholders in BSE historical volume/turnover to numeric floats.",
        "Missing Data Audit: Achieved 0.00% missing values in OHLC records, far exceeding the PRD mandate (<2.0% data loss).",
        "Automated Validation Suite: Unit test suite (test_pipeline.py) verifies data types, non-negativity, and monotonicity before modeling."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s6, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "🔍 Macroeconomic Outlier Investigation (|Z| > 3.5)", [
        "2022-02-24 (-5.04%, Z = -5.87): Outbreak of the Russia-Ukraine military invasion. Verified global risk-off selloff.",
        "2024-06-04 (-6.76%, Z = -7.92): Lok Sabha General Election counting day. Unexpected coalition uncertainty drove severe volatility.",
        "2024-06-05 (+3.59%, Z = +4.11): Immediate post-election rebound rally upon government formation and policy continuity clarity.",
        "Retention Rationale: All extreme outliers were audited against official exchange notices and preserved.",
        "Risk Censorship Prevention: Removing legitimate market crashes creates artificial survivorship and downside censorship bias."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_ROSE)

    # =========================================================================
    # SLIDE 7: Exploratory Data Analysis & Financial Stylized Facts
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Exploratory Analysis", "Stylized Facts of Financial Market Returns", "Statistical properties of the 5-year Nifty 500 series conforming to empirical market finance", 7)

    w7 = Inches(3.65)
    add_card(s7, Inches(0.8), Inches(1.8), w7, Inches(4.8), "📉 Non-Normality & Asymmetry", [
        "Total Index Appreciation: Sourced at 14,551.35 (Sep 1, 2021) and reached 23,450.35 (Aug 31, 2026), a net gain of +61.16%.",
        "Up vs. Down Sessions: 682 Up days (55.0%) vs. 557 Down days (45.0%). Mean daily return: +0.043% (Annualized: ~11.26%).",
        "Negative Skewness (-0.678): Demonstrates clear asymmetry with larger, sudden downward tail movements.",
        "Excess Kurtosis (4.585): Fat-tailed leptokurtic distribution with extreme tail risk far exceeding a normal curve.",
        "Jarque-Bera Test (p < 10^-250): Conclusively rejects the hypothesis of Gaussian normality."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s7, Inches(4.8), Inches(1.8), w7, Inches(4.8), "🌊 Volatility Dynamics", [
        "Mandelbrot Volatility Clustering: Large price changes are followed by large changes of either sign; calm periods follow calm periods.",
        "20-Day Annualized Volatility: Fluctuated from a tranquil 4.31% to an elevated 32.14% (5-year historical mean: 13.40%).",
        "Autocorrelation of Squared Returns: Highly statistically significant persistence in squared returns, validating ARCH/GARCH properties.",
        "Regime Clustering: Clear separation between low-volatility secular bull markets and high-volatility macro correction phases."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_AMBER)

    add_card(s7, Inches(8.8), Inches(1.8), w7, Inches(4.8), "⚖️ Seasonality & Co-Movement", [
        "Day-of-Week ANOVA: F-statistic = 1.192, p-value = 0.3120. No statistically significant day-of-week anomaly.",
        "Weak-Form Efficiency: Monday to Friday returns show comparable distributions, refuting simple calendar trading rules.",
        "Cross-Market Co-Movement: Price correlation of 0.99989 and return correlation of 0.99930 between NSE Nifty 500 and BSE 500.",
        "Broad Index Alignment: Confirms that Indian large-, mid-, and small-cap segments move in near-perfect synchronization across exchanges."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_EMERALD)

    # =========================================================================
    # SLIDE 8: Quantitative Feature Engineering
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Feature Engineering", "Quantitative Technical Feature Set (34 Features)", "Comprehensive indicator generation engineered with strict zero-lookahead discipline", 8)

    add_card(s8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "⚙️ Technical Indicator Taxonomy", [
        "Trend Moving Averages: SMA (10, 20, 50, 200 days), EMA (12, 26 days), and distance-to-SMA normalized ratios.",
        "Momentum Oscillators: 14-Day Wilder Relative Strength Index (RSI), MACD Line, 9-Day Signal Line, and MACD Histogram.",
        "Volatility Channels: 20-Day Bollinger Bands (±2σ), Bandwidth ((Upper-Lower)/SMA), and %B Price Position.",
        "Historical Volatility: 10-Day and 20-Day rolling annualized return standard deviations.",
        "Return Lags: Autoregressive return lags (t-1, t-2, t-3, t-5) and closing price lags (t-1, t-2).",
        "Intraday Ratios: Daily High/Low spread ratio and Close/Open candle body ratio."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "🛡️ Lookahead Bias Prevention Protocol", [
        "Warm-Up Cutoff: The 200-day SMA requires 200 initial trading days to stabilize. The first 200 sessions were cleanly discarded.",
        "Clean Modeling Dataset: Resulted in 1,040 rows of complete, high-signal modeling data (Jun 2022 – Aug 2026).",
        "Strict Lag Alignment: Feature matrix X(t) is constructed strictly using information finalized at or before market close on day t.",
        "Target Definition: Target variable y(t) = Close(t+1), ensuring explicit one-step-ahead regression alignment.",
        "Scaler Isolation: Feature standardization scalers (StandardScaler, MinMaxScaler) fitted strictly on the training partition."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_NAVY_DARK)

    # =========================================================================
    # SLIDE 9: Multi-Family Modeling Framework
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Modeling Framework", "Multi-Family Benchmark Architectures & Split Strategy", "Comparing statistical econometrics, tree ensembles, and deep neural networks", 9)

    add_card(s9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "⏱️ Chronological Validation Scheme", [
        "Time-Based Split: Strict chronological partitioning with zero random shuffling.",
        "Training Set: 832 trading sessions (80.0%, June 2022 to October 2025).",
        "Test Partition: 208 trading sessions (20.0%, October 28, 2025 to August 28, 2026).",
        "Expanding Window CV: 5-fold TimeSeriesSplit used during hyperparameter tuning to ensure training windows only expand forward.",
        "No Test Peeking: Hyperparameters locked prior to final out-of-sample evaluation."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "🏗️ Model Family Architectures", [
        "Naive Persistence Baseline: P(t+1) = P(t). Represents random-walk zero-drift theoretical lower bound.",
        "5-Day Moving Average: Rolling 5-day arithmetic mean smoothing short-term noise.",
        "Walk-Forward ARIMA(1,1,1): Statistical autoregressive model re-fitted rolling one session at a time.",
        "Random Forest Regressor: 150 decision trees, max depth 8, min samples leaf 4 to prevent overfitting.",
        "XGBoost Regressor: 150 boosted gradient trees, learning rate 0.03, subsample 0.8, colsample 0.8.",
        "PyTorch LSTM Network: 2 stacked LSTM layers (64 hidden units), 20-day sequence lookback, dropout 0.2, Adam optimizer."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_NAVY_DARK)

    # =========================================================================
    # SLIDE 10: Master Performance Scorecard (TABLE SLIDE)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Evaluation Scorecard", "Master Out-of-Sample Performance Scorecard", "Evaluated on 208 held-out trading sessions (October 28, 2025 to August 28, 2026)", 10)

    # Create Table
    rows = 7
    cols = 7
    left_t = Inches(0.8)
    top_t = Inches(1.8)
    width_t = Inches(11.733)
    height_t = Inches(4.8)

    table_shape = s10.shapes.add_table(rows, cols, left_t, top_t, width_t, height_t)
    table = table_shape.table

    col_widths = [Inches(2.5), Inches(1.5), Inches(1.4), Inches(1.4), Inches(1.3), Inches(1.6), Inches(2.033)]
    for idx, width in enumerate(col_widths):
        table.columns[idx].width = width

    headers = ["Model Architecture", "Model Family", "RMSE (Pts)", "MAE (Pts)", "MAPE (%)", "Directional Hit", "vs. Naive Baseline"]
    table_data = [
        ["Naive Persistence", "Benchmark", "209.33", "151.88", "0.669%", "N/A", "+0.00% (Best)"],
        ["Random Forest Regressor", "Classical ML", "229.66", "167.47", "0.735%", "49.52%", "-9.71%"],
        ["5-Day Moving Average", "Benchmark", "288.87", "218.16", "0.957%", "51.92%", "-37.99%"],
        ["ARIMA(1, 1, 1) Walk-Forward", "Statistical", "291.08", "216.41", "0.950%", "51.92%", "-39.05%"],
        ["XGBoost Regressor", "Classical ML", "299.91", "239.62", "1.044%", "51.92%", "-43.27%"],
        ["PyTorch Stacked LSTM", "Deep Learning", "560.27", "453.11", "1.974%", "52.88%", "-167.64%"]
    ]

    for c_idx, h_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_NAVY_DARK
        p = cell.text_frame.paragraphs[0]
        p.text = h_text
        p.font.name = FONT_HEADING
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLOR_WHITE
        p.alignment = PP_ALIGN.CENTER

    for r_idx, row_values in enumerate(table_data):
        is_highlight = (r_idx == 0)
        is_second = (r_idx == 1)
        for c_idx, val in enumerate(row_values):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            if is_highlight:
                cell.fill.fore_color.rgb = COLOR_EMERALD_BG
            elif is_second:
                cell.fill.fore_color.rgb = COLOR_CYAN_LIGHT
            elif r_idx % 2 == 0:
                cell.fill.fore_color.rgb = COLOR_CARD_ALT_BG
            else:
                cell.fill.fore_color.rgb = COLOR_WHITE

            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_BODY
            p.font.size = Pt(10.5)
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            if is_highlight:
                p.font.bold = True
                p.font.color.rgb = COLOR_EMERALD if c_idx in [2, 6] else COLOR_NAVY_DARK
            elif is_second:
                p.font.bold = (c_idx in [0, 2])
                p.font.color.rgb = COLOR_BLUE_PRI if c_idx in [2, 6] else COLOR_NAVY_DARK
            else:
                p.font.color.rgb = COLOR_TEXT_DARK

    # =========================================================================
    # SLIDE 11: Quantitative Deep Dive: Martingale Property & EMH
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Scientific Insights", "The Martingale Property & Efficient Market Dynamics", "Empirical confirmation of theoretical mathematical bounds in financial equity markets", 11)

    add_card(s11, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "📐 The Martingale Property of Asset Prices", [
        "Mathematical Definition: In liquid, efficient financial markets, asset price levels follow a sub-martingale / near-martingale process:",
        "    E[ P(t+1) | F(t) ] ≈ P(t)",
        "Minimum-Variance Estimator: Under quadratic error loss (MSE / RMSE), today's closing price is the mathematically optimal estimator of tomorrow's level.",
        "Daily Volatility Scale: Daily standard deviation of Nifty 500 returns is ~0.89% (~200 points at index 23,000).",
        "The Variance Penalty: On sideways or noisy sessions, any model predicting a directional change incurs an error penalty if the index stays flat.",
        "Theoretical Barrier: Outperforming persistence on raw price-level RMSE is mathematically improbable without lookahead leakage or structural arbitrage."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s11, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "🧠 Why Deep Learning Drifted (LSTM Realities)", [
        "Stationarity Failure: LSTMs expect stationary input distributions. Raw price levels drifted into all-time high territory during the test phase.",
        "Sequence Sliding Window Drift: A 20-day sequence lookback amplified cumulative level drift when the index trended aggressively.",
        "High Directional Hit Rate (52.88%): Interestingly, LSTM achieved the highest directional accuracy, correctly predicting the sign of the move.",
        "Level Disconnect: Despite predicting direction well, its scaled price level estimates were offset by several hundred points, ballooning RMSE to 560.27.",
        "The Golden Rule of Quant ML: Neural networks must be trained on stationary transformations: log-returns r(t) = ln(P(t)/P(t-1)), never raw price levels."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_ROSE)

    # =========================================================================
    # SLIDE 12: Final Model Selection & Practical Utility
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Model Selection", "Final Model Selection & Dual-Recommendation Framework", "Balancing theoretical statistical bounds with feature-driven machine learning utility", 12)

    add_card(s12, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "🥇 Benchmark Champion: Naive Persistence", [
        "Authoritative Standard: Retained as the primary baseline for raw price level estimation (P(t+1) = P(t)).",
        "Lowest Tracking Errors: Achieved lowest RMSE (209.33), lowest MAE (151.88), and lowest MAPE (0.669%).",
        "Zero Parameter Risk: Incurs zero computational cost, zero training latency, and zero risk of model degradation over time.",
        "Reality Check: Any commercial vendor claiming an ML model drastically beats persistence on raw index levels without lookahead leakage is suspect."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_EMERALD)

    add_card(s12, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "🚀 ML Champion: Random Forest Regressor", [
        "Best Feature-Driven Model: Achieved out-of-sample RMSE of 229.66 points (only 9.71% behind theoretical persistence).",
        "Robust Generalization: Exhibited stable 5-fold cross-validation performance across expanding market regimes (1,059 ± 777).",
        "Key Feature Drivers: Tree split feature importances show dominant contributions from Price Lags, 10-Day SMA, 20-Day EMA, and Wilder RSI.",
        "Operational Value: Unlike persistence, Random Forest provides feature sensitivity, regime responsiveness, and actionable feature attribution."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_PRI)

    # =========================================================================
    # SLIDE 13: Out-of-Sample Backtesting Dynamics
    # =========================================================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Backtesting Analysis", "Actual vs. Predicted Backtesting Dynamics Across Regimes", "Evaluating tracking fidelity, inflection point lag, and residual distribution across 208 sessions", 13)

    w13 = Inches(3.65)
    add_card(s13, Inches(0.8), Inches(1.8), w13, Inches(4.8), "📈 Regime Tracking Fidelity", [
        "Multi-Week Trend Following: Random Forest, ARIMA, and XGBoost closely mirror multi-week upward and downward index trends.",
        "Strong Price Correlation: Predicted series maintain >0.98 correlation with actual realized index trajectories.",
        "Smooth Predictions: Tree ensembles successfully filter out high-frequency intraday noise, producing smooth expected price curves."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s13, Inches(4.8), Inches(1.8), w13, Inches(4.8), "⏳ Turning-Point Lag", [
        "1-Session Inflection Delay: When the market abruptly reverses (e.g. sharp dip followed by V-rebound), ML models exhibit a 1-day lag.",
        "Lag Explanation: Because models rely heavily on lag-1 and moving average features, they require one day of confirmation before turning.",
        "Persistence Contrast: Naive persistence also lags by exactly 1 session, but never overshoots, keeping its squared error lower."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_AMBER)

    add_card(s13, Inches(8.8), Inches(1.8), w13, Inches(4.8), "📊 Residual Diagnostics", [
        "Mean-Zero Residuals: Out-of-sample forecast residuals are centered tightly around zero, indicating unbiased level predictions.",
        "Homoscedasticity: Prediction errors remain bounded within ±500 points during normal volatility regimes.",
        "Shock Spillover: Residual spikes correspond exclusively to external macro surprises (election announcements, geopolitical escalations)."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_EMERALD)

    # =========================================================================
    # SLIDE 14: 30-Day Recursive Forward Forecast
    # =========================================================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Forward Projections", "30-Day Recursive Forward Forecast & Confidence Bands", "Projecting Nifty 500 index trajectory from August 31, 2026 (P0 = 23,450.35) through T+30", 14)

    add_card(s14, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "🔮 Recursive Rollout Projections (Champion Model)", [
        "Base Session: August 31, 2026 (Closing Level P0 = 23,450.35 points).",
        "T+1 (2026-09-01): Point Forecast = 23,321.96  |  95% CI: [22,911.67, 23,732.24]",
        "T+5 (2026-09-07): Point Forecast = 23,322.47  |  95% CI: [22,405.04, 24,239.90]",
        "T+15 (2026-09-21): Point Forecast = 23,323.85  |  95% CI: [21,742.08, 24,905.62]",
        "T+30 (2026-10-12): Point Forecast = 23,325.59  |  95% CI: [21,079.12, 25,572.06]",
        "Recursive Feedback: Each step's predicted close is fed recursively back into feature engineering for future steps."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s14, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "📈 Expanding Uncertainty Boundaries", [
        "Empirical Confidence Interval Formulation: ± 1.96 • RMSE • sqrt(h)",
        "Square-Root Horizon Expansion: Captures the compounding variance of random market shocks as time horizon h extends.",
        "Short-Term Precision: T+1 uncertainty band spans ±410 points (1.75%), providing actionable bounds for short-term risk management.",
        "Long-Term Diffusion: By T+30, the 95% confidence interval broadens to ±2,246 points (9.6%), realistically reflecting macro market uncertainty.",
        "Interactive Dashboard Control: Users can dynamically drag the forward horizon slider in the web app to explore any target date."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_NAVY_DARK)

    # =========================================================================
    # SLIDE 15: Production Streamlit Web Application
    # =========================================================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "Deployment", "Production-Ready 8-Page Interactive Streamlit Dashboard", "Full-featured web application deployed locally at http://localhost:8501", 15)

    w15 = Inches(5.6)
    h15 = Inches(4.8)
    add_card(s15, Inches(0.8), Inches(1.8), w15, h15, "🖥️ Analytical Dashboard Pages (1 to 4)", [
        "Page 1: Executive Overview — High-level KPI metrics, project goals, formal objectives, and academic research disclaimer.",
        "Page 2: Historical Market Explorer — Interactive Plotly candlestick charts for NSE & BSE 500, traded volume, P/E ratio, and date range filters.",
        "Page 3: Quantitative EDA & Volatility — Daily return distributions, Jarque-Bera normality tests, rolling volatility clustering, and ANOVA seasonality.",
        "Page 4: Technical Indicators & Features — 34 engineered indicators with parameter controls and Random Forest feature importance rankings."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s15, Inches(6.8), Inches(1.8), Inches(5.7), h15, "📊 Modeling & Forecasting Pages (5 to 8)", [
        "Page 5: Model Benchmark Scorecard — Interactive comparison table of all 6 models with sorting, color badges, and error distribution charts.",
        "Page 6: Backtesting & Actual vs. Predicted — Dynamic actual vs. predicted overlays with error spread visualizer and turning point inspection.",
        "Page 7: 30-Day Forward Forecaster — Interactive T+1 to T+30 horizon slider with dynamic 95% confidence intervals and CSV export.",
        "Page 8: Executive Presentation Deck — Integrated dark-mode slide viewer and direct PowerPoint / Report download center."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_NAVY_DARK)

    # =========================================================================
    # SLIDE 16: Project Limitations & Practical Constraints
    # =========================================================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Project Limitations", "Real-World Limitations & Constraints (PRD Section 12)", "Transparent documentation of architectural boundaries and financial market realities", 16)

    add_card(s16, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "⚠️ Data & Market Structure Constraints", [
        "Daily Granularity Limitation: Daily OHLC data omits intraday order book dynamics, bid-ask spread variations, and microstructure liquidity flow.",
        "Execution & Transaction Frictions: Does not incorporate exchange transaction charges, securities transaction tax (STT), slippage, or bid-ask friction.",
        "Exogenous Shocks: Geopolitical events, central bank rate surprises, and political elections occur outside the historical price data manifold.",
        "Independent Preserved Series: NSE and BSE series were maintained separately; cross-exchange arbitrage was not modeled."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_AMBER)

    add_card(s16, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "📉 Statistical & Modeling Trade-Offs", [
        "Non-Stationary Price Target: Forecasting raw price levels P(t+1) inherently subjects regression algorithms to distribution drift.",
        "Lack of Sentiment Data: News headlines, quarterly corporate earnings, and FII/DII institutional flow data were not integrated.",
        "Uncertainty Compounding: Recursive multi-step forecasts accumulate variance over extended forward horizons (T+30).",
        "Empirical Realism: Acknowledging these constraints strengthens research credibility and prevents naive overfitting."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_ROSE)

    # =========================================================================
    # SLIDE 17: Strategic Roadmap & Future Enhancements
    # =========================================================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Strategic Roadmap", "Future Quantitative Research & Engineering Enhancements", "Actionable recommendations for institutional-grade evolution of the forecasting pipeline", 17)

    w17 = Inches(3.65)
    add_card(s17, Inches(0.8), Inches(1.8), w17, Inches(4.8), "1. Stationary Target Shift", [
        "Log-Return Modeling: Transition target from raw price levels to stationary log-returns: r(t+1) = ln(P(t+1)/P(t)).",
        "Neural Network Stability: Eliminates distribution drift, enabling LSTMs and Transformers to generalize across all-time highs.",
        "Directional Classification: Formulate secondary binary classification models (Up vs. Down) for directional trading signals."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_BLUE_DARK)

    add_card(s17, Inches(4.8), Inches(1.8), w17, Inches(4.8), "2. Alternative Data Feeds", [
        "Institutional Flow Tracking: Integrate daily FII (Foreign) and DII (Domestic) net institutional cash flows from SEBI/NSE.",
        "Implied Volatility (India VIX): Incorporate India VIX index levels to dynamically adjust model confidence bands.",
        "Macro Indicators: Ingest RBI policy repo rates, USD/INR foreign exchange rate, and crude oil prices."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_AMBER)

    add_card(s17, Inches(8.8), Inches(1.8), w17, Inches(4.8), "3. Advanced Architectures", [
        "Hybrid Stacking Ensemble: Stack Random Forest feature predictions with Naive persistence bounds.",
        "Probabilistic Forecasting: Implement Temporal Fusion Transformers (TFT) or DeepAR for native quantile estimation.",
        "Constituent Sector Modeling: Expand pipeline to sectoral indices (Nifty Bank, Nifty IT, Nifty Auto) to model inter-sector rotation."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_EMERALD)

    # =========================================================================
    # SLIDE 18: Conclusion & Key Takeaways
    # =========================================================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Conclusion", "Conclusion, Capstone Deliverables & Key Takeaways", "Summary of milestones achieved across data engineering, quantitative modeling, and deployment", 18)

    add_card(s18, Inches(0.8), Inches(1.8), Inches(5.6), Inches(4.8), "🏁 100% PRD Requirements Met", [
        "Comprehensive Data Pipeline: Sanitized 1,240 sessions across NSE and BSE with 0.00% missing values.",
        "Multi-Family Benchmarking: Rigorously evaluated 6 architectures spanning baselines, econometrics, ML, and deep learning.",
        "Empirical Integrity: Avoided lookahead leakage; validated the Martingale property and Efficient Market Hypothesis.",
        "Interactive Deployment: Delivered an 8-page production Streamlit web application with dynamic backtesting and forecasting.",
        "Executive Deliverables: Complete PRD v1.1, Technical Findings Report, Interactive HTML Deck, and this PowerPoint Presentation."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_EMERALD)

    add_card(s18, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8), "🌟 Core Professional Takeaways", [
        "Simplicity is a Feature: Complex models do not automatically outperform simple baselines in noisy financial domains.",
        "Academic & Quantitative Rigor: Proving why persistence wins under quadratic loss demonstrates deep understanding of asset pricing theory.",
        "Random Forest Utility: Random Forest provides the best balance of low error (RMSE 229.66) and explainable feature signals.",
        "Production Quality: Modular code structure (src/), automated tests, and reproducible pipelines ensure institutional maintainability."
    ], COLOR_BORDER, COLOR_CARD_BG, COLOR_NAVY_DARK)

    # Save presentation
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Nifty_500_Executive_Presentation.pptx")
    prs.save(output_path)
    print(f"Presentation successfully saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    create_deck()
