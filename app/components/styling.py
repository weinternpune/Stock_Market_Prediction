"""
styling.py
----------
Modern, theme-adaptive CSS styling and design system for NIFTY 50 Analytics Dashboard.
Engineered for institutional fintech research aesthetics (sleek dark/light mode compatibility,
fixed-height KPI cards, refined typography, and glassmorphism accents).
"""

import streamlit as st

def apply_custom_styles():
    """Injects modern, theme-adaptive CSS styles into the Streamlit application."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Modern Streamlit Container Spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
    }

    /* Fixed-Height Modern Metric KPI Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
        height: 140px;
        min-height: 140px;
        max-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        backdrop-filter: blur(10px);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 6px 22px rgba(99, 102, 241, 0.18);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0;
        white-space: normal;
        line-height: 1.25;
        min-height: 28px;
        display: flex;
        align-items: flex-start;
        overflow: hidden;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.15;
        margin: 2px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-value-compact {
        font-size: 1.15rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.25;
        margin: 2px 0;
        white-space: normal;
        word-break: break-word;
        overflow: visible;
    }
    .metric-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.78rem;
        min-height: 20px;
        margin: 0;
    }
    .metric-delta-pos {
        font-weight: 700;
        color: #10b981;
        display: inline-flex;
        align-items: center;
        gap: 3px;
    }
    .metric-delta-neg {
        font-weight: 700;
        color: #f43f5e;
        display: inline-flex;
        align-items: center;
        gap: 3px;
    }
    .metric-delta-neu {
        font-weight: 700;
        color: #94a3b8;
        display: inline-flex;
        align-items: center;
        gap: 3px;
    }
    .metric-subtext {
        color: #94a3b8;
        font-size: 0.75rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Section Headings with Gradient Accent */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 24px;
        margin-bottom: 14px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badges */
    .badge-model {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-sector {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 600;
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Top gainers / losers pills */
    .stock-chip-positive {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: transform 0.15s ease;
    }
    .stock-chip-positive:hover {
        transform: translateX(3px);
        background: rgba(16, 185, 129, 0.14);
    }
    .stock-chip-negative {
        background: rgba(244, 63, 94, 0.08);
        border: 1px solid rgba(244, 63, 94, 0.25);
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        transition: transform 0.15s ease;
    }
    .stock-chip-negative:hover {
        transform: translateX(3px);
        background: rgba(244, 63, 94, 0.14);
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Modern Sidebar Navigation Links */
    a[data-testid="stSidebarNavLink"] {
        border-radius: 8px !important;
        margin: 3px 0 !important;
        padding: 6px 12px !important;
        transition: all 0.18s ease !important;
    }
    a[data-testid="stSidebarNavLink"]:hover {
        background: rgba(99, 102, 241, 0.14) !important;
        transform: translateX(2px);
    }
    a[data-testid="stSidebarNavLink"][aria-current="page"] {
        background: rgba(99, 102, 241, 0.22) !important;
        border-left: 3px solid #6366f1 !important;
        font-weight: 700 !important;
    }

    /* Polished Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }

    /* Footer Note */
    .app-footer {
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        font-size: 0.78rem;
        color: #64748b;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def render_disclaimer():
    """Reserved for standalone methodology page."""
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.6); border-left: 4px solid #3b82f6; border-radius: 8px; padding: 14px 18px; margin: 16px 0; color: #cbd5e1; font-size: 0.85rem; line-height: 1.5;">
        <strong>⚠️ Analytical Notice:</strong> This platform is designed exclusively for quantitative research and educational exploration. 
        Predictions are empirical model outputs subject to market noise and non-stationarity. None of the content constitutes financial or investment advice.
    </div>
    """, unsafe_allow_html=True)
