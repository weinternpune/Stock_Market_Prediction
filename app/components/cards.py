"""
cards.py
--------
Reusable visual cards and KPI metric components for Streamlit pages.
Strictly eliminates raw HTML code-block leaks by stripping indentation and
guarantees identical card dimensions across all columns.
"""

import streamlit as st

def metric_card(title: str, value: str, delta: str = None, is_positive: bool = True, subtext: str = None):
    """
    Renders a modern, fixed-height metric card.
    Indentation is stripped so Streamlit markdown never interprets inner tags as code blocks.
    """
    delta_part = ""
    if delta:
        if is_positive is None:
            delta_class = "metric-delta-neu"
            arrow = "■"
        else:
            delta_class = "metric-delta-pos" if is_positive else "metric-delta-neg"
            arrow = "▲" if is_positive else "▼"
        delta_part = f'<span class="{delta_class}">{arrow} {delta}</span>'
        
    subtext_part = f'<span class="metric-subtext">{subtext}</span>' if subtext else ""
    
    # If neither delta nor subtext exists, provide a non-breaking space to keep identical vertical balance
    footer_content = f"{delta_part} {subtext_part}".strip()
    if not footer_content:
        footer_content = '<span class="metric-subtext">&nbsp;</span>'
        
    # If the text is long or represents a range interval, apply compact styling to prevent ellipsis truncation
    val_str = str(value)
    is_compact = len(val_str) > 12 or "–" in val_str or " - " in val_str
    val_class = "metric-value-compact" if is_compact else "metric-value"
        
    html = (
        f'<div class="metric-card">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="{val_class}">{value}</div>'
        f'<div class="metric-footer">{footer_content}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_stock_pill(rank: int, symbol: str, company: str, expected_return: float, current_p: float, pred_p: float, model: str):
    """Renders a sleek performance card for top gainers/losers."""
    is_pos = expected_return >= 0
    chip_class = "stock-chip-positive" if is_pos else "stock-chip-negative"
    ret_color = "#10b981" if is_pos else "#f43f5e"
    sign = "+" if is_pos else ""
    
    html = (
        f'<div class="{chip_class}">'
        f'<div style="display: flex; justify-content: space-between; align-items: center;">'
        f'<div>'
        f'<span style="font-weight: 800; color: #f8fafc; font-size: 0.95rem;">#{rank} {symbol}</span>'
        f'<span style="color: #94a3b8; font-size: 0.78rem; margin-left: 6px;">({company[:22]})</span>'
        f'</div>'
        f'<div style="font-weight: 800; font-size: 1.05rem; color: {ret_color};">'
        f'{sign}{expected_return:.2f}%'
        f'</div>'
        f'</div>'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-top: 8px; font-size: 0.78rem; color: #94a3b8;">'
        f'<div>₹{current_p:,.2f} → <span style="color: #f1f5f9; font-weight: 600;">₹{pred_p:,.2f}</span></div>'
        f'<div class="badge-model">{model}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
