"""
Trading Term Tooltip Component
===============================

Streamlit component for displaying interactive tooltips with AI-generated
trading term definitions.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Optional
from .term_definitions import get_term_definition


def render_term_with_tooltip(
    term: str,
    definition: Optional[str] = None,
    icon: str = "❓",
    icon_size: str = "0.7em",
    tooltip_width: str = "350px"
) -> None:
    """
    Render a trading term with an interactive hover tooltip.

    Args:
        term: The term to display
        definition: Optional pre-defined definition. If None, will fetch from API
        icon: The icon to display next to the term (default: ❓)
        icon_size: CSS font-size for the icon
        tooltip_width: CSS width for the tooltip popup
    """
    # Get definition if not provided
    if definition is None:
        definition = get_term_definition(term)

    # Generate unique ID for this tooltip
    tooltip_id = f"tooltip_{term.replace(' ', '_').replace('/', '_').lower()}"

    # Render HTML/CSS for tooltip
    st.markdown(f"""
    <style>
        .term-container-{tooltip_id} {{
            display: inline-flex;
            align-items: center;
            gap: 0.3em;
        }}

        .tooltip-icon-{tooltip_id} {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.2em;
            height: 1.2em;
            font-size: {icon_size};
            cursor: help;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50%;
            margin-left: 0.3em;
            transition: all 0.2s ease;
            position: relative;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .tooltip-icon-{tooltip_id}:hover {{
            transform: scale(1.15);
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
        }}

        .tooltip-popup-{tooltip_id} {{
            visibility: hidden;
            opacity: 0;
            width: {tooltip_width};
            background-color: #2d3748;
            color: #ffffff;
            text-align: left;
            border-radius: 8px;
            padding: 12px 16px;
            position: absolute;
            z-index: 9999;
            bottom: 125%;
            left: 50%;
            margin-left: calc(-{tooltip_width} / 2);
            transition: opacity 0.3s, visibility 0.3s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            font-size: 0.9em;
            line-height: 1.5;
        }}

        .tooltip-popup-{tooltip_id}::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #2d3748 transparent transparent transparent;
        }}

        .tooltip-icon-{tooltip_id}:hover .tooltip-popup-{tooltip_id} {{
            visibility: visible;
            opacity: 1;
        }}

        .tooltip-term-{tooltip_id} {{
            font-weight: 600;
            color: #667eea;
            margin-bottom: 6px;
            font-size: 1.05em;
        }}

        .tooltip-definition-{tooltip_id} {{
            color: #e2e8f0;
            line-height: 1.6;
        }}
    </style>

    <div class="term-container-{tooltip_id}">
        <span class="tooltip-icon-{tooltip_id}">
            {icon}
            <div class="tooltip-popup-{tooltip_id}">
                <div class="tooltip-term-{tooltip_id}">{term}</div>
                <div class="tooltip-definition-{tooltip_id}">{definition}</div>
            </div>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_inline_term_tooltip(
    label: str,
    term: str,
    definition: Optional[str] = None
) -> None:
    """
    Render a label with inline tooltip (e.g., "Portfolio Delta ❓")

    Args:
        label: The label text to display
        term: The term for definition lookup
        definition: Optional pre-defined definition
    """
    col1, col2 = st.columns([20, 1])

    with col1:
        st.markdown(f"**{label}**")

    with col2:
        render_term_with_tooltip(term, definition)


def render_metric_with_tooltip(
    label: str,
    value: str,
    term: str,
    delta: Optional[str] = None,
    definition: Optional[str] = None
):
    """
    Render a Streamlit metric with an adjacent tooltip.

    Args:
        label: Metric label
        value: Metric value
        term: Term for tooltip
        delta: Optional delta value for metric
        definition: Optional pre-defined definition
    """
    # Create container with label and tooltip
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5em; margin-bottom: 0.5em;">
        <span style="font-size: 0.875rem; color: rgb(49, 51, 63);">{label}</span>
    """, unsafe_allow_html=True)

    render_term_with_tooltip(term, definition, icon_size="0.6em")

    st.markdown("</div>", unsafe_allow_html=True)

    # Render the metric value
    if delta:
        st.metric("", value, delta, label_visibility="collapsed")
    else:
        st.metric("", value, label_visibility="collapsed")


def get_tooltip_html(term: str, definition: Optional[str] = None, icon: str = "❓", icon_size: str = "0.7em") -> str:
    """
    Get HTML string for inline tooltip (for use in markdown strings).

    Args:
        term: The term to display tooltip for
        definition: Optional pre-defined definition
        icon: The icon to display (default: ❓)
        icon_size: CSS font-size for the icon

    Returns:
        HTML string that can be embedded in markdown
    """
    if definition is None:
        definition = get_term_definition(term)

    tooltip_id = f"tooltip_{term.replace(' ', '_').replace('/', '_').lower()}"

    return f"""
    <style>
        .tooltip-icon-{tooltip_id} {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.2em;
            height: 1.2em;
            font-size: {icon_size};
            cursor: help;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50%;
            margin-left: 0.3em;
            transition: all 0.2s ease;
            position: relative;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .tooltip-icon-{tooltip_id}:hover {{
            transform: scale(1.15);
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
        }}
        .tooltip-popup-{tooltip_id} {{
            visibility: hidden;
            opacity: 0;
            width: 350px;
            background-color: #2d3748;
            color: #ffffff;
            text-align: left;
            border-radius: 8px;
            padding: 12px 16px;
            position: absolute;
            z-index: 9999;
            bottom: 125%;
            left: 50%;
            margin-left: -175px;
            transition: opacity 0.3s, visibility 0.3s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            font-size: 0.9em;
            line-height: 1.5;
        }}
        .tooltip-popup-{tooltip_id}::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #2d3748 transparent transparent transparent;
        }}
        .tooltip-icon-{tooltip_id}:hover .tooltip-popup-{tooltip_id} {{
            visibility: visible;
            opacity: 1;
        }}
    </style>
    <span class="tooltip-icon-{tooltip_id}">
        {icon}
        <div class="tooltip-popup-{tooltip_id}">
            <div style="font-weight: 600; color: #667eea; margin-bottom: 6px; font-size: 1.05em;">{term}</div>
            <div style="color: #e2e8f0; line-height: 1.6;">{definition}</div>
        </div>
    </span>
    """
