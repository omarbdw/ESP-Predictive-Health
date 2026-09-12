"""Shared Streamlit presentation helpers."""

from __future__ import annotations

import streamlit as st


def apply_app_theme() -> None:
    """Apply the shared operational UI theme."""
    st.markdown(
        """
        <style>
        :root {
            --esp-ink: #18323b;
            --esp-muted: #61737a;
            --esp-teal: #147d78;
            --esp-teal-soft: #e5f3f1;
            --esp-amber: #c9821b;
            --esp-line: #dce5e5;
            --esp-paper: #fbfcfb;
        }
        .stApp { background: var(--esp-paper); }
        [data-testid="stSidebar"] { background: #f1f6f5; border-right: 1px solid var(--esp-line); }
        [data-testid="stSidebar"] h1 { color: var(--esp-ink); letter-spacing: 0; }
        h1, h2, h3 { color: var(--esp-ink); letter-spacing: 0; }
        h1 { font-size: 2.1rem; margin-bottom: .2rem; }
        h2 { font-size: 1.35rem; }
        .esp-kicker { color: var(--esp-teal); font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
        .esp-subtitle { color: var(--esp-muted); margin: 0 0 1.5rem; }
        .esp-brand { padding: .5rem 0 1rem; border-bottom: 1px solid var(--esp-line); margin-bottom: 1rem; }
        .esp-brand-title { color: var(--esp-ink); font-size: 1.15rem; font-weight: 750; }
        .esp-brand-note { color: var(--esp-muted); font-size: .78rem; margin-top: .2rem; }
        .esp-section-label { color: var(--esp-muted); font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; margin: 1rem 0 .3rem; }
        .esp-status { padding: .7rem .8rem; border: 1px solid var(--esp-line); border-radius: 8px; background: white; margin-top: .8rem; }
        .esp-status strong { color: var(--esp-ink); }
        div[data-testid="stMetric"] { background: white; border: 1px solid var(--esp-line); border-radius: 8px; padding: .8rem; }
        div[data-testid="stMetricLabel"] { color: var(--esp-muted); }
        .stButton > button[kind="primary"] { background: var(--esp-teal); border-color: var(--esp-teal); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(kicker: str, title: str, description: str) -> None:
    """Render a consistent page heading."""
    st.markdown(f'<div class="esp-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="esp-subtitle">{description}</p>', unsafe_allow_html=True)


def render_sidebar_status() -> None:
    """Show the current in-memory pipeline state in the sidebar."""
    standardized = st.session_state.get("standardized_import")
    engineered = st.session_state.get("engineered_features")
    events = st.session_state.get("detected_events")
    baseline = st.session_state.get("baseline_profile")
    st.sidebar.markdown('<div class="esp-section-label">Current workspace</div>', unsafe_allow_html=True)
    if standardized is None:
        st.sidebar.markdown('<div class="esp-status"><strong>No dataset loaded</strong><br><small>Start with Import Data.</small></div>', unsafe_allow_html=True)
        return
    rows = len(standardized)
    st.sidebar.markdown(
        f'<div class="esp-status"><strong>{rows:,} telemetry rows</strong><br>'
        f'<small>Standardized data ready</small></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        " | ".join(
            (
                "Features ready" if engineered is not None else "Features pending",
                "Events ready" if events is not None else "Events pending",
                "Baseline ready" if baseline is not None else "Baseline pending",
            )
        )
    )


def require_quality_gate() -> bool:
    """Stop downstream pages when the current dataset lacks an approved quality state."""
    status = st.session_state.get("quality_status")
    if status == "BLOCKED":
        st.error("This dataset is blocked by critical quality findings. Resolve them on Import Data before continuing.")
        return False
    if status == "PASS_WITH_WARNINGS":
        st.warning("This dataset passed with warnings. Review the Data Quality Report before using results operationally.")
        return True
    if status != "PASS":
        st.info("Run the Data Quality Report on Import Data before continuing.")
        return False
    return True
