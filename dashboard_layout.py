"""
Dashboard layout helpers — Power BI-inspired Streamlit UI components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from utils import chart_size_columns, format_percent


CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .main-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0ea5e9 140%);
        color: #f8fafc;
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    }
    .main-hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        font-family: "Segoe UI", "Trebuchet MS", sans-serif;
    }
    .main-hero p {
        margin: 0.45rem 0 0 0;
        opacity: 0.92;
        font-size: 1.02rem;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .kpi-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        min-height: 100px;
    }
    .kpi-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .insight-card {
        background: #f8fafc;
        border-left: 4px solid #0ea5e9;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.55rem;
    }
    .muted { color: #64748b; font-size: 0.92rem; }
    .status-pill {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        border-radius: 999px;
        padding: 0.15rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.35rem;
    }
</style>
"""


def apply_dashboard_style() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_app_header() -> None:
    st.markdown(
        """
        <div class="main-hero">
            <h1>AI-Powered Automatic BI Dashboard Generator</h1>
            <p>Upload → Analyze → Generate — Power BI-inspired dashboards from raw data and natural language.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_header(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h2 style="margin:0;">{title}</h2>
            <p class="muted" style="margin:0.4rem 0 0 0;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def render_kpi_grid(kpis: List[Dict[str, Any]], columns: int = 4) -> None:
    if not kpis:
        st.info("No KPIs available.")
        return
    cols = st.columns(columns)
    for i, kpi in enumerate(kpis):
        with cols[i % columns]:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">{kpi.get('name', 'KPI')}</div>
                    <div class="kpi-value">{kpi.get('display', 'N/A')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_insight_cards(insights: List[Any]) -> None:
    if not insights:
        st.info("No insights generated.")
        return
    for item in insights:
        if isinstance(item, dict):
            msg = item.get("message", "")
            typ = item.get("type", "insight")
        else:
            msg = str(item)
            typ = "insight"
        if not msg:
            continue
        st.markdown(
            f"""
            <div class="insight-card">
                <span class="status-pill">{typ}</span>
                {msg}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dataset_summary_metrics(summary: Dict[str, Any]) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{summary.get('rows', 0):,}")
    c2.metric("Columns", f"{summary.get('columns', 0)}")
    c3.metric("Numerical", summary.get("numerical_variables", 0))
    c4.metric("Categorical", summary.get("categorical_variables", 0))
    c5.metric("Date", summary.get("date_variables", 0))
    score = summary.get("data_quality_score")
    c6.metric("Data Quality", format_percent(score) if score is not None else "N/A")


def render_cleaning_report(report: Dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Before", f"{report.get('original_rows', 0):,}")
    c2.metric("Rows After", f"{report.get('final_rows', 0):,}")
    c3.metric("Duplicates Removed", report.get("duplicates_removed", 0))
    c4.metric(
        "Data Quality",
        format_percent(report.get("data_quality_score"))
        if report.get("data_quality_score") is not None
        else "N/A",
    )
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Missing Before", report.get("missing_values_before", 0))
    c6.metric("Missing After", report.get("missing_values_after", 0))
    c7.metric("Columns Converted", len(report.get("converted_columns") or []))
    c8.metric("Outlier Columns", len(report.get("outlier_columns") or {}))

    with st.expander("Detailed cleaning report", expanded=False):
        st.write("**Actions taken**")
        for action in report.get("actions_taken") or []:
            st.write(f"- {action}")
        if report.get("warnings"):
            st.write("**Warnings**")
            for w in report["warnings"]:
                st.warning(w)
        if report.get("data_quality_explanation"):
            st.caption(report["data_quality_explanation"])
        st.json(
            {
                "converted_columns": report.get("converted_columns"),
                "imputed_columns": report.get("imputed_columns"),
                "constant_columns": report.get("constant_columns"),
                "identifier_columns": report.get("identifier_columns"),
                "outlier_columns": report.get("outlier_columns"),
            }
        )


def render_process_status(stage: int, total: int = 11, label: str = "") -> None:
    st.progress(min(stage / total, 1.0), text=label or f"Step {stage}/{total}")


def chart_columns_for_size(size: str):
    """Return streamlit columns layout based on size."""
    span = chart_size_columns(size)
    if span == 1:
        return [st.container()]
    return st.columns(2)
