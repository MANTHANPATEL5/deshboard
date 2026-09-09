"""
AI-Powered Automatic Business Intelligence Dashboard Generator

Entry point: streamlit run app.py
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from ai_planner import generate_dashboard_plan
from chart_engine import create_chart, create_ml_charts
from dashboard_engine import (
    apply_filters,
    build_filter_options,
    build_table,
    compute_all_kpis,
    organize_visualizations,
)
from dashboard_layout import (
    apply_dashboard_style,
    render_app_header,
    render_cleaning_report,
    render_dashboard_header,
    render_dataset_summary_metrics,
    render_insight_cards,
    render_kpi_grid,
    render_section_title,
)
from data_analyzer import generate_analysis_package
from data_engine import (
    clean_data,
    generate_profile_dataframe,
    load_file,
    summarize_dataset,
)
from export_engine import (
    export_analysis_summary,
    export_csv,
    export_dashboard_plan,
    export_excel_report,
    export_json,
    export_pdf_report,
    export_visualization_config,
)
from insight_engine import generate_insights
from ml_engine import evaluate_ml_opportunity, run_ml_analysis
from utils import get_openai_api_key, to_json_string


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI BI Dashboard Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dashboard_style()


# ==========================================================
# SESSION STATE
# ==========================================================

SESSION_DEFAULTS = {
    "original_df": None,
    "cleaned_df": None,
    "column_types": None,
    "cleaning_report": None,
    "profile_df": None,
    "dataset_summary": None,
    "analysis_package": None,
    "ml_package": None,
    "automatic_insights": None,
    "dashboard_plan": None,
    "user_request": "",
    "file_meta": None,
    "pipeline_complete": False,
    "generation_error": None,
}


def init_session() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_dashboard_outputs() -> None:
    st.session_state.dashboard_plan = None
    st.session_state.ml_package = None
    st.session_state.automatic_insights = None
    st.session_state.generation_error = None


def reset_all() -> None:
    for key, value in SESSION_DEFAULTS.items():
        st.session_state[key] = value


init_session()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("Controls")
    st.caption("Upload → Analyze → Describe → Generate")

    focus = st.selectbox(
        "Dashboard focus",
        [
            "Let AI decide everything",
            "General overview",
            "Sales",
            "Finance",
            "Marketing",
            "Operations",
            "Customer",
            "HR",
            "Custom",
        ],
        index=0,
    )
    density = st.selectbox(
        "Chart density",
        ["Balanced", "Detailed", "Maximum meaningful analysis"],
        index=0,
    )
    st.divider()
    st.markdown(
        """
        **Pipeline**
        1. Upload data  
        2. Clean & profile  
        3. Statistical analysis  
        4. ML (if useful)  
        5. AI dashboard plan  
        6. Interactive dashboard  
        """
    )
    if st.button("Create New Dashboard", use_container_width=True):
        reset_all()
        st.rerun()


render_app_header()


# ==========================================================
# API KEY
# ==========================================================

def resolve_api_key() -> Optional[str]:
    try:
        return get_openai_api_key(st.secrets)
    except Exception:
        return get_openai_api_key(None)


api_key = resolve_api_key()
if not api_key:
    st.warning(
        "OPENAI_API_KEY is not configured. "
        "Add it to `.streamlit/secrets.toml` or set the OPENAI_API_KEY environment variable. "
        "Without it, dashboard planning will use the analytical fallback planner."
    )


# ==========================================================
# STEP 1 — UPLOAD
# ==========================================================

st.subheader("1. Upload Dataset")
uploaded = st.file_uploader(
    "Upload CSV, XLSX, or XLS",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
)

if uploaded is not None:
    # Reload only when file changes
    file_id = f"{uploaded.name}-{getattr(uploaded, 'size', 0)}"
    if st.session_state.file_meta != file_id:
        reset_all()
        st.session_state.file_meta = file_id
        with st.status("Loading dataset...", expanded=True) as status:
            st.write("Validating file...")
            df, meta = load_file(uploaded)
            if df is None:
                for err in meta.get("errors") or ["Failed to load file."]:
                    st.error(err)
                status.update(label="Upload failed", state="error")
            else:
                st.session_state.original_df = df
                st.write("Cleaning data...")
                cleaned, column_types, cleaning_report = clean_data(df)
                st.session_state.cleaned_df = cleaned
                st.session_state.column_types = column_types
                st.session_state.cleaning_report = cleaning_report

                st.write("Profiling dataset...")
                st.session_state.profile_df = generate_profile_dataframe(
                    cleaned, column_types
                )
                st.session_state.dataset_summary = summarize_dataset(
                    cleaned, column_types, cleaning_report
                )

                st.write("Running statistical analysis...")
                st.session_state.analysis_package = generate_analysis_package(
                    cleaned, column_types
                )

                st.write("Generating automatic insights...")
                st.session_state.automatic_insights = generate_insights(
                    cleaned,
                    column_types,
                    st.session_state.analysis_package,
                )
                st.session_state.pipeline_complete = True
                status.update(label="Data ready for AI dashboard generation", state="complete")

                st.success(
                    f"Loaded **{meta.get('filename')}** "
                    f"({meta.get('size_label')}) — "
                    f"{meta.get('rows'):,} rows × {meta.get('columns')} columns"
                )


# ==========================================================
# DATA SECTIONS
# ==========================================================

if st.session_state.cleaned_df is not None:
    st.subheader("2. Data Cleaning & Quality")
    render_cleaning_report(st.session_state.cleaning_report or {})

    st.subheader("3. Dataset Summary")
    render_dataset_summary_metrics(st.session_state.dataset_summary or {})

    with st.expander("View Original Dataset", expanded=False):
        st.dataframe(st.session_state.original_df.head(50), use_container_width=True)

    with st.expander("View Cleaned Dataset", expanded=False):
        st.dataframe(st.session_state.cleaned_df.head(50), use_container_width=True)

    with st.expander("Column Profile", expanded=False):
        if st.session_state.profile_df is not None:
            st.dataframe(st.session_state.profile_df, use_container_width=True)
        st.json(st.session_state.column_types)

    with st.expander("Statistical Analysis & Correlations", expanded=False):
        analysis = st.session_state.analysis_package or {}
        st.write("**Strong correlations**")
        strong = analysis.get("strong_correlations") or []
        if strong:
            st.dataframe(pd.DataFrame(strong), use_container_width=True)
        else:
            st.caption("No strong correlations detected.")
        st.write("**Outliers (flagged, not removed)**")
        st.json(analysis.get("outliers") or {})
        st.write("**Trends**")
        st.json(analysis.get("trends") or {})

    with st.expander("Automatic Insights (pre-dashboard)", expanded=False):
        render_insight_cards(st.session_state.automatic_insights or [])

    # ==========================================================
    # USER REQUIREMENT
    # ==========================================================

    st.subheader("4. Tell AI What You Want")
    default_prompt = st.session_state.user_request or (
        "Create a complete analytical dashboard with the most important KPIs, "
        "trends, category comparisons, distributions, relationships, tables, "
        "and any other meaningful visualizations for this dataset."
    )
    user_request = st.text_area(
        "Describe your analysis goal in plain language",
        value=default_prompt,
        height=140,
        placeholder=(
            "Example: Create a complete sales dashboard showing revenue, profit, "
            "monthly trends, regional performance, top products, and key relationships."
        ),
    )
    st.session_state.user_request = user_request

    col_a, col_b, col_c = st.columns([1.4, 1, 1])
    generate_clicked = col_a.button(
        "Generate Complete AI Dashboard",
        type="primary",
        use_container_width=True,
    )
    regenerate_clicked = col_b.button(
        "Regenerate Dashboard",
        use_container_width=True,
        disabled=st.session_state.dashboard_plan is None,
    )
    clear_plan = col_c.button("Clear Dashboard", use_container_width=True)
    if clear_plan:
        reset_dashboard_outputs()
        st.rerun()

    if generate_clicked or regenerate_clicked:
        if not user_request.strip():
            st.error("Please describe what you want to analyze.")
        else:
            try:
                with st.status("Building AI dashboard...", expanded=True) as status:
                    cleaned = st.session_state.cleaned_df
                    column_types = st.session_state.column_types
                    analysis = st.session_state.analysis_package

                    st.write("1/11 Loading dataset...")
                    st.write("2/11 Cleaning data... (cached)")
                    st.write("3/11 Profiling dataset... (cached)")
                    st.write("4/11 Running statistical analysis... (cached)")
                    st.write("5/11 Detecting relationships... (cached)")

                    st.write("6/11 Evaluating ML opportunities...")
                    opportunity = evaluate_ml_opportunity(
                        cleaned, column_types, user_request, analysis
                    )
                    st.write(opportunity.get("reason", ""))
                    ml_package = run_ml_analysis(
                        cleaned,
                        column_types,
                        user_request,
                        analysis,
                        opportunity,
                    )
                    st.session_state.ml_package = ml_package

                    st.write("7/11 Asking AI to design dashboard...")
                    focus_value = (
                        "General overview"
                        if focus == "Let AI decide everything"
                        else focus
                    )
                    try:
                        if api_key:
                            plan = generate_dashboard_plan(
                                cleaned,
                                column_types,
                                user_request,
                                api_key,
                                analysis,
                                ml_package=ml_package,
                                cleaning_report=st.session_state.cleaning_report,
                                focus=focus_value,
                                density=density,
                            )
                        else:
                            # Explicit fallback without pretending AI succeeded
                            from ai_planner import _rule_based_plan

                            plan = _rule_based_plan(
                                cleaned, column_types, user_request, density
                            )
                            plan.setdefault("validation_warnings", []).append(
                                "OPENAI_API_KEY is not configured. Used analytical fallback planner."
                            )
                            st.warning("OPENAI_API_KEY is not configured.")
                    except Exception as api_exc:
                        from ai_planner import _rule_based_plan

                        st.warning(
                            f"AI planning failed ({api_exc}). Using analytical fallback planner."
                        )
                        plan = _rule_based_plan(
                            cleaned, column_types, user_request, density
                        )
                        plan.setdefault("validation_warnings", []).append(str(api_exc))

                    st.write("8/11 Validating dashboard plan...")
                    st.session_state.dashboard_plan = plan

                    st.write("9/11 Creating visualizations...")
                    st.write("10/11 Generating insights...")
                    insights = generate_insights(
                        cleaned,
                        column_types,
                        analysis,
                        ml_package,
                    )
                    st.session_state.automatic_insights = insights

                    st.write("11/11 Dashboard ready.")
                    status.update(label="Dashboard ready", state="complete")
            except Exception as exc:
                st.session_state.generation_error = str(exc)
                st.error("Dashboard generation failed. Please check your data and try again.")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())


# ==========================================================
# GENERATED DASHBOARD
# ==========================================================

plan = st.session_state.dashboard_plan
cleaned_df = st.session_state.cleaned_df

if plan and cleaned_df is not None:
    st.divider()
    render_dashboard_header(
        plan.get("dashboard_title", "AI Dashboard"),
        plan.get("description", ""),
    )

    if plan.get("validation_warnings"):
        with st.expander("Plan validation notes", expanded=False):
            for w in plan["validation_warnings"]:
                st.caption(f"• {w}")

    # ---- FILTERS ----
    render_section_title("Filters", "Interactive controls update KPIs, charts, and tables.")
    filter_options = build_filter_options(cleaned_df, plan.get("filters") or [])
    filter_state: Dict[str, Any] = {}

    if filter_options:
        filter_cols = st.columns(min(4, len(filter_options)))
        for i, opt in enumerate(filter_options):
            with filter_cols[i % len(filter_cols)]:
                col = opt["column"]
                if opt["type"] == "date":
                    date_range = st.date_input(
                        f"{col}",
                        value=(opt["min"], opt["max"]),
                        min_value=opt["min"],
                        max_value=opt["max"],
                        key=f"filter_date_{col}",
                    )
                    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                        filter_state[col] = date_range
                else:
                    selected = st.multiselect(
                        f"{col}",
                        options=opt["values"],
                        default=[],
                        key=f"filter_cat_{col}",
                        help=opt.get("reason") or "",
                    )
                    if selected:
                        filter_state[col] = selected
    else:
        st.caption("No useful filters detected for this dataset.")

    filtered_df = apply_filters(cleaned_df, filter_state, plan.get("filters"))
    st.caption(
        f"Showing **{len(filtered_df):,}** of **{len(cleaned_df):,}** rows after filters."
    )

    # ---- KPIs ----
    render_section_title("KPIs")
    kpis = compute_all_kpis(filtered_df, plan.get("kpis") or [])
    render_kpi_grid(kpis, columns=4)

    # ---- CHARTS ----
    sections = organize_visualizations(plan.get("visualizations") or [])
    chart_count = len(plan.get("visualizations") or [])
    st.caption(f"Rendering **{chart_count}** AI-planned visualizations (dynamic count — no fixed limit).")

    def _render_chart_group(title: str, items):
        if not items:
            return
        render_section_title(title)
        # Large charts full width; others in pairs
        buffer = []
        for viz in items:
            size = (viz.get("size") or "medium").lower()
            if size == "large" or viz.get("type") in {"heatmap", "table", "line", "area", "map"}:
                if buffer:
                    cols = st.columns(2)
                    for idx, pending in enumerate(buffer):
                        with cols[idx]:
                            _show_viz(filtered_df, pending)
                    buffer = []
                _show_viz(filtered_df, viz)
            else:
                buffer.append(viz)
                if len(buffer) == 2:
                    cols = st.columns(2)
                    for idx, pending in enumerate(buffer):
                        with cols[idx]:
                            _show_viz(filtered_df, pending)
                    buffer = []
        if buffer:
            cols = st.columns(len(buffer))
            for idx, pending in enumerate(buffer):
                with cols[idx]:
                    _show_viz(filtered_df, pending)

    def _show_viz(df: pd.DataFrame, viz: Dict[str, Any]) -> None:
        try:
            result = create_chart(df, viz)
            if result is None:
                st.warning(f"Could not render: {viz.get('title', viz.get('type'))}")
                return
            if isinstance(result, pd.DataFrame):
                st.markdown(f"**{viz.get('title', 'Table')}**")
                st.dataframe(result, use_container_width=True)
            else:
                st.plotly_chart(result, use_container_width=True)
        except Exception:
            st.warning(f"Chart failed: {viz.get('title', 'untitled')}")

    _render_chart_group("Main Analysis", sections["main"])
    _render_chart_group("Secondary Analysis", sections["secondary"])
    _render_chart_group("Relationship Analysis", sections["relationship"])
    _render_chart_group("Distribution Analysis", sections["distribution"])

    # ---- TABLES ----
    tables_cfg = plan.get("tables") or []
    detail_charts = sections.get("detail") or []
    if tables_cfg or detail_charts:
        render_section_title("Detail Tables")
        for tcfg in tables_cfg:
            st.markdown(f"**{tcfg.get('title', 'Table')}**")
            st.dataframe(build_table(filtered_df, tcfg), use_container_width=True)
        for viz in detail_charts:
            _show_viz(filtered_df, viz)

    # ---- INSIGHTS ----
    render_section_title("AI / Analytical Insights")
    # Refresh insights on filtered data where practical
    filtered_insights = generate_insights(
        filtered_df,
        st.session_state.column_types,
        st.session_state.analysis_package,
        st.session_state.ml_package,
        max_insights=25,
    )
    render_insight_cards(filtered_insights)

    if plan.get("insights_to_generate"):
        with st.expander("Requested insight themes from AI plan", expanded=False):
            for theme in plan["insights_to_generate"]:
                st.write(f"- {theme}")

    # ---- ML SECTION ----
    ml_package = st.session_state.ml_package or {}
    render_section_title("Machine Learning")
    if ml_package.get("ml_used"):
        st.success(ml_package.get("message") or "ML analysis complete.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", ml_package.get("model_type", "—"))
        m2.metric("Target", ml_package.get("target") or "—")
        metrics = ml_package.get("metrics") or {}
        if "accuracy" in metrics:
            m3.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        elif "r2" in metrics:
            m3.metric("R²", f"{metrics['r2']:.3f}")
        elif "silhouette" in metrics:
            m3.metric("Silhouette", f"{metrics['silhouette']:.3f}")
        elif "anomaly_count" in metrics:
            m3.metric("Anomalies", metrics["anomaly_count"])
        else:
            m3.metric("Status", "Trained")

        st.json(metrics)
        if ml_package.get("important_features"):
            st.write("**Feature importance**")
            st.dataframe(
                pd.DataFrame(ml_package["important_features"]),
                use_container_width=True,
            )

        ml_figs = create_ml_charts(ml_package)
        if ml_figs:
            cols = st.columns(2)
            for i, item in enumerate(ml_figs):
                with cols[i % 2]:
                    st.plotly_chart(item["fig"], use_container_width=True)
        if ml_package.get("warnings"):
            for w in ml_package["warnings"]:
                st.caption(f"ML note: {w}")
    else:
        st.info(
            ml_package.get("message")
            or "Machine learning was not applied because no reliable prediction target was detected."
        )

    # ---- PLAN INSPECTION ----
    with st.expander("View AI Dashboard Plan", expanded=False):
        st.json(plan)

    # ---- EXPORTS ----
    render_section_title("Export Results")
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button(
        "Cleaned CSV",
        data=export_csv(cleaned_df),
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    e2.download_button(
        "Filtered CSV",
        data=export_csv(filtered_df),
        file_name="filtered_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
    e3.download_button(
        "Dashboard Plan JSON",
        data=export_dashboard_plan(plan),
        file_name="dashboard_plan.json",
        mime="application/json",
        use_container_width=True,
    )
    e4.download_button(
        "Visualization Config",
        data=export_visualization_config(plan),
        file_name="visualization_config.json",
        mime="application/json",
        use_container_width=True,
    )

    e5, e6, e7, e8 = st.columns(4)
    e5.download_button(
        "Analysis Summary",
        data=export_analysis_summary(
            st.session_state.dataset_summary or {},
            st.session_state.cleaning_report or {},
            st.session_state.analysis_package or {},
            filtered_insights,
            ml_package,
        ),
        file_name="analysis_summary.json",
        mime="application/json",
        use_container_width=True,
    )
    excel_bytes = export_excel_report(
        filtered_df,
        kpis,
        filtered_insights,
        st.session_state.dataset_summary or {},
        plan,
    )
    e6.download_button(
        "Excel Report",
        data=excel_bytes,
        file_name="dashboard_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    pdf_bytes = export_pdf_report(
        plan.get("dashboard_title", "Dashboard"),
        plan.get("description", ""),
        kpis,
        filtered_insights,
        st.session_state.dataset_summary or {},
    )
    if pdf_bytes:
        e7.download_button(
            "PDF Report",
            data=pdf_bytes,
            file_name="dashboard_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        e7.caption("PDF optional (install reportlab)")
    e8.download_button(
        "Column Types JSON",
        data=export_json(st.session_state.column_types),
        file_name="column_types.json",
        mime="application/json",
        use_container_width=True,
    )

elif cleaned_df is None:
    st.info(
        "Upload a CSV or Excel file to begin. The system will automatically clean, "
        "profile, analyze, and — after you describe your goal — generate a full dashboard."
    )
