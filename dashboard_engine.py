"""
Dashboard engine — KPIs, filters, table builders, orchestration helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from chart_engine import create_chart
from utils import MAX_FILTER_CARDINALITY, MAX_TABLE_ROWS, format_number, safe_number


def apply_filters(
    df: pd.DataFrame,
    filter_state: Dict[str, Any],
    filter_configs: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Apply active dashboard filters to a dataframe."""
    if df is None or df.empty or not filter_state:
        return df

    filtered = df.copy()
    for column, value in filter_state.items():
        if column not in filtered.columns or value is None:
            continue
        # Date range
        if isinstance(value, (list, tuple)) and len(value) == 2 and _is_date_like(filtered[column]):
            start, end = value
            dates = pd.to_datetime(filtered[column], errors="coerce")
            mask = pd.Series(True, index=filtered.index)
            if start is not None:
                mask &= dates >= pd.to_datetime(start)
            if end is not None:
                mask &= dates <= pd.to_datetime(end)
            filtered = filtered[mask]
            continue
        # Multiselect categories
        if isinstance(value, (list, tuple, set)):
            if len(value) == 0:
                continue
            filtered = filtered[filtered[column].astype(str).isin([str(v) for v in value])]
            continue
        filtered = filtered[filtered[column].astype(str) == str(value)]
    return filtered


def _is_date_like(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series) or _looks_datetime(series)


def _looks_datetime(series: pd.Series) -> bool:
    sample = series.dropna().head(20)
    if sample.empty:
        return False
    converted = pd.to_datetime(sample, errors="coerce")
    return float(converted.notna().mean()) >= 0.8


def compute_kpi(df: pd.DataFrame, kpi: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a single KPI on (possibly filtered) data."""
    name = kpi.get("name") or "KPI"
    column = kpi.get("column")
    agg = str(kpi.get("aggregation") or "sum").lower()

    if df is None or df.empty:
        return {"name": name, "value": None, "display": "N/A", "aggregation": agg}

    try:
        if agg == "count" and (column is None or column not in df.columns):
            value = float(len(df))
        elif column not in df.columns:
            return {"name": name, "value": None, "display": "N/A", "aggregation": agg}
        elif agg == "count":
            value = float(df[column].count())
        elif agg == "nunique":
            value = float(df[column].nunique(dropna=True))
        else:
            series = pd.to_numeric(df[column], errors="coerce")
            if agg == "sum":
                value = safe_number(series.sum())
            elif agg == "mean":
                value = safe_number(series.mean())
            elif agg == "median":
                value = safe_number(series.median())
            elif agg == "min":
                value = safe_number(series.min())
            elif agg == "max":
                value = safe_number(series.max())
            else:
                value = safe_number(series.sum())
        return {
            "name": name,
            "value": value,
            "display": format_number(value),
            "aggregation": agg,
            "column": column,
            "reason": kpi.get("reason"),
        }
    except Exception:
        return {"name": name, "value": None, "display": "N/A", "aggregation": agg}


def compute_all_kpis(df: pd.DataFrame, kpis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [compute_kpi(df, k) for k in kpis or []]


def build_filter_options(
    df: pd.DataFrame,
    filter_configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepare filter widget metadata."""
    options = []
    for cfg in filter_configs or []:
        col = cfg.get("column")
        if col not in df.columns:
            continue
        ftype = cfg.get("type")
        if ftype == "date" or _is_date_like(df[col]):
            dates = pd.to_datetime(df[col], errors="coerce").dropna()
            if dates.empty:
                continue
            options.append(
                {
                    "column": col,
                    "type": "date",
                    "min": dates.min().date(),
                    "max": dates.max().date(),
                    "reason": cfg.get("reason"),
                }
            )
        else:
            values = (
                df[col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(MAX_FILTER_CARDINALITY)
                .index.tolist()
            )
            options.append(
                {
                    "column": col,
                    "type": "categorical",
                    "values": values,
                    "reason": cfg.get("reason"),
                }
            )
    return options


def build_table(
    df: pd.DataFrame,
    table_cfg: Dict[str, Any],
) -> pd.DataFrame:
    cols = [c for c in table_cfg.get("columns") or [] if c in df.columns]
    if not cols:
        cols = list(df.columns)[:8]

    group_by = table_cfg.get("group_by")
    aggregation = table_cfg.get("aggregation")
    if group_by and group_by in df.columns and aggregation:
        num_cols = [
            c for c in cols if c != group_by and pd.api.types.is_numeric_dtype(df[c])
        ]
        if num_cols:
            how = aggregation if aggregation != "nunique" else "nunique"
            grouped = df.groupby(group_by, as_index=False)[num_cols].agg(how)
            return grouped.head(MAX_TABLE_ROWS)
    return df[cols].head(MAX_TABLE_ROWS)


def render_visualization(df: pd.DataFrame, config: Dict[str, Any]):
    """Proxy to chart engine with safety."""
    return create_chart(df, config)


def organize_visualizations(
    visualizations: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Organize charts into dashboard sections by type/priority."""
    sections = {
        "main": [],
        "secondary": [],
        "relationship": [],
        "distribution": [],
        "detail": [],
    }
    for viz in sorted(visualizations or [], key=lambda v: -int(v.get("priority") or 50)):
        vtype = viz.get("type")
        priority = int(viz.get("priority") or 50)
        if vtype in {"line", "area"} or priority >= 90:
            sections["main"].append(viz)
        elif vtype in {"scatter", "heatmap"}:
            sections["relationship"].append(viz)
        elif vtype in {"histogram", "box"}:
            sections["distribution"].append(viz)
        elif vtype == "table":
            sections["detail"].append(viz)
        else:
            sections["secondary"].append(viz)
    return sections


def prepare_dashboard_bundle(
    df: pd.DataFrame,
    plan: Dict[str, Any],
    filter_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute filtered frame, KPIs, and organized chart configs."""
    filtered = apply_filters(df, filter_state or {}, plan.get("filters"))
    kpis = compute_all_kpis(filtered, plan.get("kpis") or [])
    sections = organize_visualizations(plan.get("visualizations") or [])
    tables = [build_table(filtered, t) for t in plan.get("tables") or []]
    return {
        "filtered_df": filtered,
        "kpis": kpis,
        "sections": sections,
        "tables": tables,
        "row_count": len(filtered),
        "original_row_count": len(df),
    }
