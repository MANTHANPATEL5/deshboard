"""
Validation engine for files, AI plans, KPIs, filters, and charts.
Never blindly trust AI-generated JSON.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils import (
    MAX_FILTER_CARDINALITY,
    MAX_PIE_CATEGORIES,
    ensure_list,
    safe_json,
)


ALLOWED_CHART_TYPES = {
    "bar",
    "column",
    "line",
    "area",
    "pie",
    "donut",
    "scatter",
    "histogram",
    "box",
    "heatmap",
    "table",
    "map",
    "kpi",
}

ALLOWED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "count",
    "nunique",
    "min",
    "max",
}

ALLOWED_SIZES = {"small", "medium", "large"}


def parse_ai_json(raw: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Parse AI response into a dict. Returns (plan, errors)."""
    errors: List[str] = []
    if raw is None:
        return None, ["Empty AI response."]
    if isinstance(raw, dict):
        return raw, errors
    text = str(raw).strip()
    if not text:
        return None, ["Empty AI response text."]

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Attempt to extract outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except Exception:
                return None, [f"Invalid JSON from AI: {exc}"]
        else:
            return None, [f"Invalid JSON from AI: {exc}"]

    if not isinstance(data, dict):
        return None, ["AI JSON root must be an object."]
    return data, errors


def validate_and_clean_plan(
    plan: Dict[str, Any],
    df: pd.DataFrame,
    column_types: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Validate AI dashboard plan against the dataset.
    Repair or drop invalid pieces. Dashboard must remain functional.
    """
    column_types = column_types or {}
    columns = set(map(str, df.columns))
    warnings: List[str] = []

    cleaned: Dict[str, Any] = {
        "dashboard_title": str(plan.get("dashboard_title") or "AI Business Intelligence Dashboard"),
        "description": str(plan.get("description") or "Automatically generated analytical dashboard."),
        "filters": [],
        "kpis": [],
        "visualizations": [],
        "tables": [],
        "insights_to_generate": ensure_list(plan.get("insights_to_generate")),
        "ml_analysis": plan.get("ml_analysis") if isinstance(plan.get("ml_analysis"), dict) else {},
        "ml_visualizations": ensure_list(plan.get("ml_visualizations")),
        "validation_warnings": warnings,
    }

    # Filters
    for item in ensure_list(plan.get("filters")):
        if not isinstance(item, dict):
            continue
        col = item.get("column")
        if col not in columns:
            warnings.append(f"Removed filter on missing column '{col}'.")
            continue
        if col in column_types.get("identifier", []):
            warnings.append(f"Skipped identifier filter '{col}'.")
            continue
        nunique = int(df[col].nunique(dropna=True))
        is_date = col in column_types.get("date", []) or pd.api.types.is_datetime64_any_dtype(df[col])
        if not is_date and nunique > MAX_FILTER_CARDINALITY:
            warnings.append(f"Skipped high-cardinality filter '{col}' ({nunique} values).")
            continue
        cleaned["filters"].append(
            {
                "column": col,
                "reason": str(item.get("reason") or "Useful analytical filter"),
                "type": "date" if is_date else "categorical",
            }
        )

    # KPIs
    for item in ensure_list(plan.get("kpis")):
        if not isinstance(item, dict):
            continue
        col = item.get("column")
        agg = str(item.get("aggregation") or "sum").lower()
        if agg not in ALLOWED_AGGREGATIONS:
            warnings.append(f"Invalid KPI aggregation '{agg}' removed.")
            continue
        if agg in {"count"} and (col is None or col not in columns):
            # count rows KPI
            cleaned["kpis"].append(
                {
                    "name": str(item.get("name") or "Row Count"),
                    "column": None,
                    "aggregation": "count",
                    "reason": str(item.get("reason") or "Record count"),
                }
            )
            continue
        if col not in columns:
            warnings.append(f"Removed KPI referencing missing column '{col}'.")
            continue
        if agg not in {"count", "nunique"} and col not in column_types.get("numerical", []):
            # Attempt nunique for categoricals
            if col in column_types.get("categorical", []) + column_types.get("identifier", []):
                agg = "nunique"
            else:
                warnings.append(f"Skipped non-numeric KPI on '{col}'.")
                continue
        cleaned["kpis"].append(
            {
                "name": str(item.get("name") or f"{agg.title()} of {col}"),
                "column": col,
                "aggregation": agg,
                "reason": str(item.get("reason") or ""),
            }
        )

    # Visualizations
    for item in ensure_list(plan.get("visualizations")):
        viz, viz_warnings = validate_visualization(item, df, column_types)
        warnings.extend(viz_warnings)
        if viz:
            cleaned["visualizations"].append(viz)

    cleaned["visualizations"] = remove_duplicate_visualizations(cleaned["visualizations"])

    # Tables
    for item in ensure_list(plan.get("tables")):
        if not isinstance(item, dict):
            continue
        cols = [c for c in ensure_list(item.get("columns")) if c in columns]
        if not cols:
            # default to a useful subset
            cols = list(df.columns)[:8]
        cleaned["tables"].append(
            {
                "title": str(item.get("title") or "Data Table"),
                "columns": cols,
                "priority": _safe_priority(item.get("priority")),
                "reason": str(item.get("reason") or ""),
                "aggregation": item.get("aggregation"),
                "group_by": item.get("group_by") if item.get("group_by") in columns else None,
            }
        )

    # Ensure at least some structure
    if not cleaned["kpis"]:
        cleaned["kpis"] = _fallback_kpis(df, column_types)
        warnings.append("AI returned no valid KPIs; applied fallback KPIs.")

    if not cleaned["visualizations"]:
        cleaned["visualizations"] = _fallback_visualizations(df, column_types)
        warnings.append("AI returned no valid charts; applied analytical fallback charts.")

    cleaned["validation_warnings"] = warnings
    return safe_json(cleaned)


def validate_visualization(
    item: Any,
    df: pd.DataFrame,
    column_types: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not isinstance(item, dict):
        return None, ["Skipped non-object visualization."]

    column_types = column_types or {}
    columns = set(map(str, df.columns))
    chart_type = str(item.get("type") or "").lower().strip()
    if chart_type not in ALLOWED_CHART_TYPES:
        return None, [f"Unsupported chart type '{chart_type}'."]

    if chart_type == "kpi":
        return None, ["KPI entries belong in kpis[], skipped from visualizations."]

    x = item.get("x")
    y = item.get("y")
    color = item.get("color")
    agg = str(item.get("aggregation") or "sum").lower()
    if agg not in ALLOWED_AGGREGATIONS:
        agg = "sum"

    # Heatmap can be correlation without x/y
    if chart_type == "heatmap":
        nums = column_types.get("numerical", [])
        if len(nums) < 2:
            return None, ["Heatmap requires ≥2 numerical columns."]
        return (
            {
                "type": "heatmap",
                "title": str(item.get("title") or "Correlation Heatmap"),
                "x": None,
                "y": None,
                "aggregation": None,
                "priority": _safe_priority(item.get("priority", 80)),
                "size": _safe_size(item.get("size", "large")),
                "reason": str(item.get("reason") or "Correlation analysis"),
            },
            warnings,
        )

    if chart_type == "histogram":
        col = y or x
        if col not in columns:
            return None, [f"Histogram missing numerical column."]
        if col not in column_types.get("numerical", []) and not pd.api.types.is_numeric_dtype(df[col]):
            return None, [f"Histogram column '{col}' is not numerical."]
        return (
            {
                "type": "histogram",
                "title": str(item.get("title") or f"Distribution of {col}"),
                "x": col,
                "y": None,
                "aggregation": None,
                "priority": _safe_priority(item.get("priority", 60)),
                "size": _safe_size(item.get("size", "medium")),
                "reason": str(item.get("reason") or "Distribution"),
            },
            warnings,
        )

    if chart_type == "box":
        col = y or x
        if col not in columns:
            return None, ["Box plot missing numerical column."]
        group = x if x in columns and x != col else item.get("group")
        if group == col:
            group = None
        if group and group not in columns:
            group = None
        return (
            {
                "type": "box",
                "title": str(item.get("title") or f"Box Plot of {col}"),
                "x": group,
                "y": col,
                "aggregation": None,
                "priority": _safe_priority(item.get("priority", 55)),
                "size": _safe_size(item.get("size", "medium")),
                "reason": str(item.get("reason") or "Distribution / outliers"),
            },
            warnings,
        )

    if chart_type == "scatter":
        if x not in columns or y not in columns:
            return None, ["Scatter requires valid x and y columns."]
        return (
            {
                "type": "scatter",
                "title": str(item.get("title") or f"{y} vs {x}"),
                "x": x,
                "y": y,
                "color": color if color in columns else None,
                "aggregation": None,
                "priority": _safe_priority(item.get("priority", 70)),
                "size": _safe_size(item.get("size", "medium")),
                "reason": str(item.get("reason") or "Relationship"),
            },
            warnings,
        )

    if chart_type == "table":
        cols = [c for c in ensure_list(item.get("columns") or [x, y]) if c in columns]
        if not cols:
            cols = list(df.columns)[:6]
        return (
            {
                "type": "table",
                "title": str(item.get("title") or "Table"),
                "x": None,
                "y": None,
                "columns": cols,
                "aggregation": None,
                "priority": _safe_priority(item.get("priority", 40)),
                "size": _safe_size(item.get("size", "large")),
                "reason": str(item.get("reason") or "Detail"),
            },
            warnings,
        )

    if chart_type == "map":
        geo_col = x or item.get("column")
        if geo_col not in columns:
            return None, ["Map requires a geographic column."]
        return (
            {
                "type": "map",
                "title": str(item.get("title") or f"Map of {geo_col}"),
                "x": geo_col,
                "y": y if y in columns else None,
                "aggregation": agg if y in columns else "count",
                "priority": _safe_priority(item.get("priority", 75)),
                "size": _safe_size(item.get("size", "large")),
                "reason": str(item.get("reason") or "Geographic analysis"),
            },
            warnings,
        )

    # Aggregated charts: bar/column/line/area/pie/donut
    if x not in columns:
        return None, [f"Chart missing x column '{x}'."]

    if chart_type in {"pie", "donut"}:
        nunique = int(df[x].nunique(dropna=True))
        if nunique > MAX_PIE_CATEGORIES:
            warnings.append(
                f"Converted pie/donut on '{x}' to bar due to high cardinality ({nunique})."
            )
            chart_type = "bar"

    if y is None or y not in columns:
        # count chart
        agg = "count"
        y = x
        count_mode = True
    else:
        count_mode = False
        if agg not in {"count", "nunique"} and y not in column_types.get("numerical", []):
            if not pd.api.types.is_numeric_dtype(df[y]):
                warnings.append(f"Skipped chart with non-numeric y '{y}'.")
                return None, warnings

    return (
        {
            "type": chart_type,
            "title": str(item.get("title") or f"{y} by {x}"),
            "x": x,
            "y": None if count_mode and chart_type in {"pie", "donut", "bar", "column"} and y == x else y,
            "aggregation": agg,
            "color": color if color in columns else None,
            "priority": _safe_priority(item.get("priority", 70)),
            "size": _safe_size(item.get("size", "medium")),
            "reason": str(item.get("reason") or ""),
            "count_mode": count_mode,
        },
        warnings,
    )


def remove_duplicate_visualizations(visualizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prevent duplicate analytical representations of the same encoding
    unless types are meaningfully different (e.g., scatter vs heatmap).
    """
    seen = set()
    result = []
    for viz in sorted(visualizations, key=lambda v: -_safe_priority(v.get("priority"))):
        vtype = viz.get("type")
        # Treat bar/column as same family for duplicate detection
        family = "bar" if vtype in {"bar", "column"} else vtype
        if family in {"pie", "donut"}:
            family = "pie"
        key = (
            family,
            viz.get("x"),
            viz.get("y"),
            viz.get("aggregation"),
            viz.get("color"),
        )
        if key in seen:
            continue
        # Also block pie+bar for identical x/y/agg
        alt_keys = []
        if family == "pie":
            alt_keys.append(("bar", viz.get("x"), viz.get("y"), viz.get("aggregation"), viz.get("color")))
        if family == "bar":
            alt_keys.append(("pie", viz.get("x"), viz.get("y"), viz.get("aggregation"), viz.get("color")))
        if any(k in seen for k in alt_keys):
            continue
        seen.add(key)
        result.append(viz)
    return result


def _safe_priority(value: Any) -> int:
    try:
        return int(max(0, min(100, float(value))))
    except Exception:
        return 50


def _safe_size(value: Any) -> str:
    v = str(value or "medium").lower()
    return v if v in ALLOWED_SIZES else "medium"


def _fallback_kpis(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    kpis = [
        {
            "name": "Total Records",
            "column": None,
            "aggregation": "count",
            "reason": "Dataset size",
        }
    ]
    for col in column_types.get("numerical", [])[:4]:
        kpis.append(
            {
                "name": f"Total {col}",
                "column": col,
                "aggregation": "sum",
                "reason": "Primary metric",
            }
        )
        kpis.append(
            {
                "name": f"Avg {col}",
                "column": col,
                "aggregation": "mean",
                "reason": "Central tendency",
            }
        )
        break
    for col in column_types.get("categorical", [])[:1]:
        kpis.append(
            {
                "name": f"Unique {col}",
                "column": col,
                "aggregation": "nunique",
                "reason": "Dimension coverage",
            }
        )
    return kpis


def _fallback_visualizations(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Rule-based charts when AI fails — still dynamic, no fixed 4–6 cap."""
    from chart_engine import recommend_charts

    return recommend_charts(df, column_types)
