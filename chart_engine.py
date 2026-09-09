"""
Chart engine — Plotly figure creation and rule-based chart recommendations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    MAX_CATEGORY_VALUES_IN_CHART,
    MAX_CORR_COLUMNS,
    MAX_PIE_CATEGORIES,
    MAX_SCATTER_POINTS,
    MAX_TABLE_ROWS,
    looks_like_geo,
    looks_like_metric,
)


THEME = dict(
    template="plotly_white",
    color_discrete_sequence=px.colors.qualitative.Set2,
)


def create_chart(df: pd.DataFrame, config: Dict[str, Any]):
    """
    Create a Plotly figure (or DataFrame for tables) from config.
    Never raises to the caller — returns None on failure.
    """
    try:
        if df is None or df.empty:
            return None
        chart_type = str(config.get("type") or "").lower()
        handlers = {
            "bar": _bar,
            "column": _column,
            "line": _line,
            "area": _area,
            "pie": _pie,
            "donut": _donut,
            "scatter": _scatter,
            "histogram": _histogram,
            "box": _box,
            "heatmap": _heatmap,
            "table": _table,
            "map": _map,
        }
        handler = handlers.get(chart_type)
        if handler is None:
            return None
        fig = handler(df, config)
        if fig is None:
            return None
        if isinstance(fig, pd.DataFrame):
            return fig
        fig.update_layout(
            margin=dict(l=40, r=20, t=50, b=40),
            title=config.get("title") or "",
            **{k: v for k, v in THEME.items() if k != "color_discrete_sequence"},
        )
        return fig
    except Exception:
        return None


def _aggregate(
    df: pd.DataFrame,
    x: str,
    y: Optional[str],
    aggregation: str = "sum",
    top_n: int = MAX_CATEGORY_VALUES_IN_CHART,
) -> Optional[pd.DataFrame]:
    if x not in df.columns:
        return None
    aggregation = (aggregation or "sum").lower()

    if y is None or y not in df.columns or config_count_mode(y, x, aggregation):
        result = (
            df.groupby(x, dropna=False)
            .size()
            .reset_index(name="count")
            .rename(columns={"count": "value"})
        )
        value_col = "value"
    else:
        temp = df[[x, y]].copy()
        temp[y] = pd.to_numeric(temp[y], errors="coerce")
        temp = temp.dropna(subset=[y])
        if temp.empty:
            return None
        agg_map = {
            "sum": "sum",
            "mean": "mean",
            "median": "median",
            "min": "min",
            "max": "max",
            "count": "count",
            "nunique": "nunique",
        }
        how = agg_map.get(aggregation, "sum")
        result = temp.groupby(x, dropna=False)[y].agg(how).reset_index()
        result = result.rename(columns={y: "value"})
        value_col = "value"

    if result.empty:
        return None

    # Sort and cap categories for readability (performance safeguard)
    result = result.sort_values("value", ascending=False)
    if len(result) > top_n:
        top = result.head(top_n)
        other_sum = result.iloc[top_n:]["value"].sum()
        if other_sum and str(result[x].dtype) == "object":
            top = pd.concat(
                [top, pd.DataFrame({x: ["Other"], "value": [other_sum]})],
                ignore_index=True,
            )
        result = top
    return result


def config_count_mode(y, x, aggregation) -> bool:
    return aggregation == "count" and (y is None or y == x)


def _bar(df, config):
    data = _aggregate(df, config.get("x"), config.get("y"), config.get("aggregation", "sum"))
    if data is None:
        return None
    x = config.get("x")
    fig = px.bar(
        data,
        x="value",
        y=x,
        orientation="h",
        title=config.get("title"),
        color_discrete_sequence=THEME["color_discrete_sequence"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def _column(df, config):
    data = _aggregate(df, config.get("x"), config.get("y"), config.get("aggregation", "sum"))
    if data is None:
        return None
    x = config.get("x")
    fig = px.bar(
        data,
        x=x,
        y="value",
        title=config.get("title"),
        color_discrete_sequence=THEME["color_discrete_sequence"],
    )
    return fig


def _line(df, config):
    x, y = config.get("x"), config.get("y")
    if x not in df.columns:
        return None
    temp = df.copy()
    # Prefer datetime sort
    if pd.api.types.is_datetime64_any_dtype(temp[x]) or "date" in str(x).lower():
        temp[x] = pd.to_datetime(temp[x], errors="coerce")
        temp = temp.dropna(subset=[x])
    data = _aggregate(temp, x, y, config.get("aggregation", "sum"), top_n=120)
    if data is None:
        return None
    data = data.sort_values(x)
    fig = px.line(data, x=x, y="value", markers=True, title=config.get("title"))
    return fig


def _area(df, config):
    fig = _line(df, config)
    if fig is None:
        return None
    fig.update_traces(fill="tozeroy")
    return fig


def _pie(df, config, hole: float = 0):
    x, y = config.get("x"), config.get("y")
    data = _aggregate(
        df, x, y, config.get("aggregation", "sum"), top_n=MAX_PIE_CATEGORIES
    )
    if data is None or data.empty:
        return None
    fig = px.pie(
        data,
        names=x,
        values="value",
        title=config.get("title"),
        hole=hole,
        color_discrete_sequence=THEME["color_discrete_sequence"],
    )
    return fig


def _donut(df, config):
    return _pie(df, config, hole=0.45)


def _scatter(df, config):
    x, y = config.get("x"), config.get("y")
    if x not in df.columns or y not in df.columns:
        return None
    temp = df[[x, y] + ([config["color"]] if config.get("color") in df.columns else [])].copy()
    temp[x] = pd.to_numeric(temp[x], errors="coerce")
    temp[y] = pd.to_numeric(temp[y], errors="coerce")
    temp = temp.dropna(subset=[x, y])
    if temp.empty:
        return None
    if len(temp) > MAX_SCATTER_POINTS:
        temp = temp.sample(MAX_SCATTER_POINTS, random_state=42)
    color = config.get("color") if config.get("color") in temp.columns else None
    fig = px.scatter(
        temp,
        x=x,
        y=y,
        color=color,
        title=config.get("title"),
        opacity=0.65,
        color_discrete_sequence=THEME["color_discrete_sequence"],
    )
    return fig


def _histogram(df, config):
    col = config.get("x") or config.get("y")
    if col not in df.columns:
        return None
    temp = df[[col]].copy()
    temp[col] = pd.to_numeric(temp[col], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        return None
    fig = px.histogram(
        temp,
        x=col,
        title=config.get("title"),
        nbins=30,
        color_discrete_sequence=THEME["color_discrete_sequence"],
    )
    return fig


def _box(df, config):
    y = config.get("y") or config.get("x")
    x = config.get("x") if config.get("x") != y else None
    if y not in df.columns:
        return None
    cols = [y] + ([x] if x in df.columns else [])
    temp = df[cols].copy()
    temp[y] = pd.to_numeric(temp[y], errors="coerce")
    temp = temp.dropna(subset=[y])
    if temp.empty:
        return None
    if x and x in temp.columns and temp[x].nunique() > MAX_CATEGORY_VALUES_IN_CHART:
        top = temp[x].value_counts().head(MAX_CATEGORY_VALUES_IN_CHART).index
        temp = temp[temp[x].isin(top)]
    fig = px.box(
        temp,
        x=x if x in temp.columns else None,
        y=y,
        title=config.get("title"),
        color_discrete_sequence=THEME["color_discrete_sequence"],
    )
    return fig


def _heatmap(df, config):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    # Prefer provided numerical list via config
    if config.get("columns"):
        nums = [c for c in config["columns"] if c in df.columns][:MAX_CORR_COLUMNS]
    else:
        nums = nums[:MAX_CORR_COLUMNS]
    if len(nums) < 2:
        return None
    corr = df[nums].apply(pd.to_numeric, errors="coerce").corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=config.get("title") or "Correlation Heatmap",
    )
    return fig


def _table(df, config):
    cols = config.get("columns") or [c for c in [config.get("x"), config.get("y")] if c]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = list(df.columns)[:8]
    return df[cols].head(MAX_TABLE_ROWS)


def _map(df, config):
    geo = config.get("x")
    metric = config.get("y")
    if geo not in df.columns:
        return None
    if metric and metric in df.columns:
        data = _aggregate(df, geo, metric, config.get("aggregation", "sum"), top_n=50)
        values = "value"
    else:
        data = df[geo].value_counts().reset_index()
        data.columns = [geo, "value"]
        values = "value"
    if data is None or data.empty:
        return None
    # Choropleth only works well with country names; fall back to bar
    fig = px.bar(
        data.sort_values("value", ascending=False).head(MAX_CATEGORY_VALUES_IN_CHART),
        x=geo,
        y=values,
        title=config.get("title") or f"Geographic distribution — {geo}",
        color_discrete_sequence=THEME["color_discrete_sequence"],
    )
    return fig


# ==========================================================
# RULE-BASED RECOMMENDATIONS (fallback / augmentation)
# ==========================================================

def recommend_charts(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    density: str = "balanced",
) -> List[Dict[str, Any]]:
    """
    Dynamically recommend meaningful charts from dataset structure.
    No fixed chart-count slice. Density only affects aggressiveness of combinations.
    """
    charts: List[Dict[str, Any]] = []
    nums = [c for c in column_types.get("numerical", []) if c in df.columns]
    cats = [
        c
        for c in column_types.get("categorical", []) + column_types.get("boolean", [])
        if c in df.columns and 1 < df[c].nunique(dropna=True) <= MAX_CATEGORY_VALUES_IN_CHART * 2
    ]
    dates = [c for c in column_types.get("date", []) if c in df.columns]
    metrics = sorted(nums, key=lambda c: (0 if looks_like_metric(c) else 1))

    max_metrics = {"balanced": 4, "detailed": 8, "maximum": 12}.get(density, 4)
    max_cats = {"balanced": 4, "detailed": 8, "maximum": 12}.get(density, 4)

    # Time series
    for d in dates[:2]:
        for m in metrics[:max_metrics]:
            charts.append(
                {
                    "type": "line",
                    "title": f"{m} over time ({d})",
                    "x": d,
                    "y": m,
                    "aggregation": "sum",
                    "priority": 95,
                    "size": "large",
                    "reason": "Date + metric trend",
                }
            )
            if density != "balanced":
                charts.append(
                    {
                        "type": "area",
                        "title": f"{m} area trend ({d})",
                        "x": d,
                        "y": m,
                        "aggregation": "sum",
                        "priority": 75,
                        "size": "medium",
                        "reason": "Time series area",
                    }
                )

    # Category + metric
    for cat in cats[:max_cats]:
        for m in metrics[:max_metrics]:
            charts.append(
                {
                    "type": "bar",
                    "title": f"{m} by {cat}",
                    "x": cat,
                    "y": m,
                    "aggregation": "sum",
                    "priority": 88,
                    "size": "medium",
                    "reason": "Category comparison",
                }
            )
        nunique = df[cat].nunique(dropna=True)
        if nunique <= MAX_PIE_CATEGORIES:
            m0 = metrics[0] if metrics else None
            charts.append(
                {
                    "type": "donut",
                    "title": f"Composition of {cat}",
                    "x": cat,
                    "y": m0,
                    "aggregation": "sum" if m0 else "count",
                    "priority": 65,
                    "size": "medium",
                    "reason": "Low-cardinality composition",
                }
            )

    # Distributions
    for m in metrics[:max_metrics]:
        charts.append(
            {
                "type": "histogram",
                "title": f"Distribution of {m}",
                "x": m,
                "y": None,
                "aggregation": None,
                "priority": 60,
                "size": "medium",
                "reason": "Distribution",
            }
        )
        charts.append(
            {
                "type": "box",
                "title": f"Box plot of {m}",
                "x": cats[0] if cats else None,
                "y": m,
                "aggregation": None,
                "priority": 55,
                "size": "medium",
                "reason": "Outlier / spread",
            }
        )

    # Scatter relationships
    for i in range(min(len(metrics), max_metrics)):
        for j in range(i + 1, min(len(metrics), max_metrics)):
            charts.append(
                {
                    "type": "scatter",
                    "title": f"{metrics[j]} vs {metrics[i]}",
                    "x": metrics[i],
                    "y": metrics[j],
                    "aggregation": None,
                    "priority": 72,
                    "size": "medium",
                    "reason": "Numeric relationship",
                }
            )

    if len(nums) >= 2:
        charts.append(
            {
                "type": "heatmap",
                "title": "Correlation Heatmap",
                "x": None,
                "y": None,
                "aggregation": None,
                "priority": 85,
                "size": "large",
                "reason": "Correlation analysis",
            }
        )

    # Geo
    for cat in cats:
        if looks_like_geo(cat) and metrics:
            charts.append(
                {
                    "type": "map",
                    "title": f"{metrics[0]} by {cat}",
                    "x": cat,
                    "y": metrics[0],
                    "aggregation": "sum",
                    "priority": 80,
                    "size": "large",
                    "reason": "Geographic breakdown",
                }
            )

    # Detail table
    cols = (dates[:1] + cats[:3] + metrics[:4]) or list(df.columns)[:8]
    charts.append(
        {
            "type": "table",
            "title": "Detailed Records",
            "x": None,
            "y": None,
            "columns": [c for c in cols if c in df.columns],
            "aggregation": None,
            "priority": 35,
            "size": "large",
            "reason": "Row-level detail",
        }
    )

    from validation_engine import remove_duplicate_visualizations

    return remove_duplicate_visualizations(charts)


def create_ml_charts(ml_package: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build Plotly figures for ML results. Returns list of {title, fig}."""
    figures = []
    if not ml_package or not ml_package.get("ml_used"):
        return figures

    try:
        if ml_package.get("confusion_matrix"):
            cm = ml_package["confusion_matrix"]
            labels = cm.get("labels", [])
            matrix = cm.get("matrix", [])
            fig = px.imshow(
                matrix,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                title="Confusion Matrix",
            )
            figures.append({"title": "Confusion Matrix", "fig": fig})

        important = ml_package.get("important_features") or []
        if important:
            imp_df = pd.DataFrame(important)
            fig = px.bar(
                imp_df.head(15),
                x="importance",
                y="feature",
                orientation="h",
                title="Feature Importance",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            figures.append({"title": "Feature Importance", "fig": fig})

        preds = ml_package.get("predictions") or {}
        if preds.get("prediction_distribution"):
            dist = preds["prediction_distribution"]
            dist_df = pd.DataFrame(
                {"class": list(dist.keys()), "count": list(dist.values())}
            )
            fig = px.bar(dist_df, x="class", y="count", title="Prediction Distribution")
            figures.append({"title": "Prediction Distribution", "fig": fig})

        if preds.get("actual") and preds.get("predicted"):
            scatter_df = pd.DataFrame(
                {"actual": preds["actual"], "predicted": preds["predicted"]}
            )
            fig = px.scatter(
                scatter_df, x="actual", y="predicted", title="Actual vs Predicted", opacity=0.6
            )
            figures.append({"title": "Actual vs Predicted", "fig": fig})

        if preds.get("residuals"):
            fig = px.histogram(
                pd.DataFrame({"residual": preds["residuals"]}),
                x="residual",
                title="Residual Distribution",
                nbins=30,
            )
            figures.append({"title": "Residuals", "fig": fig})

        if preds.get("cluster_distribution"):
            dist = preds["cluster_distribution"]
            fig = px.bar(
                x=list(dist.keys()),
                y=list(dist.values()),
                title="Cluster Distribution",
                labels={"x": "Cluster", "y": "Count"},
            )
            figures.append({"title": "Cluster Distribution", "fig": fig})

        scatter = preds.get("scatter")
        if scatter and scatter.get("points"):
            sdf = pd.DataFrame(scatter["points"])
            color_col = "cluster" if "cluster" in sdf.columns else (
                "anomaly" if "anomaly" in sdf.columns else None
            )
            fig = px.scatter(
                sdf,
                x=scatter.get("x"),
                y=scatter.get("y"),
                color=color_col,
                title="Cluster / Anomaly Scatter",
                opacity=0.7,
            )
            figures.append({"title": "ML Scatter", "fig": fig})

        if preds.get("history") and preds.get("forecast"):
            hist = pd.DataFrame(preds["history"])
            fut = pd.DataFrame(preds["forecast"])
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist["period"], y=hist["value"], mode="lines+markers", name="History"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=fut["period"], y=fut["value"], mode="lines+markers", name="Forecast"
                )
            )
            if "lower" in fut.columns and "upper" in fut.columns:
                fig.add_trace(
                    go.Scatter(
                        x=list(fut["period"]) + list(fut["period"])[::-1],
                        y=list(fut["upper"]) + list(fut["lower"])[::-1],
                        fill="toself",
                        name="95% interval",
                        line=dict(width=0),
                        opacity=0.2,
                    )
                )
            fig.update_layout(title="Forecast", template="plotly_white")
            figures.append({"title": "Forecast", "fig": fig})

        if preds.get("anomaly_count") is not None and not scatter:
            fig = go.Figure(
                go.Indicator(
                    mode="number",
                    value=preds["anomaly_count"],
                    title={"text": "Anomalies Detected"},
                )
            )
            figures.append({"title": "Anomaly Count", "fig": fig})
    except Exception:
        pass

    return figures
