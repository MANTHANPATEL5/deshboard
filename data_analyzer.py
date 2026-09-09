"""
Statistical analysis, correlations, outliers, and trends.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils import (
    MAX_CATEGORY_VALUES_IN_CHART,
    MAX_CORR_COLUMNS,
    correlation_direction,
    correlation_strength,
    looks_like_metric,
    safe_json,
    safe_number,
)


def generate_analysis_package(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Full analysis package consumed by AI planner and UI."""
    numerical_analysis = analyze_numerical(df, column_types)
    categorical_analysis = analyze_categorical(df, column_types)
    date_analysis = analyze_dates(df, column_types)
    correlations = analyze_correlations(df, column_types)
    strong_correlations = [
        c for c in correlations if abs(c.get("correlation", 0)) >= 0.5
    ]
    outliers = analyze_outliers(df, column_types)
    trends = analyze_trends(df, column_types)
    important = detect_important_columns(
        df, column_types, numerical_analysis, categorical_analysis
    )
    metrics = detect_business_metrics(df, column_types, numerical_analysis)
    quality = analyze_quality(df)

    package = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "memory_usage_mb": round(
                float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 3
            ),
        },
        "numerical_analysis": numerical_analysis,
        "categorical_analysis": categorical_analysis,
        "date_analysis": date_analysis,
        "correlations": correlations,
        "strong_correlations": strong_correlations,
        "outliers": outliers,
        "trends": trends,
        "important_columns": important,
        "business_metrics": metrics,
        "quality": quality,
    }
    return safe_json(package)


def analyze_numerical(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for col in column_types.get("numerical", []):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        skew = safe_number(series.skew())
        result[col] = {
            "count": int(series.count()),
            "mean": safe_number(series.mean()),
            "median": safe_number(series.median()),
            "min": safe_number(series.min()),
            "max": safe_number(series.max()),
            "std": safe_number(series.std()),
            "variance": safe_number(series.var()),
            "skewness": skew,
            "quantiles": {
                "q05": safe_number(series.quantile(0.05)),
                "q25": safe_number(series.quantile(0.25)),
                "q50": safe_number(series.quantile(0.50)),
                "q75": safe_number(series.quantile(0.75)),
                "q95": safe_number(series.quantile(0.95)),
            },
        }
    return result


def analyze_categorical(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for col in column_types.get("categorical", []) + column_types.get("boolean", []):
        series = df[col].dropna()
        if series.empty:
            continue
        vc = series.astype(str).value_counts()
        total = int(vc.sum()) or 1
        top = vc.head(MAX_CATEGORY_VALUES_IN_CHART)
        dominant = str(vc.index[0])
        result[col] = {
            "n_categories": int(series.nunique()),
            "dominant_category": dominant,
            "dominant_count": int(vc.iloc[0]),
            "dominant_percentage": round(100.0 * vc.iloc[0] / total, 2),
            "distribution": [
                {
                    "value": str(idx),
                    "count": int(cnt),
                    "percentage": round(100.0 * cnt / total, 2),
                }
                for idx, cnt in top.items()
            ],
        }
    return result


def analyze_dates(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for col in column_types.get("date", []):
        dates = pd.to_datetime(df[col], errors="coerce").dropna()
        if dates.empty:
            continue
        span_days = int((dates.max() - dates.min()).days)
        monthly = dates.dt.to_period("M").value_counts().sort_index()
        result[col] = {
            "earliest": dates.min().isoformat(),
            "latest": dates.max().isoformat(),
            "time_span_days": span_days,
            "monthly_counts": {
                str(k): int(v) for k, v in monthly.tail(24).items()
            },
            "yearly_counts": {
                str(k): int(v)
                for k, v in dates.dt.to_period("Y").value_counts().sort_index().items()
            },
        }
    return result


def analyze_correlations(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    nums = column_types.get("numerical", [])[:MAX_CORR_COLUMNS]
    if len(nums) < 2:
        return []

    numeric_df = df[nums].apply(pd.to_numeric, errors="coerce")
    # Drop columns that are duplicates / near-identical
    keep = []
    for col in numeric_df.columns:
        if numeric_df[col].nunique(dropna=True) <= 1:
            continue
        keep.append(col)
    numeric_df = numeric_df[keep]
    if numeric_df.shape[1] < 2:
        return []

    corr = numeric_df.corr(method="pearson")
    pairs: List[Dict[str, Any]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            value = corr.loc[c1, c2]
            if pd.isna(value):
                continue
            # Skip near-perfect duplicates
            if abs(value) >= 0.999 and numeric_df[c1].equals(numeric_df[c2]):
                continue
            val = float(value)
            pairs.append(
                {
                    "column_1": c1,
                    "column_2": c2,
                    "correlation": round(val, 4),
                    "absolute_correlation": round(abs(val), 4),
                    "strength": correlation_strength(val),
                    "direction": correlation_direction(val),
                }
            )
    pairs.sort(key=lambda x: x["absolute_correlation"], reverse=True)
    return pairs


def analyze_outliers(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    from data_engine import detect_outliers_iqr

    result: Dict[str, Any] = {}
    for col in column_types.get("numerical", []):
        info = detect_outliers_iqr(df[col])
        if info["count"] > 0:
            result[col] = info
    return result


def analyze_trends(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Basic time trends when date + numerical metrics exist."""
    dates = column_types.get("date", [])
    nums = column_types.get("numerical", [])
    if not dates or not nums:
        return {}

    trends: Dict[str, Any] = {}
    date_col = dates[0]
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    if temp.empty:
        return {}

    temp["_period"] = temp[date_col].dt.to_period("M").astype(str)

    for metric in nums[:8]:
        series = pd.to_numeric(temp[metric], errors="coerce")
        grouped = (
            temp.assign(**{metric: series})
            .groupby("_period", as_index=False)[metric]
            .sum()
            .dropna()
            .sort_values("_period")
        )
        if len(grouped) < 2:
            continue
        values = grouped[metric].tolist()
        first, last = values[0], values[-1]
        change = None
        if first not in (0, None) and safe_number(first):
            change = round(100.0 * (last - first) / abs(first), 2)
        best_idx = int(np.argmax(values))
        worst_idx = int(np.argmin(values))
        trends[metric] = {
            "date_column": date_col,
            "granularity": "month",
            "points": [
                {"period": str(r["_period"]), "value": safe_number(r[metric])}
                for _, r in grouped.tail(36).iterrows()
            ],
            "best_period": str(grouped.iloc[best_idx]["_period"]),
            "best_value": safe_number(grouped.iloc[best_idx][metric]),
            "worst_period": str(grouped.iloc[worst_idx]["_period"]),
            "worst_value": safe_number(grouped.iloc[worst_idx][metric]),
            "pct_change_first_to_last": change,
        }
    return trends


def detect_important_columns(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    numerical_analysis: Dict[str, Any],
    categorical_analysis: Dict[str, Any],
) -> Dict[str, List[str]]:
    important_num = sorted(
        numerical_analysis.keys(),
        key=lambda c: abs(safe_number(numerical_analysis[c].get("mean"), 0) or 0)
        + abs(safe_number(numerical_analysis[c].get("std"), 0) or 0),
        reverse=True,
    )
    # Prefer metric-like names
    important_num = sorted(
        important_num,
        key=lambda c: (0 if looks_like_metric(c) else 1, important_num.index(c)),
    )

    important_cat = sorted(
        categorical_analysis.keys(),
        key=lambda c: categorical_analysis[c].get("n_categories", 999),
    )
    # Prefer moderate cardinality
    important_cat = [
        c
        for c in important_cat
        if 1 < categorical_analysis[c].get("n_categories", 0) <= MAX_CATEGORY_VALUES_IN_CHART * 2
    ]

    return {
        "numerical": important_num[:10],
        "categorical": important_cat[:10],
        "date": column_types.get("date", [])[:5],
        "identifier": column_types.get("identifier", []),
    }


def detect_business_metrics(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    numerical_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metrics = []
    for col in column_types.get("numerical", []):
        stats = numerical_analysis.get(col, {})
        preferred_agg = "sum" if looks_like_metric(col) else "mean"
        # counts/rates often better as mean
        lowered = col.lower()
        if any(k in lowered for k in ("rate", "ratio", "percent", "score", "age", "avg", "average")):
            preferred_agg = "mean"
        metrics.append(
            {
                "column": col,
                "preferred_aggregation": preferred_agg,
                "mean": stats.get("mean"),
                "sum": safe_number(pd.to_numeric(df[col], errors="coerce").sum()),
                "is_metric_like": looks_like_metric(col),
            }
        )
    metrics.sort(key=lambda m: (not m["is_metric_like"], -(abs(m.get("sum") or 0))))
    return metrics


def analyze_quality(df: pd.DataFrame) -> Dict[str, Any]:
    cells = max(len(df) * max(df.shape[1], 1), 1)
    missing = int(df.isna().sum().sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": missing,
        "missing_percentage": round(100.0 * missing / cells, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [
            c for c in df.columns if df[c].nunique(dropna=True) <= 1
        ],
    }
