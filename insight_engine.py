"""
Insight engine — calculates real business insights from data.
Never invents numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from utils import format_number, format_percent, safe_json, safe_number


def generate_insights(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    analysis_package: Optional[Dict[str, Any]] = None,
    ml_package: Optional[Dict[str, Any]] = None,
    max_insights: int = 40,
) -> List[Dict[str, str]]:
    """
    Return typed insights: [{type, message}, ...] based on actual calculations.
    """
    analysis_package = analysis_package or {}
    insights: List[Dict[str, str]] = []

    insights.extend(_dataset_insights(df, column_types, analysis_package))
    insights.extend(_categorical_insights(df, column_types, analysis_package))
    insights.extend(_numerical_insights(df, column_types, analysis_package))
    insights.extend(_correlation_insights(analysis_package))
    insights.extend(_trend_insights(analysis_package))
    insights.extend(_outlier_insights(analysis_package))
    insights.extend(_ml_insights(ml_package))

    # Deduplicate messages
    seen = set()
    unique: List[Dict[str, str]] = []
    for item in insights:
        msg = item.get("message", "").strip()
        if not msg or msg in seen:
            continue
        seen.add(msg)
        unique.append({"type": item.get("type", "general"), "message": msg})
        if len(unique) >= max_insights:
            break
    return unique


def generate_insight_package(
    df: pd.DataFrame,
    column_types: Optional[Dict[str, List[str]]] = None,
    analysis_package: Optional[Dict[str, Any]] = None,
    ml_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if column_types is None:
        from data_engine import detect_column_types

        column_types = detect_column_types(df)
    insights = generate_insights(df, column_types, analysis_package, ml_package)
    return safe_json(
        {
            "insights": insights,
            "insight_count": len(insights),
            "messages": [i["message"] for i in insights],
        }
    )


def generate_ai_interpretation_context(
    insights: List[Dict[str, str]],
) -> List[str]:
    """Plain strings for AI interpretation prompts."""
    return [i["message"] for i in insights]


def _dataset_insights(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    analysis_package: Dict[str, Any],
) -> List[Dict[str, str]]:
    out = []
    quality = analysis_package.get("quality", {})
    out.append(
        {
            "type": "dataset",
            "message": (
                f"Dataset contains {len(df):,} rows and {df.shape[1]} columns "
                f"({len(column_types.get('numerical', []))} numerical, "
                f"{len(column_types.get('categorical', []))} categorical, "
                f"{len(column_types.get('date', []))} date)."
            ),
        }
    )
    miss = quality.get("missing_percentage")
    if miss is not None:
        out.append(
            {
                "type": "quality",
                "message": f"Overall missing-value rate is {format_percent(miss)} of all cells.",
            }
        )
    return out


def _categorical_insights(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    analysis_package: Dict[str, Any],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    cat_analysis = analysis_package.get("categorical_analysis", {})
    nums = column_types.get("numerical", [])
    metrics = analysis_package.get("business_metrics", [])
    primary_metric = None
    if metrics:
        primary_metric = metrics[0].get("column")
    elif nums:
        primary_metric = nums[0]

    for col, info in cat_analysis.items():
        dominant = info.get("dominant_category")
        pct = info.get("dominant_percentage")
        n_cat = info.get("n_categories")
        if dominant is not None and pct is not None:
            out.append(
                {
                    "type": "concentration",
                    "message": (
                        f"In '{col}', '{dominant}' is the dominant category "
                        f"({format_percent(pct)} of records, {n_cat} categories total)."
                    ),
                }
            )
            if pct and pct >= 60:
                out.append(
                    {
                        "type": "concentration",
                        "message": (
                            f"Values in '{col}' are highly concentrated: "
                            f"'{dominant}' alone accounts for {format_percent(pct)}."
                        ),
                    }
                )

        # Top / bottom by metric
        if primary_metric and primary_metric in df.columns and col in df.columns:
            temp = df[[col, primary_metric]].copy()
            temp[primary_metric] = pd.to_numeric(temp[primary_metric], errors="coerce")
            temp = temp.dropna()
            if temp.empty:
                continue
            grouped = temp.groupby(col, as_index=False)[primary_metric].sum()
            if grouped.empty:
                continue
            grouped = grouped.sort_values(primary_metric, ascending=False)
            top = grouped.iloc[0]
            bottom = grouped.iloc[-1]
            total = grouped[primary_metric].sum()
            contrib = 100.0 * top[primary_metric] / total if total else 0
            out.append(
                {
                    "type": "top_category",
                    "message": (
                        f"Highest '{primary_metric}' by '{col}' is '{top[col]}' "
                        f"at {format_number(top[primary_metric])} "
                        f"({format_percent(contrib)} of total)."
                    ),
                }
            )
            out.append(
                {
                    "type": "bottom_category",
                    "message": (
                        f"Lowest '{primary_metric}' by '{col}' is '{bottom[col]}' "
                        f"at {format_number(bottom[primary_metric])}."
                    ),
                }
            )
            # Top 3 contribution
            if len(grouped) >= 3 and total:
                top3 = grouped.head(3)[primary_metric].sum()
                out.append(
                    {
                        "type": "concentration",
                        "message": (
                            f"Top 3 '{col}' values contribute "
                            f"{format_percent(100.0 * top3 / total)} of total '{primary_metric}'."
                        ),
                    }
                )
    return out


def _numerical_insights(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    analysis_package: Dict[str, Any],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    num_analysis = analysis_package.get("numerical_analysis", {})
    for col, stats in num_analysis.items():
        out.append(
            {
                "type": "metric",
                "message": (
                    f"'{col}' averages {format_number(stats.get('mean'))} "
                    f"(median {format_number(stats.get('median'))}, "
                    f"range {format_number(stats.get('min'))}–{format_number(stats.get('max'))})."
                ),
            }
        )
        skew = safe_number(stats.get("skewness"))
        if skew is not None and abs(skew) >= 1.0:
            direction = "right" if skew > 0 else "left"
            out.append(
                {
                    "type": "distribution",
                    "message": (
                        f"'{col}' is {direction}-skewed (skewness={skew:.2f}), "
                        f"so extremes may influence averages."
                    ),
                }
            )
        # Total if sum-like
        total = safe_number(pd.to_numeric(df[col], errors="coerce").sum())
        if total is not None and abs(total) > abs(safe_number(stats.get("mean"), 0) or 0):
            out.append(
                {
                    "type": "metric",
                    "message": f"Total '{col}' across the dataset is {format_number(total)}.",
                }
            )
    return out


def _correlation_insights(analysis_package: Dict[str, Any]) -> List[Dict[str, str]]:
    out = []
    strong = analysis_package.get("strong_correlations") or analysis_package.get(
        "correlations", []
    )
    if not strong:
        return out
    # strongest overall
    ranked = sorted(
        strong, key=lambda x: abs(x.get("correlation", 0) or 0), reverse=True
    )
    top = ranked[0]
    corr = top.get("correlation", 0)
    out.append(
        {
            "type": "correlation",
            "message": (
                f"Strongest relationship is between '{top.get('column_1')}' and "
                f"'{top.get('column_2')}' (Pearson r={corr}, "
                f"{top.get('strength', '')} {top.get('direction', '')})."
            ),
        }
    )
    for item in ranked[1:6]:
        if abs(item.get("correlation", 0) or 0) < 0.5:
            break
        out.append(
            {
                "type": "correlation",
                "message": (
                    f"'{item.get('column_1')}' and '{item.get('column_2')}' show a "
                    f"{item.get('strength')} {item.get('direction')} correlation "
                    f"(r={item.get('correlation')})."
                ),
            }
        )
    return out


def _trend_insights(analysis_package: Dict[str, Any]) -> List[Dict[str, str]]:
    out = []
    trends = analysis_package.get("trends", {})
    for metric, info in trends.items():
        best_p = info.get("best_period")
        worst_p = info.get("worst_period")
        if best_p:
            out.append(
                {
                    "type": "trend",
                    "message": (
                        f"Best month for '{metric}' is {best_p} "
                        f"({format_number(info.get('best_value'))})."
                    ),
                }
            )
        if worst_p:
            out.append(
                {
                    "type": "trend",
                    "message": (
                        f"Worst month for '{metric}' is {worst_p} "
                        f"({format_number(info.get('worst_value'))})."
                    ),
                }
            )
        change = info.get("pct_change_first_to_last")
        if change is not None:
            direction = "increased" if change >= 0 else "declined"
            out.append(
                {
                    "type": "trend",
                    "message": (
                        f"'{metric}' {direction} by {format_percent(abs(change))} "
                        f"from the first to the latest period."
                    ),
                }
            )
    return out


def _outlier_insights(analysis_package: Dict[str, Any]) -> List[Dict[str, str]]:
    out = []
    outliers = analysis_package.get("outliers", {})
    for col, info in outliers.items():
        count = info.get("count", 0)
        if count:
            out.append(
                {
                    "type": "outlier",
                    "message": (
                        f"Detected {count} unusual values in '{col}' "
                        f"({format_percent(info.get('percentage'))} via {info.get('method', 'IQR')}). "
                        f"Outliers were flagged, not removed."
                    ),
                }
            )
    return out


def _ml_insights(ml_package: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not ml_package or not ml_package.get("ml_used"):
        return []
    out = []
    model_type = ml_package.get("model_type", "model")
    target = ml_package.get("target")
    metrics = ml_package.get("metrics", {})
    out.append(
        {
            "type": "ml",
            "message": (
                f"Machine learning ({model_type}) was applied"
                + (f" with target '{target}'." if target else ".")
            ),
        }
    )
    if "accuracy" in metrics:
        out.append(
            {
                "type": "ml",
                "message": f"Classification accuracy: {format_percent(100 * metrics['accuracy'])}.",
            }
        )
    if "r2" in metrics:
        out.append(
            {
                "type": "ml",
                "message": f"Regression R²: {safe_number(metrics['r2']):.3f}.",
            }
        )
    if "silhouette" in metrics:
        out.append(
            {
                "type": "ml",
                "message": f"Clustering silhouette score: {safe_number(metrics['silhouette']):.3f}.",
            }
        )
    important = ml_package.get("important_features") or []
    if important:
        top_feats = ", ".join(
            f"{f['feature']} ({format_percent(100 * f['importance'])})"
            for f in important[:5]
            if isinstance(f, dict)
        )
        if top_feats:
            out.append(
                {
                    "type": "ml",
                    "message": f"Top predictive features: {top_feats}.",
                }
            )
    return out
