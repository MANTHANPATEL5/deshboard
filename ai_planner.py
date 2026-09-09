"""
AI Dashboard Planner — OpenAI integration for structured dashboard plans.

AI reasons and plans. Python calculates numbers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from utils import (
    SAMPLE_ROWS_FOR_AI,
    safe_json,
    sample_dataframe,
    to_json_string,
    truncate_text,
)


SYSTEM_PROMPT = """
You are an expert AI Business Intelligence dashboard designer.

Your job is to design a complete Power BI-style analytical dashboard
from structured dataset analysis (NOT raw row dumps).

You receive:
1. User requirement in natural language
2. Dataset schema and column types
3. Statistical analysis
4. Correlations and outliers
5. Trends
6. ML results (when available)
7. A small representative sample

RULES:
- Never invent column names. Use only columns present in the schema.
- Never invent numerical values or fake KPIs numbers.
- Do NOT use a fixed number of charts. Generate ALL meaningful visualizations
  that are useful for this dataset and the user request.
- Avoid duplicate charts (same measure/dimension with bar+column+pie+donut).
- Prefer analytical diversity: trends, comparisons, distributions,
  relationships, composition (only when cardinality is small), tables.
- Pie/donut only when categories ≤ 8.
- Avoid identifier columns as dimensions/filters.
- Priority: 90-100 critical, 70-89 important, 50-69 supporting, 30-49 optional.
- Size: small | medium | large
- Return STRICT JSON only matching the schema below.

Chart types allowed:
bar, column, line, area, pie, donut, scatter, histogram, box, heatmap, table, map

Aggregations allowed:
sum, mean, median, count, nunique, min, max

JSON schema:
{
  "dashboard_title": "...",
  "description": "...",
  "filters": [{"column": "...", "reason": "..."}],
  "kpis": [{"name": "...", "column": "...", "aggregation": "sum", "reason": "..."}],
  "visualizations": [
    {
      "type": "line",
      "title": "...",
      "x": "...",
      "y": "...",
      "aggregation": "sum",
      "priority": 95,
      "size": "large",
      "reason": "..."
    }
  ],
  "tables": [
    {"title": "...", "columns": ["..."], "priority": 50, "reason": "..."}
  ],
  "insights_to_generate": ["..."],
  "ml_analysis": {
    "recommended": true,
    "type": "classification",
    "target": "...",
    "reason": "..."
  }
}
""".strip()


def build_dataset_summary(
    df,
    column_types: Dict[str, List[str]],
    analysis_package: Dict[str, Any],
    ml_package: Optional[Dict[str, Any]] = None,
    cleaning_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact structured summary for the LLM (not the full dataset)."""
    correlations = analysis_package.get("correlations", [])[:20]
    strong = analysis_package.get("strong_correlations", correlations[:10])[:10]

    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(map(str, df.columns)),
        "column_types": column_types,
        "sample_rows": sample_dataframe(df, SAMPLE_ROWS_FOR_AI),
        "quality": analysis_package.get("quality", {}),
        "cleaning_report_highlights": {
            "duplicates_removed": (cleaning_report or {}).get("duplicates_removed"),
            "missing_values_before": (cleaning_report or {}).get("missing_values_before"),
            "missing_values_after": (cleaning_report or {}).get("missing_values_after"),
            "data_quality_score": (cleaning_report or {}).get("data_quality_score"),
            "constant_columns": (cleaning_report or {}).get("constant_columns"),
            "identifier_columns": (cleaning_report or {}).get("identifier_columns"),
        },
        "numerical_analysis": _shrink_dict(analysis_package.get("numerical_analysis", {}), 12),
        "categorical_analysis": _shrink_categorical(
            analysis_package.get("categorical_analysis", {}), 10
        ),
        "date_analysis": analysis_package.get("date_analysis", {}),
        "correlations": correlations,
        "strong_correlations": strong,
        "outliers": analysis_package.get("outliers", {}),
        "trends": _shrink_trends(analysis_package.get("trends", {})),
        "important_columns": analysis_package.get("important_columns", {}),
        "business_metrics": analysis_package.get("business_metrics", [])[:10],
        "ml_results": _shrink_ml(ml_package),
    }
    return safe_json(summary)


def _shrink_dict(d: Dict[str, Any], limit: int) -> Dict[str, Any]:
    out = {}
    for i, (k, v) in enumerate(d.items()):
        if i >= limit:
            break
        out[k] = v
    return out


def _shrink_categorical(d: Dict[str, Any], limit: int) -> Dict[str, Any]:
    out = {}
    for i, (k, v) in enumerate(d.items()):
        if i >= limit:
            break
        item = dict(v)
        if "distribution" in item:
            item["distribution"] = item["distribution"][:8]
        out[k] = item
    return out


def _shrink_trends(trends: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for i, (k, v) in enumerate(trends.items()):
        if i >= 6:
            break
        item = dict(v)
        if "points" in item:
            item["points"] = item["points"][-12:]
        out[k] = item
    return out


def _shrink_ml(ml_package: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not ml_package:
        return {"ml_used": False}
    return {
        "ml_used": ml_package.get("ml_used", False),
        "model_type": ml_package.get("model_type"),
        "target": ml_package.get("target"),
        "features": (ml_package.get("features") or [])[:15],
        "metrics": ml_package.get("metrics", {}),
        "important_features": (ml_package.get("important_features") or [])[:10],
        "message": ml_package.get("message"),
        "warnings": ml_package.get("warnings", []),
    }


def build_user_prompt(
    user_request: str,
    summary: Dict[str, Any],
    focus: str = "General overview",
    density: str = "Balanced",
) -> str:
    density_guidance = {
        "Balanced": "Generate a balanced set of all meaningfully useful charts.",
        "Detailed": "Be thorough — include supporting distributions and relationships.",
        "Maximum meaningful analysis": (
            "Maximize analytical coverage. Include every meaningful non-duplicate visualization "
            "justified by the schema and user intent."
        ),
    }
    guidance = density_guidance.get(density, density_guidance["Balanced"])

    return f"""
USER REQUEST:
{user_request}

DASHBOARD FOCUS: {focus}
CHART DENSITY: {density}
DENSITY GUIDANCE: {guidance}

DATASET ANALYSIS SUMMARY (JSON):
{truncate_text(to_json_string(summary), 12000)}

Design the complete dashboard plan now as strict JSON.
""".strip()


def generate_dashboard_plan(
    df,
    column_types: Dict[str, List[str]],
    user_request: str,
    api_key: str,
    analysis_package: Dict[str, Any],
    ml_package: Optional[Dict[str, Any]] = None,
    cleaning_report: Optional[Dict[str, Any]] = None,
    focus: str = "General overview",
    density: str = "Balanced",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Call OpenAI to produce a dashboard plan, then validate/repair it.
    """
    from validation_engine import parse_ai_json, validate_and_clean_plan

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    summary = build_dataset_summary(
        df, column_types, analysis_package, ml_package, cleaning_report
    )
    user_prompt = build_user_prompt(user_request, summary, focus, density)

    raw_plan = _call_openai(api_key, user_prompt, model=model)
    plan, parse_errors = parse_ai_json(raw_plan)
    if plan is None:
        # Fallback: rule-based plan
        plan = _rule_based_plan(df, column_types, user_request, density)
        plan["validation_warnings"] = parse_errors + [
            "AI response could not be parsed; used analytical fallback planner."
        ]
    else:
        plan = validate_and_clean_plan(plan, df, column_types)

    # Augment with rule-based charts for density=maximum if AI under-produced
    if density == "Maximum meaningful analysis":
        plan = _augment_with_recommendations(plan, df, column_types, density)

    # Ensure heatmap present when enough numerics
    plan = _ensure_core_charts(plan, df, column_types)
    return safe_json(plan)


def _call_openai(api_key: str, user_prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    return content or ""


def _rule_based_plan(df, column_types, user_request: str, density: str) -> Dict[str, Any]:
    from chart_engine import recommend_charts
    from validation_engine import validate_and_clean_plan

    dens = {
        "Balanced": "balanced",
        "Detailed": "detailed",
        "Maximum meaningful analysis": "maximum",
    }.get(density, "balanced")

    charts = recommend_charts(df, column_types, density=dens)
    cats = column_types.get("categorical", [])[:5]
    nums = column_types.get("numerical", [])
    dates = column_types.get("date", [])

    filters = [{"column": c, "reason": "Useful dimension filter"} for c in cats[:4]]
    if dates:
        filters.append({"column": dates[0], "reason": "Date range filter"})

    kpis = [{"name": "Total Records", "column": None, "aggregation": "count", "reason": "Volume"}]
    for col in nums[:4]:
        kpis.append(
            {"name": f"Total {col}", "column": col, "aggregation": "sum", "reason": "Metric total"}
        )
        kpis.append(
            {"name": f"Average {col}", "column": col, "aggregation": "mean", "reason": "Metric average"}
        )

    tables = [
        {
            "title": "Summary Preview",
            "columns": list(df.columns)[:8],
            "priority": 40,
            "reason": "Detail",
        }
    ]

    plan = {
        "dashboard_title": "Automated Analytics Dashboard",
        "description": f"Rule-assisted dashboard for: {user_request[:180]}",
        "filters": filters,
        "kpis": kpis,
        "visualizations": [c for c in charts if c.get("type") != "table"],
        "tables": tables,
        "insights_to_generate": [
            "Highlight top and bottom categories",
            "Summarize trends if dates exist",
            "Call out strongest correlations",
        ],
        "ml_analysis": {"recommended": False, "reason": "Deferred to ML engine"},
    }
    return validate_and_clean_plan(plan, df, column_types)


def _augment_with_recommendations(plan, df, column_types, density: str):
    from chart_engine import recommend_charts
    from validation_engine import remove_duplicate_visualizations

    dens = {
        "Balanced": "balanced",
        "Detailed": "detailed",
        "Maximum meaningful analysis": "maximum",
    }.get(density, "maximum")
    recommended = recommend_charts(df, column_types, density=dens)
    merged = list(plan.get("visualizations") or []) + recommended
    plan["visualizations"] = remove_duplicate_visualizations(merged)
    return plan


def _ensure_core_charts(plan, df, column_types):
    from validation_engine import remove_duplicate_visualizations

    viz = list(plan.get("visualizations") or [])
    nums = column_types.get("numerical", [])
    has_heatmap = any(v.get("type") == "heatmap" for v in viz)
    if len(nums) >= 2 and not has_heatmap:
        viz.append(
            {
                "type": "heatmap",
                "title": "Correlation Heatmap",
                "x": None,
                "y": None,
                "aggregation": None,
                "priority": 85,
                "size": "large",
                "reason": "Relationship analysis",
            }
        )
    plan["visualizations"] = remove_duplicate_visualizations(viz)
    return plan
