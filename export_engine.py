"""
Export engine — downloads for cleaned data, plans, summaries, and reports.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from utils import safe_json, to_json_string


def export_csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def export_json(obj: Any) -> bytes:
    return to_json_string(obj).encode("utf-8")


def export_analysis_summary(
    summary: Dict[str, Any],
    cleaning_report: Dict[str, Any],
    analysis_package: Dict[str, Any],
    insights: List[Any],
    ml_package: Optional[Dict[str, Any]] = None,
) -> bytes:
    payload = {
        "dataset_summary": summary,
        "cleaning_report": cleaning_report,
        "analysis": {
            "quality": analysis_package.get("quality"),
            "important_columns": analysis_package.get("important_columns"),
            "strong_correlations": analysis_package.get("strong_correlations"),
            "business_metrics": analysis_package.get("business_metrics"),
            "trends": analysis_package.get("trends"),
        },
        "insights": insights,
        "ml": {
            "ml_used": (ml_package or {}).get("ml_used"),
            "model_type": (ml_package or {}).get("model_type"),
            "target": (ml_package or {}).get("target"),
            "metrics": (ml_package or {}).get("metrics"),
            "important_features": (ml_package or {}).get("important_features"),
            "message": (ml_package or {}).get("message"),
        },
    }
    return export_json(payload)


def export_dashboard_plan(plan: Dict[str, Any]) -> bytes:
    return export_json(plan)


def export_visualization_config(plan: Dict[str, Any]) -> bytes:
    payload = {
        "kpis": plan.get("kpis"),
        "filters": plan.get("filters"),
        "visualizations": plan.get("visualizations"),
        "tables": plan.get("tables"),
    }
    return export_json(payload)


def export_excel_report(
    df: pd.DataFrame,
    kpis: List[Dict[str, Any]],
    insights: List[Any],
    summary: Dict[str, Any],
    plan: Optional[Dict[str, Any]] = None,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.head(5000).to_excel(writer, sheet_name="Data", index=False)
        pd.DataFrame(kpis).to_excel(writer, sheet_name="KPIs", index=False)
        insight_rows = []
        for item in insights or []:
            if isinstance(item, dict):
                insight_rows.append(item)
            else:
                insight_rows.append({"message": str(item)})
        pd.DataFrame(insight_rows).to_excel(writer, sheet_name="Insights", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        if plan:
            pd.DataFrame(plan.get("visualizations") or []).to_excel(
                writer, sheet_name="Charts", index=False
            )
    buffer.seek(0)
    return buffer.getvalue()


def export_pdf_report(
    title: str,
    description: str,
    kpis: List[Dict[str, Any]],
    insights: List[Any],
    summary: Dict[str, Any],
) -> Optional[bytes]:
    """
    Optional lightweight PDF via reportlab if installed.
    Returns None if dependency is unavailable — core app still works.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(inch, y, title[:90])
    y -= 0.35 * inch
    c.setFont("Helvetica", 10)
    for line in _wrap(description, 95):
        c.drawString(inch, y, line)
        y -= 0.2 * inch

    y -= 0.2 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "Dataset Summary")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        c.drawString(inch, y, f"{k}: {v}")
        y -= 0.18 * inch
        if y < inch:
            c.showPage()
            y = height - inch

    y -= 0.15 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "KPIs")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    for kpi in kpis or []:
        c.drawString(inch, y, f"{kpi.get('name')}: {kpi.get('display')}")
        y -= 0.18 * inch
        if y < inch:
            c.showPage()
            y = height - inch

    y -= 0.15 * inch
    c.setFont("Helvetica-Bold", 12)
    c.drawString(inch, y, "Insights")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    for item in insights or []:
        msg = item.get("message") if isinstance(item, dict) else str(item)
        for line in _wrap(msg, 95):
            c.drawString(inch, y, line)
            y -= 0.18 * inch
            if y < inch:
                c.showPage()
                y = height - inch
        y -= 0.08 * inch

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def _wrap(text: str, width: int) -> List[str]:
    text = str(text or "")
    words = text.split()
    lines: List[str] = []
    current = ""
    for w in words:
        trial = f"{current} {w}".strip()
        if len(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [""]
