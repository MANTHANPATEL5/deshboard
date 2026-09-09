"""
Shared utility helpers for the AI Dashboard Generator.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


RANDOM_STATE = 42

# Performance safeguards (NOT arbitrary chart-count limits)
MAX_CATEGORY_VALUES_IN_CHART = 30
MAX_SCATTER_POINTS = 5000
MAX_CORR_COLUMNS = 40
MAX_ROWS_FOR_ML = 200_000
MIN_ROWS_FOR_ML = 30
MAX_PIE_CATEGORIES = 8
MAX_FILTER_CARDINALITY = 100
MAX_TABLE_ROWS = 200
MAX_PROFILE_TOP_VALUES = 10
SAMPLE_ROWS_FOR_AI = 5
HIGH_CARDINALITY_RATIO = 0.90
IDENTIFIER_UNIQUE_RATIO = 0.95


def safe_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert pandas/numpy scalars to plain Python floats safely."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        if hasattr(value, "item"):
            value = value.item()
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return default
        return num
    except (TypeError, ValueError):
        return default


def format_number(value: Any, decimals: int = 2) -> str:
    """Format numbers for KPI / insight display."""
    num = safe_number(value)
    if num is None:
        return "N/A"
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.{decimals}f}B"
    if abs_num >= 1_000_000:
        return f"{num / 1_000_000:.{decimals}f}M"
    if abs_num >= 1_000:
        return f"{num / 1_000:.{decimals}f}K"
    if abs_num >= 100 or float(num).is_integer():
        return f"{num:,.0f}"
    return f"{num:,.{decimals}f}"


def format_percent(value: Any, decimals: int = 1) -> str:
    num = safe_number(value)
    if num is None:
        return "N/A"
    return f"{num:.{decimals}f}%"


def safe_json(value: Any) -> Any:
    """Recursively convert objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_json(v) for v in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, pd.Series):
        return safe_json(value.tolist())
    if isinstance(value, pd.DataFrame):
        return safe_json(value.to_dict(orient="records"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return safe_number(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if hasattr(value, "item"):
        try:
            return safe_json(value.item())
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def to_json_string(obj: Any, indent: int = 2) -> str:
    return json.dumps(safe_json(obj), indent=indent, ensure_ascii=False)


def clean_column_name(name: Any) -> str:
    text = str(name).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def make_unique_columns(columns: Sequence[Any]) -> List[str]:
    """Preserve meaningful names while ensuring uniqueness."""
    seen: Dict[str, int] = {}
    result: List[str] = []
    for col in columns:
        base = clean_column_name(col) or "column"
        key = base.lower()
        if key not in seen:
            seen[key] = 0
            result.append(base)
        else:
            seen[key] += 1
            result.append(f"{base}_{seen[key]}")
    return result


def looks_like_identifier(name: str) -> bool:
    lowered = str(name).lower().strip()
    patterns = [
        r"(^|_)id($|_)",
        r"uuid",
        r"guid",
        r"sku",
        r"invoice",
        r"order[_ ]?id",
        r"transaction",
        r"customer[_ ]?id",
        r"employee[_ ]?id",
        r"account[_ ]?number",
        r"ssn",
        r"passport",
        r"tracking",
        r"reference",
    ]
    return any(re.search(p, lowered) for p in patterns)


def looks_like_metric(name: str) -> bool:
    lowered = str(name).lower()
    keywords = [
        "amount", "revenue", "sales", "profit", "cost", "price", "qty",
        "quantity", "total", "count", "score", "rate", "ratio", "margin",
        "salary", "wage", "fee", "charge", "balance", "value", "spend",
        "budget", "income", "expense", "units", "volume", "duration",
        "latency", "conversion", "clicks", "impressions", "sessions",
    ]
    return any(k in lowered for k in keywords)


def looks_like_dimension(name: str) -> bool:
    lowered = str(name).lower()
    keywords = [
        "region", "country", "state", "city", "category", "segment",
        "department", "product", "gender", "status", "type", "channel",
        "brand", "store", "branch", "team", "role", "grade", "class",
        "source", "campaign", "device", "browser", "os",
    ]
    return any(k in lowered for k in keywords)


def looks_like_date(name: str) -> bool:
    lowered = str(name).lower()
    keywords = ["date", "time", "timestamp", "datetime", "year", "month", "day", "period"]
    return any(k in lowered for k in keywords)


def looks_like_geo(name: str) -> bool:
    lowered = str(name).lower()
    keywords = [
        "country", "state", "city", "region", "province", "latitude",
        "longitude", "lat", "lon", "lng", "zip", "postal",
    ]
    return any(k in lowered for k in keywords)


BOOLEAN_TRUE = {"true", "t", "yes", "y", "1", "on"}
BOOLEAN_FALSE = {"false", "f", "no", "n", "0", "off"}


def is_boolean_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    sample = series.dropna().astype(str).str.strip().str.lower()
    if sample.empty:
        return False
    unique = set(sample.unique())
    return unique.issubset(BOOLEAN_TRUE | BOOLEAN_FALSE) and len(unique) <= 2


def coerce_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    mapped = series.astype(str).str.strip().str.lower().map(
        {**{k: True for k in BOOLEAN_TRUE}, **{k: False for k in BOOLEAN_FALSE}}
    )
    return mapped.astype("boolean")


def file_size_label(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 ** 2):.2f} MB"


def dataframe_hash(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(payload).hexdigest()


def sample_dataframe(df: pd.DataFrame, n: int = SAMPLE_ROWS_FOR_AI) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    n = min(n, len(df))
    sample = df.head(n)
    return safe_json(sample.to_dict(orient="records"))


def truncate_text(text: str, max_len: int = 4000) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def correlation_strength(corr: float) -> str:
    abs_c = abs(corr)
    if abs_c >= 0.7:
        return "strong"
    if abs_c >= 0.4:
        return "moderate"
    if abs_c >= 0.2:
        return "weak"
    return "very weak"


def correlation_direction(corr: float) -> str:
    if corr > 0:
        return "positive"
    if corr < 0:
        return "negative"
    return "none"


def bytes_io_from_text(text: str, encoding: str = "utf-8") -> io.BytesIO:
    buffer = io.BytesIO(text.encode(encoding))
    buffer.seek(0)
    return buffer


def get_openai_api_key(secrets: Any = None, env_key: str = "OPENAI_API_KEY") -> Optional[str]:
    """Resolve API key from Streamlit secrets or environment variables."""
    import os

    if secrets is not None:
        try:
            key = secrets.get(env_key) if hasattr(secrets, "get") else secrets[env_key]
            if key:
                return str(key).strip()
        except Exception:
            pass
    return os.environ.get(env_key) or os.environ.get("OPENAI_KEY")


def priority_band(priority: Any) -> str:
    p = safe_number(priority, 50) or 50
    if p >= 90:
        return "critical"
    if p >= 70:
        return "important"
    if p >= 50:
        return "supporting"
    if p >= 30:
        return "optional"
    return "low"


def chart_size_columns(size: str) -> int:
    """Streamlit column span hint: 1=full, 2=half."""
    size = (size or "medium").lower()
    if size == "large":
        return 1
    if size == "small":
        return 2
    return 2
