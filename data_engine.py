"""
Data loading, validation, cleaning, profiling, and type detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    HIGH_CARDINALITY_RATIO,
    IDENTIFIER_UNIQUE_RATIO,
    MAX_PROFILE_TOP_VALUES,
    clean_column_name,
    coerce_boolean,
    file_size_label,
    is_boolean_like,
    looks_like_date,
    looks_like_identifier,
    make_unique_columns,
    safe_number,
)


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# ==========================================================
# FILE VALIDATION & LOADING
# ==========================================================

def validate_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """Validate uploaded file before loading. Never crashes."""
    result = {
        "valid": False,
        "filename": None,
        "extension": None,
        "size_bytes": 0,
        "size_label": "0 B",
        "errors": [],
        "warnings": [],
    }

    if uploaded_file is None:
        result["errors"].append("No file uploaded.")
        return result

    filename = getattr(uploaded_file, "name", "") or ""
    result["filename"] = filename
    lower = filename.lower()

    ext = None
    for candidate in SUPPORTED_EXTENSIONS:
        if lower.endswith(candidate):
            ext = candidate
            break

    if ext is None:
        result["errors"].append(
            "Unsupported format. Please upload CSV, XLSX, or XLS."
        )
        return result

    result["extension"] = ext

    try:
        size = int(getattr(uploaded_file, "size", 0) or 0)
    except Exception:
        size = 0
    result["size_bytes"] = size
    result["size_label"] = file_size_label(size)

    if size == 0:
        # Some uploaders report 0 until read; soft warning only
        result["warnings"].append("File size reported as 0; will attempt to load.")

    result["valid"] = True
    return result


def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Load CSV/XLSX/XLS into a DataFrame.
    Returns (df, meta). df is None on failure.
    """
    meta = validate_uploaded_file(uploaded_file)
    if not meta["valid"]:
        return None, meta

    filename = meta["filename"].lower()

    try:
        raw = None
        if hasattr(uploaded_file, "getvalue"):
            raw = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            raw = uploaded_file.read()
            try:
                uploaded_file.seek(0)
            except Exception:
                pass

        if filename.endswith(".csv"):
            df = _load_csv(uploaded_file if raw is None else raw)
        elif filename.endswith(".xlsx"):
            from io import BytesIO

            source = BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else uploaded_file
            df = pd.read_excel(source, engine="openpyxl")
        elif filename.endswith(".xls"):
            from io import BytesIO

            source = BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else uploaded_file
            try:
                df = pd.read_excel(source, engine="xlrd")
            except Exception:
                source = BytesIO(raw) if isinstance(raw, (bytes, bytearray)) else uploaded_file
                df = pd.read_excel(source)
        else:
            meta["errors"].append("Unsupported format.")
            meta["valid"] = False
            return None, meta
    except Exception as exc:
        meta["valid"] = False
        meta["errors"].append(f"Could not read file: {exc}")
        return None, meta

    if df is None or df.empty:
        meta["valid"] = False
        meta["errors"].append("The file is empty or has no readable rows.")
        return None, meta

    if df.columns.isna().any() or any(str(c).startswith("Unnamed") for c in df.columns):
        meta["warnings"].append(
            "Some columns appear unnamed or missing headers."
        )

    # Duplicate column names before cleaning
    cols = [str(c) for c in df.columns]
    if len(cols) != len(set(cols)):
        meta["warnings"].append("Duplicate column names detected; they will be uniquified.")

    meta["rows"] = int(len(df))
    meta["columns"] = int(df.shape[1])
    meta["valid"] = True
    return df, meta


def _load_csv(uploaded_file) -> pd.DataFrame:
    """Try common CSV encodings/separators."""
    if isinstance(uploaded_file, (bytes, bytearray)):
        raw = bytes(uploaded_file)
    elif hasattr(uploaded_file, "getvalue"):
        raw = uploaded_file.getvalue()
    else:
        raw = uploaded_file.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")

    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        for sep in (None, ",", ";", "\t", "|"):
            try:
                from io import BytesIO

                buffer = BytesIO(raw)
                kwargs = {"encoding": encoding}
                if sep is None:
                    kwargs["sep"] = None
                    kwargs["engine"] = "python"
                else:
                    kwargs["sep"] = sep
                df = pd.read_csv(buffer, **kwargs)
                if df.shape[1] >= 1:
                    return df
            except Exception as exc:
                last_error = exc
                continue
    raise ValueError(f"Unable to parse CSV ({last_error})")


# ==========================================================
# COLUMN TYPE DETECTION
# ==========================================================

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Intelligent type detection beyond pandas dtypes.
    Returns numerical, categorical, date, boolean, text, identifier.
    """
    numerical: List[str] = []
    categorical: List[str] = []
    date_cols: List[str] = []
    boolean: List[str] = []
    text: List[str] = []
    identifier: List[str] = []

    n_rows = max(len(df), 1)

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        nunique = int(non_null.nunique())
        unique_ratio = nunique / max(len(non_null), 1)
        name = str(column)

        if is_boolean_like(series):
            boolean.append(column)
            continue

        # Native datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            date_cols.append(column)
            continue

        # Numeric BEFORE datetime parsing — integers can falsely parse as dates
        if pd.api.types.is_numeric_dtype(series):
            if looks_like_identifier(name) and unique_ratio >= IDENTIFIER_UNIQUE_RATIO:
                identifier.append(column)
            else:
                numerical.append(column)
            continue

        # Object / string columns — prefer numeric/date parse before ID rules
        numeric_ratio = _numeric_parse_ratio(non_null)
        if numeric_ratio >= 0.85 and nunique > 1:
            if looks_like_identifier(name) and unique_ratio >= IDENTIFIER_UNIQUE_RATIO:
                identifier.append(column)
            else:
                numerical.append(column)
            continue

        if _is_mostly_datetime(non_null) or (
            looks_like_date(name) and _datetime_parse_ratio(non_null) >= 0.6
        ):
            date_cols.append(column)
            continue

        # Explicit identifier names / ultra-high-cardinality codes
        if looks_like_identifier(name) or (
            unique_ratio >= IDENTIFIER_UNIQUE_RATIO
            and nunique > 50
            and not looks_like_date(name)
        ):
            identifier.append(column)
            continue

        # Text vs categorical
        avg_len = non_null.astype(str).str.len().mean() if len(non_null) else 0
        if unique_ratio >= HIGH_CARDINALITY_RATIO and avg_len and avg_len > 40:
            text.append(column)
        elif unique_ratio >= HIGH_CARDINALITY_RATIO and nunique > 100:
            text.append(column)
        else:
            categorical.append(column)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "date": date_cols,
        "boolean": boolean,
        "text": text,
        "identifier": identifier,
    }


def _numeric_parse_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    converted = pd.to_numeric(cleaned, errors="coerce")
    return float(converted.notna().mean())


def _datetime_parse_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            converted = pd.to_datetime(series, errors="coerce", format="mixed")
        except TypeError:
            converted = pd.to_datetime(series, errors="coerce")
    return float(converted.notna().mean())


def _is_mostly_datetime(series: pd.Series, threshold: float = 0.80) -> bool:
    """Detect date strings without treating plain numbers as dates."""
    if series.empty:
        return False
    # Reject pure numeric series (ints/floats parse as unix timestamps)
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.astype(str).str.strip()
    # Require at least one date separator in a majority of values
    sep_ratio = sample.str.contains(r"[-/\s:T]", regex=True, na=False).mean()
    if sep_ratio < 0.5:
        return False
    return _datetime_parse_ratio(series) >= threshold


# ==========================================================
# CLEANING
# ==========================================================

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Any]]:
    """
    Automatic intelligent cleaning with a structured report.
    Does not silently delete outliers.
    """
    actions: List[str] = []
    warnings: List[str] = []
    converted_columns: List[str] = []
    imputed_columns: List[str] = []
    constant_columns: List[str] = []
    identifier_columns: List[str] = []
    outlier_columns: Dict[str, Any] = {}

    original_rows = int(len(df))
    original_columns = int(df.shape[1])
    working = df.copy()

    # Drop fully empty rows/columns
    before = len(working)
    working = working.dropna(how="all")
    empty_rows = before - len(working)
    if empty_rows:
        actions.append(f"Removed {empty_rows} completely empty rows.")

    empty_cols = [c for c in working.columns if working[c].isna().all()]
    if empty_cols:
        working = working.drop(columns=empty_cols)
        actions.append(f"Dropped empty columns: {', '.join(map(str, empty_cols))}")

    # Column name cleaning
    new_cols = make_unique_columns(working.columns)
    if list(working.columns) != new_cols:
        actions.append("Normalized and uniquified column names.")
    working.columns = new_cols

    # Exact duplicate rows
    dup_count = int(working.duplicated().sum())
    if dup_count:
        working = working.drop_duplicates()
        actions.append(f"Removed {dup_count} exact duplicate rows.")

    # Whitespace cleaning for object columns
    for col in working.select_dtypes(include=["object", "string"]).columns:
        try:
            stripped = working[col].astype(str).where(working[col].notna(), other=np.nan)
            # Only strip non-null values
            mask = working[col].notna()
            working.loc[mask, col] = working.loc[mask, col].astype(str).str.strip()
        except Exception:
            warnings.append(f"Could not strip whitespace for column '{col}'.")

    # Initial type detection
    column_types = detect_column_types(working)

    # Convert numeric-looking strings (preserve nulls; don't stringify NaN)
    for col in list(column_types["numerical"]):
        if not pd.api.types.is_numeric_dtype(working[col]):
            mask = working[col].notna()
            cleaned = (
                working.loc[mask, col]
                .astype(str)
                .str.strip()
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            converted = pd.to_numeric(cleaned, errors="coerce")
            # Ratio among originally non-null values
            if len(converted) and float(converted.notna().mean()) >= 0.85:
                new_col = pd.Series(np.nan, index=working.index, dtype="float64")
                new_col.loc[mask] = converted.values
                working[col] = new_col
                converted_columns.append(col)
                actions.append(f"Converted '{col}' to numeric.")

    # Convert dates
    for col in list(column_types["date"]):
        if not pd.api.types.is_datetime64_any_dtype(working[col]):
            converted = pd.to_datetime(working[col], errors="coerce")
            if converted.notna().mean() >= 0.6:
                working[col] = converted
                converted_columns.append(col)
                actions.append(f"Converted '{col}' to datetime.")

    # Convert booleans
    for col in list(column_types["boolean"]):
        try:
            working[col] = coerce_boolean(working[col])
            converted_columns.append(col)
            actions.append(f"Converted '{col}' to boolean.")
        except Exception:
            warnings.append(f"Could not convert boolean column '{col}'.")

    # Re-detect after conversions
    column_types = detect_column_types(working)
    identifier_columns = list(column_types.get("identifier", []))

    missing_before = int(working.isna().sum().sum())

    # Intelligent missing-value handling
    for col in column_types["numerical"]:
        miss = int(working[col].isna().sum())
        if miss == 0:
            continue
        miss_pct = miss / max(len(working), 1)
        if miss_pct > 0.6:
            warnings.append(
                f"Column '{col}' has {miss_pct:.0%} missing; left as-is (meaningful missingness)."
            )
            continue
        numeric_col = pd.to_numeric(working[col], errors="coerce")
        median_value = numeric_col.median()
        if pd.notna(median_value):
            working[col] = numeric_col.fillna(median_value)
            imputed_columns.append(col)
            actions.append(f"Imputed '{col}' missing values with median ({median_value}).")

    for col in column_types["categorical"] + column_types["boolean"]:
        miss = int(working[col].isna().sum())
        if miss == 0:
            continue
        miss_pct = miss / max(len(working), 1)
        if miss_pct > 0.6:
            warnings.append(
                f"Column '{col}' has {miss_pct:.0%} missing; left as-is."
            )
            continue
        mode = working[col].mode(dropna=True)
        fill_value = mode.iloc[0] if len(mode) else "Unknown"
        working[col] = working[col].fillna(fill_value)
        imputed_columns.append(col)
        actions.append(f"Imputed '{col}' missing values with mode/Unknown.")

    for col in column_types["date"]:
        # Do not blindly impute dates
        miss = int(working[col].isna().sum())
        if miss:
            warnings.append(
                f"Date column '{col}' has {miss} missing values; not imputed."
            )

    missing_after = int(working.isna().sum().sum())

    # Constant / low-information columns
    for col in working.columns:
        nunique = working[col].nunique(dropna=True)
        if nunique <= 1:
            constant_columns.append(col)
    if constant_columns:
        actions.append(
            f"Marked low-information constant columns: {', '.join(constant_columns)}"
        )
        warnings.append("Constant columns retained but flagged as low-information.")

    # Outlier detection (flag only)
    for col in column_types["numerical"]:
        outlier_info = detect_outliers_iqr(working[col])
        if outlier_info["count"] > 0:
            outlier_columns[col] = outlier_info

    if outlier_columns:
        actions.append(
            f"Detected outliers in {len(outlier_columns)} numerical column(s) (flagged, not deleted)."
        )

    # Final type detection
    column_types = detect_column_types(working)

    quality_score, quality_explain = calculate_data_quality_score(
        original_rows=original_rows,
        final_df=working,
        duplicates_removed=dup_count,
        missing_before=missing_before,
        missing_after=missing_after,
        constant_columns=constant_columns,
    )

    cleaning_report: Dict[str, Any] = {
        "original_rows": original_rows,
        "final_rows": int(len(working)),
        "original_columns": original_columns,
        "final_columns": int(working.shape[1]),
        "duplicates_removed": dup_count,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "converted_columns": list(dict.fromkeys(converted_columns)),
        "imputed_columns": list(dict.fromkeys(imputed_columns)),
        "outlier_columns": outlier_columns,
        "constant_columns": constant_columns,
        "identifier_columns": identifier_columns,
        "warnings": warnings,
        "actions_taken": actions,
        "data_quality_score": quality_score,
        "data_quality_explanation": quality_explain,
    }

    return working, column_types, cleaning_report


def detect_outliers_iqr(series: pd.Series) -> Dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 8:
        return {"count": 0, "percentage": 0.0, "method": "iqr", "lower": None, "upper": None}

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        # fallback z-score style
        mean = numeric.mean()
        std = numeric.std()
        if not std or std == 0:
            return {"count": 0, "percentage": 0.0, "method": "none", "lower": None, "upper": None}
        z = (numeric - mean).abs() / std
        count = int((z > 3).sum())
        return {
            "count": count,
            "percentage": round(100.0 * count / len(numeric), 2),
            "method": "zscore",
            "lower": safe_number(mean - 3 * std),
            "upper": safe_number(mean + 3 * std),
        }

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (numeric < lower) | (numeric > upper)
    count = int(mask.sum())
    return {
        "count": count,
        "percentage": round(100.0 * count / len(numeric), 2),
        "method": "iqr",
        "lower": safe_number(lower),
        "upper": safe_number(upper),
    }


def calculate_data_quality_score(
    original_rows: int,
    final_df: pd.DataFrame,
    duplicates_removed: int,
    missing_before: int,
    missing_after: int,
    constant_columns: List[str],
) -> Tuple[float, str]:
    """
    Data quality score (0-100).
    Starts at 100 and subtracts penalties for missingness, duplicates,
    constant columns, and remaining missing values after cleaning.
    """
    score = 100.0
    reasons: List[str] = []

    cells = max(original_rows * max(final_df.shape[1], 1), 1)
    missing_pct = 100.0 * missing_before / cells
    if missing_pct > 0:
        penalty = min(35.0, missing_pct * 0.7)
        score -= penalty
        reasons.append(f"-{penalty:.1f} for {missing_pct:.1f}% missing values before cleaning")

    if original_rows > 0:
        dup_pct = 100.0 * duplicates_removed / original_rows
        if dup_pct > 0:
            penalty = min(20.0, dup_pct * 0.5)
            score -= penalty
            reasons.append(f"-{penalty:.1f} for {dup_pct:.1f}% duplicate rows")

    if final_df.shape[1] > 0 and constant_columns:
        const_pct = 100.0 * len(constant_columns) / final_df.shape[1]
        penalty = min(15.0, const_pct * 0.4)
        score -= penalty
        reasons.append(f"-{penalty:.1f} for {len(constant_columns)} constant column(s)")

    remaining_cells = max(len(final_df) * max(final_df.shape[1], 1), 1)
    remain_pct = 100.0 * missing_after / remaining_cells
    if remain_pct > 0:
        penalty = min(20.0, remain_pct * 0.5)
        score -= penalty
        reasons.append(f"-{penalty:.1f} for remaining missing values after cleaning")

    score = max(0.0, min(100.0, round(score, 1)))
    explanation = (
        "Score starts at 100 and applies penalties for missing values, "
        "duplicates, constant columns, and remaining gaps. "
        + (" ".join(reasons) if reasons else "No major quality issues detected.")
    )
    return score, explanation


# ==========================================================
# PROFILING
# ==========================================================

def generate_profile(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Complete column-level profile."""
    type_lookup = {}
    for t, cols in column_types.items():
        for c in cols:
            type_lookup[c] = t

    profiles: List[Dict[str, Any]] = []
    n = max(len(df), 1)

    for column in df.columns:
        series = df[column]
        col_type = type_lookup.get(column, "text")
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        entry: Dict[str, Any] = {
            "name": column,
            "dtype": str(series.dtype),
            "inferred_type": col_type,
            "missing_count": missing,
            "missing_percentage": round(100.0 * missing / n, 2),
            "unique_count": unique,
            "unique_percentage": round(100.0 * unique / n, 2),
        }

        if col_type == "numerical":
            numeric = pd.to_numeric(series, errors="coerce")
            entry.update(
                {
                    "minimum": safe_number(numeric.min()),
                    "maximum": safe_number(numeric.max()),
                    "mean": safe_number(numeric.mean()),
                    "median": safe_number(numeric.median()),
                    "standard_deviation": safe_number(numeric.std()),
                    "quantiles": {
                        "q25": safe_number(numeric.quantile(0.25)),
                        "q50": safe_number(numeric.quantile(0.50)),
                        "q75": safe_number(numeric.quantile(0.75)),
                    },
                    "outlier_count": detect_outliers_iqr(numeric)["count"],
                }
            )
        elif col_type == "categorical" or col_type == "boolean":
            vc = series.astype(str).value_counts(dropna=True).head(MAX_PROFILE_TOP_VALUES)
            total = max(int(series.notna().sum()), 1)
            entry["top_values"] = [
                {
                    "value": str(idx),
                    "count": int(cnt),
                    "percentage": round(100.0 * cnt / total, 2),
                }
                for idx, cnt in vc.items()
            ]
            mode = series.mode(dropna=True)
            entry["mode"] = str(mode.iloc[0]) if len(mode) else None
        elif col_type == "date":
            dates = pd.to_datetime(series, errors="coerce")
            valid = dates.dropna()
            if not valid.empty:
                entry["minimum_date"] = valid.min().isoformat()
                entry["maximum_date"] = valid.max().isoformat()
                entry["date_range_days"] = int((valid.max() - valid.min()).days)
        elif col_type == "identifier":
            entry["note"] = "High-cardinality identifier; avoid as chart dimension."

        profiles.append(entry)

    return profiles


def generate_profile_dataframe(df: pd.DataFrame, column_types: Dict[str, List[str]]) -> pd.DataFrame:
    """UI-friendly profile table."""
    rows = []
    for p in generate_profile(df, column_types):
        rows.append(
            {
                "Column": p["name"],
                "Type": p["inferred_type"],
                "Missing %": p["missing_percentage"],
                "Unique": p["unique_count"],
                "Min": p.get("minimum", p.get("minimum_date", "-")),
                "Max": p.get("maximum", p.get("maximum_date", "-")),
                "Mean": p.get("mean", "-"),
                "Median": p.get("median", "-"),
            }
        )
    return pd.DataFrame(rows)


def summarize_dataset(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    cleaning_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    missing = int(df.isna().sum().sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numerical_variables": len(column_types.get("numerical", [])),
        "categorical_variables": len(column_types.get("categorical", [])),
        "date_variables": len(column_types.get("date", [])),
        "boolean_variables": len(column_types.get("boolean", [])),
        "text_variables": len(column_types.get("text", [])),
        "identifier_variables": len(column_types.get("identifier", [])),
        "missing_values": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "data_quality_score": (cleaning_report or {}).get("data_quality_score"),
        "memory_usage_mb": round(float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 3),
    }
