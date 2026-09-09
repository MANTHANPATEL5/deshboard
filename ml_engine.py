"""
Machine learning engine.

Trains real models only when analytically useful.
Python calculates metrics; AI does not invent ML results.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils import (
    MIN_ROWS_FOR_ML,
    RANDOM_STATE,
    looks_like_identifier,
    looks_like_metric,
    safe_json,
    safe_number,
)


def evaluate_ml_opportunity(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    user_request: str = "",
    analysis_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide whether ML is useful and which task to run."""
    n = len(df)
    nums = column_types.get("numerical", [])
    cats = column_types.get("categorical", []) + column_types.get("boolean", [])
    dates = column_types.get("date", [])
    ids = set(column_types.get("identifier", []))
    request = (user_request or "").lower()

    if n < MIN_ROWS_FOR_ML:
        return {
            "recommended": False,
            "reason": f"Dataset too small for reliable ML ({n} rows; need ≥ {MIN_ROWS_FOR_ML}).",
        }

    # Explicit user intent
    wants_forecast = any(k in request for k in ("forecast", "predict future", "time series", "projection"))
    wants_cluster = any(k in request for k in ("cluster", "segment", "segmentation", "group customers"))
    wants_anomaly = any(k in request for k in ("anomal", "outlier detection", "fraud"))
    wants_classify = any(k in request for k in ("classif", "churn", "attrition", "predict whether", "likelihood"))
    wants_regress = any(k in request for k in ("predict", "regression", "estimate", "forecast value"))

    # Classification target: low-cardinality categorical not identifier
    class_targets = [
        c
        for c in cats
        if c not in ids
        and 2 <= df[c].nunique(dropna=True) <= 12
        and df[c].nunique(dropna=True) / max(n, 1) < 0.2
    ]
    # Prefer binary / name hints
    preferred_class = [
        c
        for c in class_targets
        if any(
            k in c.lower()
            for k in ("churn", "attrition", "status", "target", "label", "default", "fraud", "outcome")
        )
    ]

    regress_targets = [
        c for c in nums if c not in ids and looks_like_metric(c)
    ] or [c for c in nums if c not in ids]

    feature_pool_num = [c for c in nums if c not in ids]
    feature_pool_cat = [
        c
        for c in cats
        if c not in ids and 1 < df[c].nunique(dropna=True) <= 30
    ]

    if wants_forecast and dates and regress_targets:
        return {
            "recommended": True,
            "type": "forecasting",
            "target": regress_targets[0],
            "date_column": dates[0],
            "reason": "Date column and metric available for basic forecasting.",
        }

    if wants_anomaly and len(feature_pool_num) >= 2:
        return {
            "recommended": True,
            "type": "anomaly",
            "target": None,
            "reason": "User requested anomaly detection with sufficient numerical features.",
        }

    if (wants_classify or preferred_class) and class_targets and (
        len(feature_pool_num) + len(feature_pool_cat) >= 2
    ):
        target = preferred_class[0] if preferred_class else class_targets[0]
        return {
            "recommended": True,
            "type": "classification",
            "target": target,
            "reason": f"Suitable categorical target '{target}' detected for classification.",
        }

    if wants_regress and regress_targets and len(feature_pool_num) + len(feature_pool_cat) >= 2:
        target = regress_targets[0]
        return {
            "recommended": True,
            "type": "regression",
            "target": target,
            "reason": f"Suitable numerical target '{target}' detected for regression.",
        }

    if wants_cluster and len(feature_pool_num) >= 2:
        return {
            "recommended": True,
            "type": "clustering",
            "target": None,
            "reason": "User requested segmentation and numerical features are available.",
        }

    # Auto: classification if clear target-like column
    if preferred_class and len(feature_pool_num) + len(feature_pool_cat) >= 2:
        return {
            "recommended": True,
            "type": "classification",
            "target": preferred_class[0],
            "reason": f"Target-like column '{preferred_class[0]}' suggests classification.",
        }

    # Auto: regression if enough features and a strong metric
    if len(regress_targets) >= 1 and len(feature_pool_num) >= 3:
        return {
            "recommended": True,
            "type": "regression",
            "target": regress_targets[0],
            "reason": f"Multiple numerical features enable regression on '{regress_targets[0]}'.",
        }

    # Auto: clustering when no clear target
    if len(feature_pool_num) >= 2 and not class_targets:
        return {
            "recommended": True,
            "type": "clustering",
            "target": None,
            "reason": "No clear prediction target; clustering can reveal segments.",
        }

    # Light anomaly pass when many outliers flagged
    outliers = (analysis_package or {}).get("outliers", {})
    if outliers and len(feature_pool_num) >= 2:
        return {
            "recommended": True,
            "type": "anomaly",
            "target": None,
            "reason": "Multiple outlier columns detected; anomaly detection may add value.",
        }

    if dates and regress_targets and len(df) >= 60:
        return {
            "recommended": True,
            "type": "forecasting",
            "target": regress_targets[0],
            "date_column": dates[0],
            "reason": "Time dimension available for a basic forecast.",
        }

    return {
        "recommended": False,
        "reason": (
            "Machine learning was not applied because no reliable prediction "
            "target or suitable ML opportunity was detected."
        ),
    }


def run_ml_analysis(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    user_request: str = "",
    analysis_package: Optional[Dict[str, Any]] = None,
    opportunity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute ML when appropriate. Always returns a structured package.
    """
    opportunity = opportunity or evaluate_ml_opportunity(
        df, column_types, user_request, analysis_package
    )

    base = {
        "ml_used": False,
        "model_type": None,
        "target": None,
        "features": [],
        "metrics": {},
        "important_features": [],
        "warnings": [],
        "predictions": None,
        "visualizations": [],
        "opportunity": opportunity,
        "message": opportunity.get("reason", ""),
    }

    if not opportunity.get("recommended"):
        return safe_json(base)

    ml_type = opportunity.get("type")
    try:
        if ml_type == "classification":
            result = _run_classification(df, column_types, opportunity["target"])
        elif ml_type == "regression":
            result = _run_regression(df, column_types, opportunity["target"])
        elif ml_type == "clustering":
            result = _run_clustering(df, column_types)
        elif ml_type == "anomaly":
            result = _run_anomaly(df, column_types)
        elif ml_type == "forecasting":
            result = _run_forecast(
                df,
                column_types,
                opportunity.get("target"),
                opportunity.get("date_column"),
            )
        else:
            base["warnings"].append(f"Unknown ML type: {ml_type}")
            return safe_json(base)
        result["opportunity"] = opportunity
        result["ml_used"] = True
        return safe_json(result)
    except Exception as exc:
        base["warnings"].append(f"ML failed: {exc}")
        base["message"] = (
            "Machine learning was attempted but failed validation or training. "
            f"Details: {exc}"
        )
        return safe_json(base)


def _feature_columns(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    target: Optional[str],
) -> Tuple[List[str], List[str]]:
    ids = set(column_types.get("identifier", []))
    exclude = ids | {target} if target else ids
    # Also exclude obvious ID-like names
    for c in list(df.columns):
        if looks_like_identifier(c):
            exclude.add(c)

    num_features = [
        c
        for c in column_types.get("numerical", [])
        if c not in exclude and c in df.columns
    ]
    cat_features = [
        c
        for c in column_types.get("categorical", []) + column_types.get("boolean", [])
        if c not in exclude
        and c in df.columns
        and 1 < df[c].nunique(dropna=True) <= 30
    ]
    # Cap features for performance
    return num_features[:20], cat_features[:12]


def _build_preprocessor(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    transformers = []
    if num_features:
        transformers.append(("num", StandardScaler(), num_features))
    if cat_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            )
        )
    if not transformers:
        raise ValueError("No usable features for ML.")
    return ColumnTransformer(transformers=transformers)


def _run_classification(
    df: pd.DataFrame, column_types: Dict[str, List[str]], target: str
) -> Dict[str, Any]:
    num_features, cat_features = _feature_columns(df, column_types, target)
    features = num_features + cat_features
    if len(features) < 1:
        raise ValueError("Insufficient features for classification.")

    data = df[features + [target]].dropna(subset=[target]).copy()
    if len(data) < MIN_ROWS_FOR_ML:
        raise ValueError("Too few rows after dropping missing targets.")

    # Limit rare-class issues
    y = data[target].astype(str)
    counts = y.value_counts()
    valid_classes = counts[counts >= 5].index
    data = data[y.isin(valid_classes)]
    y = data[target].astype(str)
    if y.nunique() < 2:
        raise ValueError("Target does not have enough class diversity.")

    X = data[features]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify
    )

    preprocessor = _build_preprocessor(num_features, cat_features)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=8
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    best_name = None
    best_pipe = None
    best_score = -1.0
    best_pred = None

    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = f1_score(y_test, pred, average="weighted", zero_division=0)
        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe
            best_pred = pred

    assert best_pipe is not None and best_pred is not None
    metrics = {
        "accuracy": float(accuracy_score(y_test, best_pred)),
        "precision": float(
            precision_score(y_test, best_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, best_pred, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(y_test, best_pred, average="weighted", zero_division=0)),
    }
    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, best_pred, labels=labels)

    important = _tree_feature_importance(best_pipe, num_features, cat_features)

    return {
        "ml_used": True,
        "model_type": f"classification:{best_name}",
        "target": target,
        "features": features,
        "metrics": metrics,
        "important_features": important,
        "warnings": [],
        "predictions": {
            "classes": labels,
            "prediction_distribution": {
                str(k): int(v) for k, v in pd.Series(best_pred).value_counts().items()
            },
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "visualizations": [
            {"type": "confusion_matrix"},
            {"type": "feature_importance"},
            {"type": "prediction_distribution"},
        ],
        "message": f"Trained {best_name} classifier on '{target}'.",
    }


def _run_regression(
    df: pd.DataFrame, column_types: Dict[str, List[str]], target: str
) -> Dict[str, Any]:
    num_features, cat_features = _feature_columns(df, column_types, target)
    features = num_features + cat_features
    if not features:
        raise ValueError("Insufficient features for regression.")

    data = df[features + [target]].copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])
    if len(data) < MIN_ROWS_FOR_ML:
        raise ValueError("Too few rows for regression.")

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )
    preprocessor = _build_preprocessor(num_features, cat_features)

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, max_depth=8
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    best_name = None
    best_pipe = None
    best_r2 = -1e9
    best_pred = None

    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        r2 = r2_score(y_test, pred)
        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_pipe = pipe
            best_pred = pred

    assert best_pipe is not None and best_pred is not None
    residuals = (y_test - best_pred).tolist()
    metrics = {
        "mae": float(mean_absolute_error(y_test, best_pred)),
        "mse": float(mean_squared_error(y_test, best_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, best_pred))),
        "r2": float(r2_score(y_test, best_pred)),
    }
    important = _tree_feature_importance(best_pipe, num_features, cat_features)

    # Sample for plots
    sample_n = min(500, len(y_test))
    idx = np.linspace(0, len(y_test) - 1, sample_n).astype(int)

    return {
        "ml_used": True,
        "model_type": f"regression:{best_name}",
        "target": target,
        "features": features,
        "metrics": metrics,
        "important_features": important,
        "warnings": [],
        "predictions": {
            "actual": [safe_number(v) for v in np.array(y_test)[idx].tolist()],
            "predicted": [safe_number(v) for v in np.array(best_pred)[idx].tolist()],
            "residuals": [safe_number(v) for v in np.array(residuals)[idx].tolist()],
        },
        "visualizations": [
            {"type": "actual_vs_predicted"},
            {"type": "residuals"},
            {"type": "feature_importance"},
        ],
        "message": f"Trained {best_name} regressor on '{target}'.",
    }


def _run_clustering(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    num_features, _ = _feature_columns(df, column_types, target=None)
    if len(num_features) < 2:
        raise ValueError("Need at least 2 numerical features for clustering.")

    data = df[num_features].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < MIN_ROWS_FOR_ML:
        raise ValueError("Too few complete rows for clustering.")

    # Sample for speed if huge
    if len(data) > 20000:
        data = data.sample(20000, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    best_k = 2
    best_sil = -1.0
    inertias = []
    max_k = min(8, max(2, len(data) // 20))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append({"k": k, "inertia": float(km.inertia_)})
        sil = float(silhouette_score(X, labels))
        if sil > best_sil:
            best_sil = sil
            best_k = k

    model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)
    data = data.copy()
    data["cluster"] = labels

    summary = (
        data.groupby("cluster")[num_features]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    distribution = {
        str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()
    }

    # 2D projection using first two features for scatter
    scatter = data[num_features[:2] + ["cluster"]].copy()
    if len(scatter) > 3000:
        scatter = scatter.sample(3000, random_state=RANDOM_STATE)

    return {
        "ml_used": True,
        "model_type": "clustering:kmeans",
        "target": None,
        "features": num_features,
        "metrics": {
            "n_clusters": best_k,
            "silhouette": best_sil,
            "elbow": inertias,
        },
        "important_features": [],
        "warnings": [],
        "predictions": {
            "cluster_distribution": distribution,
            "cluster_summary": safe_json(summary),
            "scatter": {
                "x": num_features[0],
                "y": num_features[1],
                "points": scatter.to_dict(orient="records"),
            },
        },
        "visualizations": [
            {"type": "cluster_distribution"},
            {"type": "cluster_scatter"},
            {"type": "cluster_summary"},
        ],
        "message": f"K-Means found {best_k} clusters (silhouette={best_sil:.3f}).",
    }


def _run_anomaly(
    df: pd.DataFrame, column_types: Dict[str, List[str]]
) -> Dict[str, Any]:
    num_features, _ = _feature_columns(df, column_types, target=None)
    if len(num_features) < 2:
        raise ValueError("Need numerical features for anomaly detection.")

    data = df[num_features].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < MIN_ROWS_FOR_ML:
        raise ValueError("Too few rows for anomaly detection.")

    model = IsolationForest(
        n_estimators=100, contamination="auto", random_state=RANDOM_STATE
    )
    preds = model.fit_predict(data)
    scores = model.score_samples(data)
    anomaly_flags = preds == -1
    n_anom = int(anomaly_flags.sum())

    sample = data.copy()
    sample["anomaly"] = anomaly_flags
    sample["score"] = scores
    if len(sample) > 3000:
        # keep anomalies + sample of normals
        anom = sample[sample["anomaly"]]
        normal = sample[~sample["anomaly"]].sample(
            min(2500, (~sample["anomaly"]).sum()), random_state=RANDOM_STATE
        )
        plot_df = pd.concat([anom, normal], ignore_index=True)
    else:
        plot_df = sample

    return {
        "ml_used": True,
        "model_type": "anomaly:isolation_forest",
        "target": None,
        "features": num_features,
        "metrics": {
            "anomaly_count": n_anom,
            "anomaly_rate": float(n_anom / max(len(data), 1)),
        },
        "important_features": [],
        "warnings": [],
        "predictions": {
            "anomaly_count": n_anom,
            "scatter": {
                "x": num_features[0],
                "y": num_features[1],
                "points": plot_df[num_features[:2] + ["anomaly", "score"]].to_dict(
                    orient="records"
                ),
            },
        },
        "visualizations": [
            {"type": "anomaly_distribution"},
            {"type": "anomaly_scatter"},
        ],
        "message": f"Isolation Forest flagged {n_anom} unusual observations.",
    }


def _run_forecast(
    df: pd.DataFrame,
    column_types: Dict[str, List[str]],
    target: Optional[str],
    date_column: Optional[str],
) -> Dict[str, Any]:
    dates = column_types.get("date", [])
    nums = column_types.get("numerical", [])
    date_col = date_column or (dates[0] if dates else None)
    metric = target or (nums[0] if nums else None)
    if not date_col or not metric:
        raise ValueError("Forecasting requires a date column and numerical metric.")

    temp = df[[date_col, metric]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
    temp = temp.dropna()
    if temp.empty:
        raise ValueError("No valid date/metric rows for forecasting.")

    monthly = (
        temp.set_index(date_col)
        .resample("MS")[metric]
        .sum()
        .dropna()
    )
    if len(monthly) < 6:
        raise ValueError("Need at least 6 monthly points for forecasting.")

    # Simple linear trend on time index + seasonal month dummies via OLS-like approach
    y = monthly.values.astype(float)
    t = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(t, y)

    horizon = min(6, max(3, len(y) // 4))
    future_t = np.arange(len(y), len(y) + horizon).reshape(-1, 1)
    forecast = model.predict(future_t)
    fitted = model.predict(t)
    resid_std = float(np.std(y - fitted)) if len(y) > 2 else 0.0

    history = [
        {"period": str(idx.date()), "value": safe_number(val), "type": "history"}
        for idx, val in monthly.items()
    ]
    last_date = monthly.index.max()
    future = []
    for i, val in enumerate(forecast, start=1):
        period = (last_date + pd.DateOffset(months=i)).date().isoformat()
        future.append(
            {
                "period": period,
                "value": safe_number(val),
                "lower": safe_number(val - 1.96 * resid_std),
                "upper": safe_number(val + 1.96 * resid_std),
                "type": "forecast",
            }
        )

    # Holdout style metrics on last 20%
    split = max(3, int(len(y) * 0.8))
    if split < len(y):
        model_h = LinearRegression()
        model_h.fit(t[:split], y[:split])
        pred_h = model_h.predict(t[split:])
        metrics = {
            "mae": float(mean_absolute_error(y[split:], pred_h)),
            "rmse": float(np.sqrt(mean_squared_error(y[split:], pred_h))),
            "r2": float(r2_score(y[split:], pred_h)) if len(y[split:]) > 1 else None,
            "horizon_months": horizon,
        }
    else:
        metrics = {"horizon_months": horizon}

    return {
        "ml_used": True,
        "model_type": "forecasting:linear_trend",
        "target": metric,
        "features": [date_col],
        "metrics": metrics,
        "important_features": [],
        "warnings": [
            "Basic linear-trend forecast; not a full seasonal ARIMA model."
        ],
        "predictions": {
            "history": history,
            "forecast": future,
            "date_column": date_col,
            "metric": metric,
        },
        "visualizations": [{"type": "forecast"}],
        "message": f"Generated {horizon}-month forecast for '{metric}'.",
    }


def _tree_feature_importance(
    pipe: Pipeline, num_features: List[str], cat_features: List[str]
) -> List[Dict[str, Any]]:
    model = pipe.named_steps.get("model")
    if model is None or not hasattr(model, "feature_importances_"):
        return []

    prep = pipe.named_steps["prep"]
    try:
        feature_names = list(prep.get_feature_names_out())
    except Exception:
        feature_names = num_features + cat_features

    importances = model.feature_importances_
    # Aggregate one-hot groups back to original categorical columns when possible
    agg: Dict[str, float] = {}
    for name, imp in zip(feature_names, importances):
        raw = str(name)
        if raw.startswith("num__"):
            key = raw.replace("num__", "", 1)
        elif raw.startswith("cat__"):
            key = raw.replace("cat__", "", 1)
            # OneHotEncoder name like col_value
            for c in cat_features:
                if key.startswith(c):
                    key = c
                    break
        else:
            key = raw
        agg[key] = agg.get(key, 0.0) + float(imp)

    total = sum(agg.values()) or 1.0
    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    return [
        {"feature": k, "importance": round(v / total, 4)}
        for k, v in ranked[:15]
    ]
