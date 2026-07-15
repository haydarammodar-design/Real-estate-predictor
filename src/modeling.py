"""Shared, point-in-time training utilities for AlfaScript release models."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_processor import (
    COMPARABLE_FEATURES,
    SOCIOECONOMIC_FEATURES,
    TIME_FEATURES,
    add_prior_year_comparable_features,
    add_time_features,
    from_dvf,
    load_data,
    merge_construction_cost,
    merge_socioeconomic,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RELEASE_TRAIN_YEAR = 2024
RELEASE_EVALUATION_YEAR = 2025
RELEASE_START_DATE = "2021-01-01"


def release_time_metadata() -> dict:
    return {
        "start_date": RELEASE_START_DATE,
        "max_date": f"{RELEASE_EVALUATION_YEAR}-12-31",
    }


def load_release_feature_frame(
    data_path: str | Path | None = None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Build the 2024/2025 point-in-time feature frame used by all release models."""
    cache_path = DATA_DIR / "derived" / "release_features_2024_2025.parquet"
    if cache_path.exists() and not rebuild:
        return pd.read_parquet(cache_path)

    df = load_data(data_path)
    if df.empty:
        raise ValueError("No DVF data loaded")
    if "valeur_fonciere" in df.columns or "surface_reelle_bati" in df.columns:
        df = from_dvf(df)

    required = {"price", "area_sqm", "rooms", "latitude", "longitude", "property_type", "date_mutation"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DVF data is missing required columns: {sorted(missing)}")

    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce")
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    residential = df["property_type"].isin(["apartment", "house"])
    required_period = df["date_mutation"].lt(pd.Timestamp(year=RELEASE_EVALUATION_YEAR + 1, month=1, day=1))
    df = df.loc[residential & required_period].copy()
    df = df[
        df["price"].ge(10_000)
        & df["area_sqm"].between(9, 10_000)
        & df["rooms"].gt(0)
        & df["latitude"].between(41, 52)
        & df["longitude"].between(-6, 10)
    ].copy()
    if df.empty:
        raise ValueError("No valid residential DVF transactions found")

    target_mask = df["date_mutation"].dt.year.isin([RELEASE_TRAIN_YEAR, RELEASE_EVALUATION_YEAR])
    target = df.loc[target_mask].copy()
    target = add_prior_year_comparable_features(
        pd.concat([df.loc[~target_mask], target], axis=0),
        target_years=[RELEASE_TRAIN_YEAR, RELEASE_EVALUATION_YEAR],
    ).loc[target.index]
    target = add_time_features(target, start_date=RELEASE_START_DATE)
    target = merge_socioeconomic(target)
    target = merge_construction_cost(target)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    target.to_parquet(cache_path, index=False)
    return target


def split_release_years(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date_mutation"], errors="coerce")
    train = df.loc[dates.dt.year.eq(RELEASE_TRAIN_YEAR)].copy()
    evaluation = df.loc[dates.dt.year.eq(RELEASE_EVALUATION_YEAR)].copy()
    if train.empty or evaluation.empty:
        raise ValueError("Release training requires both 2024 and 2025 transactions")
    return train, evaluation


def filter_target_range(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    target: pd.Series,
    evaluation_target: pd.Series,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit target bounds on 2024 only and apply those fixed bounds to 2025."""
    low = float(target.quantile(lower_quantile))
    high = float(target.quantile(upper_quantile))
    train = train.loc[target.between(low, high)].copy()
    evaluation = evaluation.loc[evaluation_target.between(low, high)].copy()
    return train, evaluation, {"lower": low, "upper": high}


def feature_frames(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_frame = pd.get_dummies(train, columns=categorical_features, dtype=float)
    evaluation_frame = pd.get_dummies(evaluation, columns=categorical_features, dtype=float)
    dummy_columns = [
        column
        for category in categorical_features
        for column in train_frame.columns
        if column.startswith(f"{category}_")
    ]
    feature_columns = [column for column in numeric_features if column in train_frame.columns] + sorted(dummy_columns)
    X_train = train_frame.reindex(columns=feature_columns, fill_value=0).fillna(0)
    X_evaluation = evaluation_frame.reindex(columns=feature_columns, fill_value=0).fillna(0)
    return X_train, X_evaluation, feature_columns


def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    departments: pd.Series,
) -> tuple[dict, dict]:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    absolute_percentage_error = np.abs((actual - predicted) / np.clip(actual, 1, None))
    ratios = actual / np.clip(predicted, 1, None)
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "mape_pct": round(float(np.median(absolute_percentage_error) * 100), 1),
        "mean_ape_pct": round(float(np.mean(absolute_percentage_error) * 100), 1),
        "within_10_pct": round(float(np.mean(absolute_percentage_error <= 0.10) * 100), 1),
        "within_20_pct": round(float(np.mean(absolute_percentage_error <= 0.20) * 100), 1),
        "within_30_pct": round(float(np.mean(absolute_percentage_error <= 0.30) * 100), 1),
        "prediction_interval": {
            "coverage": 0.80,
            "lower_multiplier": round(float(np.quantile(ratios, 0.10)), 4),
            "upper_multiplier": round(float(np.quantile(ratios, 0.90)), 4),
        },
    }

    geographic_metrics = {}
    for department in sorted(departments.fillna("").astype(str).unique()):
        mask = departments.fillna("").astype(str).to_numpy() == department
        if int(mask.sum()) < 30:
            continue
        dept_actual = actual[mask]
        dept_predicted = predicted[mask]
        dept_ape = np.abs((dept_actual - dept_predicted) / np.clip(dept_actual, 1, None))
        geographic_metrics[department] = {
            "count": int(mask.sum()),
            "mae": float(mean_absolute_error(dept_actual, dept_predicted)),
            "rmse": float(np.sqrt(mean_squared_error(dept_actual, dept_predicted))),
            "r2": float(r2_score(dept_actual, dept_predicted)) if len(dept_actual) > 1 else 0.0,
            "mape_pct": round(float(np.median(dept_ape) * 100), 1),
        }
    return metrics, geographic_metrics


def model_features(include_land: bool = False) -> list[str]:
    core = [
        "area_sqm",
        "rooms",
        "latitude",
        "longitude",
        "construction_cost_m2",
        *SOCIOECONOMIC_FEATURES,
        *TIME_FEATURES,
        *COMPARABLE_FEATURES,
    ]
    return (["land_sqm"] if include_land else []) + core
