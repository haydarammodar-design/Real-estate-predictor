"""Train an indicative residual-land-value model from house transactions.

DVF does not reliably provide nationwide vacant-land sales. The target is therefore the
transaction price less an indicative replacement construction cost. It is a transparent
land-value proxy, not a cadastral valuation or a planning opinion.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

pd.Int64Index = pd.Index

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processor import COMPARABLE_FEATURES, SOCIOECONOMIC_FEATURES, TIME_FEATURES
from src.modeling import (
    RELEASE_EVALUATION_YEAR,
    RELEASE_TRAIN_YEAR,
    evaluate_predictions,
    feature_frames,
    filter_target_range,
    load_release_feature_frame,
    release_time_metadata,
    split_release_years,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LAND_FEATURES = [
    "land_sqm",
    "latitude",
    "longitude",
    "construction_cost_m2",
    *SOCIOECONOMIC_FEATURES,
    *TIME_FEATURES,
    *COMPARABLE_FEATURES,
]


def _new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _with_residual_land_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["property_type"].eq("house")].copy()
    out["land_sqm"] = pd.to_numeric(out["land_sqm"], errors="coerce").fillna(0)
    out = out.loc[out["land_sqm"].gt(0)].copy()
    estimated_building_cost = out["area_sqm"] * out["construction_cost_m2"]
    out["residual_land_value"] = (out["price"] - estimated_building_cost).clip(lower=10_000)
    out["residual_land_price_per_sqm"] = out["residual_land_value"] / out["land_sqm"].clip(lower=1)
    return out


def train(data_path: str | Path | None = None, rebuild_features: bool = False) -> dict:
    df = _with_residual_land_target(load_release_feature_frame(data_path, rebuild=rebuild_features))
    train_df, evaluation_df = split_release_years(df)
    train_df, evaluation_df, bounds = filter_target_range(
        train_df,
        evaluation_df,
        train_df["residual_land_price_per_sqm"],
        evaluation_df["residual_land_price_per_sqm"],
        0.01,
        0.99,
    )
    X_train, X_evaluation, feature_cols = feature_frames(
        train_df, evaluation_df, LAND_FEATURES, ["department"]
    )
    y_train = np.log1p(train_df["residual_land_value"])
    y_evaluation = np.log1p(evaluation_df["residual_land_value"])

    evaluation_model = _new_model()
    evaluation_model.fit(X_train, y_train)
    evaluation_predictions = np.expm1(evaluation_model.predict(X_evaluation))
    metrics, geographic_metrics = evaluate_predictions(
        np.expm1(y_evaluation), evaluation_predictions, evaluation_df["department"]
    )
    metrics["type"] = "residual_land_proxy"

    final_df = pd.concat([train_df, evaluation_df], ignore_index=True)
    final_frame = pd.get_dummies(final_df, columns=["department"], dtype=float)
    X_final = final_frame.reindex(columns=feature_cols, fill_value=0).fillna(0)
    final_model = _new_model()
    final_model.fit(X_final, np.log1p(final_df["residual_land_value"]))

    departments = sorted(column.replace("department_", "") for column in feature_cols if column.startswith("department_"))
    artifacts = {
        "model": final_model,
        "feature_cols": feature_cols,
        "departments": departments,
        "time_metadata": release_time_metadata(),
        "comparable_features": [feature for feature in feature_cols if feature.startswith("comp_")],
        "comparable_cutoff": release_time_metadata()["max_date"],
        "geographic_metrics": geographic_metrics,
        "metrics": metrics,
        "target_definition": "transaction price minus benchmark replacement construction cost",
        "validation": {
            "protocol": "point-in-time annual backtest",
            "train_year": RELEASE_TRAIN_YEAR,
            "evaluation_year": RELEASE_EVALUATION_YEAR,
            "outlier_bounds": bounds,
            "training_records": int(len(train_df)),
            "evaluation_records": int(len(evaluation_df)),
            "final_training_records": int(len(final_df)),
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "model_land.joblib"
    joblib.dump(artifacts, path)
    print(f"Residual land-value proxy saved to {path}")
    print(f"2025 backtest: R²={metrics['r2']:.3f}, MAE={metrics['mae']:,.0f} EUR, median APE={metrics['mape_pct']:.1f}%")
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AlfaScript residual land-value proxy")
    parser.add_argument("--data", default=None, help="Path to cleaned DVF parquet or CSV")
    parser.add_argument("--rebuild-features", action="store_true", help="Rebuild point-in-time comparable features")
    args = parser.parse_args()
    train(args.data, rebuild_features=args.rebuild_features)
