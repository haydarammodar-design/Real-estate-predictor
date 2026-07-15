"""Train the point-in-time general property valuation model."""
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

from src.modeling import (
    RELEASE_EVALUATION_YEAR,
    RELEASE_TRAIN_YEAR,
    evaluate_predictions,
    feature_frames,
    filter_target_range,
    load_release_feature_frame,
    model_features,
    release_time_metadata,
    split_release_years,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        max_bin=256,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def train(data_path: str | Path | None = None, rebuild_features: bool = False) -> dict:
    df = load_release_feature_frame(data_path, rebuild=rebuild_features)
    train_df, evaluation_df = split_release_years(df)

    train_target = train_df["price"] / train_df["area_sqm"].clip(lower=1)
    evaluation_target = evaluation_df["price"] / evaluation_df["area_sqm"].clip(lower=1)
    train_df, evaluation_df, bounds = filter_target_range(
        train_df, evaluation_df, train_target, evaluation_target, 0.01, 0.99
    )
    X_train, X_evaluation, feature_cols = feature_frames(
        train_df,
        evaluation_df,
        model_features(),
        ["department", "property_type"],
    )
    y_train = np.log1p(train_df["price"])
    y_evaluation = np.log1p(evaluation_df["price"])

    evaluation_model = _new_model()
    evaluation_model.fit(X_train, y_train)
    evaluation_predictions = np.expm1(evaluation_model.predict(X_evaluation))
    metrics, geographic_metrics = evaluate_predictions(
        np.expm1(y_evaluation), evaluation_predictions, evaluation_df["department"]
    )

    final_df = pd.concat([train_df, evaluation_df], ignore_index=True)
    final_frame = pd.get_dummies(final_df, columns=["department", "property_type"], dtype=float)
    X_final = final_frame.reindex(columns=feature_cols, fill_value=0).fillna(0)
    final_model = _new_model()
    final_model.fit(X_final, np.log1p(final_df["price"]))

    departments = sorted(column.replace("department_", "") for column in feature_cols if column.startswith("department_"))
    property_types = sorted(column.replace("property_type_", "") for column in feature_cols if column.startswith("property_type_"))
    artifacts = {
        "model": final_model,
        "feature_cols": feature_cols,
        "departments": departments,
        "property_types": property_types,
        "time_metadata": release_time_metadata(),
        "comparable_features": [feature for feature in feature_cols if feature.startswith("comp_")],
        "comparable_cutoff": release_time_metadata()["max_date"],
        "geographic_metrics": geographic_metrics,
        "metrics": metrics,
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
    path = MODELS_DIR / "model.joblib"
    joblib.dump(artifacts, path)
    print(f"General model saved to {path}")
    print(f"2025 backtest: R²={metrics['r2']:.3f}, MAE={metrics['mae']:,.0f} EUR, median APE={metrics['mape_pct']:.1f}%")
    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AlfaScript general valuation model")
    parser.add_argument("--data", default=None, help="Path to cleaned DVF parquet or CSV")
    parser.add_argument("--rebuild-features", action="store_true", help="Rebuild point-in-time comparable features")
    args = parser.parse_args()
    train(args.data, rebuild_features=args.rebuild_features)
