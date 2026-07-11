"""Train and save the price prediction model (XGBoost)."""
import sys, warnings
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

pd.Int64Index = pd.Index

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.data_processor import COMPARABLE_FEATURES, get_time_metadata, load_data, from_dvf, prepare_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _metrics_by_department(test_departments, y_actual, y_pred, min_count: int = 30) -> dict:
    if test_departments is None:
        return {}
    departments = pd.Series(test_departments).fillna("").astype(str).to_numpy()
    results = {}
    for dept in sorted(set(departments)):
        mask = departments == dept
        if int(mask.sum()) < min_count:
            continue
        dept_actual = y_actual[mask]
        dept_pred = y_pred[mask]
        dept_mape = np.median(np.abs((dept_actual - dept_pred) / dept_actual.clip(1))) * 100
        results[dept] = {
            "count": int(mask.sum()),
            "mae": float(mean_absolute_error(dept_actual, dept_pred)),
            "rmse": float(np.sqrt(mean_squared_error(dept_actual, dept_pred))),
            "r2": float(r2_score(dept_actual, dept_pred)) if len(dept_actual) > 1 else 0.0,
            "mape_pct": round(float(dept_mape), 1),
        }
    return results


def train(data_path: str | Path = None):
    df = load_data(data_path)
    if df.empty:
        raise ValueError("No data loaded")

    is_dvf = "valeur_fonciere" in df.columns or "surface_reelle_bati" in df.columns
    if is_dvf:
        df = from_dvf(df)

    print(f"Loaded {len(df):,} records")

    X, y, feature_cols, prepared_df = prepare_features(df, return_frame=True)
    print(f"Features: {len(feature_cols)}, Records after cleaning: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
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
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_actual = np.expm1(y_test)

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    median_price = np.median(y_actual)
    mape = np.median(np.abs((y_actual - y_pred) / y_actual)) * 100
    dept_metrics = _metrics_by_department(
        prepared_df.loc[X_test.index, "department"] if "department" in prepared_df.columns else None,
        y_actual,
        y_pred,
    )

    print(f"\nModel Performance:")
    print(f"  MAE:  {mae:,.0f} EUR")
    print(f"  RMSE: {rmse:,.0f} EUR")
    print(f"  R2:   {r2:.3f}")
    print(f"  Median Price: {median_price:,.0f} EUR")
    print(f"  Median Error: {mape:.1f}%")
    print(f"  Department validation groups: {len(dept_metrics)}")

    dept_cols = sorted([c for c in feature_cols if c.startswith("department_")])
    type_cols = sorted([c for c in feature_cols if c.startswith("property_type_")])

    artifacts = {
        "model": model,
        "feature_cols": feature_cols,
        "departments": [c.replace("department_", "") for c in dept_cols],
        "property_types": [c.replace("property_type_", "") for c in type_cols],
        "time_metadata": get_time_metadata(prepared_df),
        "comparable_features": [c for c in COMPARABLE_FEATURES if c in feature_cols],
        "geographic_metrics": dept_metrics,
        "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "mape_pct": round(float(mape), 1)},
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "model.joblib"
    joblib.dump(artifacts, path)
    print(f"\nModel saved to {path}")
    return artifacts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to data file (csv/parquet)")
    args = parser.parse_args()
    train(args.data)
