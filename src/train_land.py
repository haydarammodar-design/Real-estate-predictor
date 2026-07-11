"""Train Model 1: Land/plot price estimator (XGBoost).

Trains on house transactions (which bundle land + building).
Predicts total property value from land characteristics.
"""
import sys, warnings
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

pd.Int64Index = pd.Index

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processor import (
    COMPARABLE_FEATURES,
    TIME_FEATURES,
    add_comparable_features,
    add_time_features,
    get_time_metadata,
    load_data,
    from_dvf,
    merge_socioeconomic,
    merge_construction_cost,
    filter_outliers,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LAND_FEATURES = [
    "area_sqm", "rooms", "land_sqm", "latitude", "longitude",
    "construction_cost_m2",
    "log_population", "pop_density", "poverty_rate", "log_income",
] + TIME_FEATURES + COMPARABLE_FEATURES


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

    # Filter to houses (they have land)
    house = df[df["property_type"] == "house"].copy()
    print(f"Houses with land: {len(house):,}")

    if house.empty:
        raise ValueError("No house transactions found")

    # Feature engineering
    house = merge_socioeconomic(house)
    house = merge_construction_cost(house)

    # Add price_per_land_m2
    house["land_sqm"] = pd.to_numeric(house.get("land_sqm", 0), errors="coerce").fillna(0)
    house = house[house["land_sqm"] > 0]
    print(f"Houses with positive land area: {len(house):,}")
    house["total_area"] = house["area_sqm"] + house["land_sqm"] * 0.3  # land contributes ~30%

    # Filter outliers by price/land_m2
    house["price_per_land"] = house["price"] / house["land_sqm"].clip(lower=1)
    q_low, q_high = house["price_per_land"].quantile(0.01), house["price_per_land"].quantile(0.99)
    house = house[(house["price_per_land"] >= q_low) & (house["price_per_land"] <= q_high)]
    house = add_time_features(house)
    house = add_comparable_features(house, source_df=house, include_self=False)

    print(f"After filtering: {len(house):,} records")

    # Build feature matrix
    cat_cols = ["department", "property_type"]
    df_with_dummies = pd.get_dummies(house, columns=cat_cols, drop_first=False)

    extra = [c for c in LAND_FEATURES if c in df_with_dummies.columns]
    dept_cols = sorted([c for c in df_with_dummies.columns if c.startswith("department_")])
    feature_cols = extra + dept_cols

    X = df_with_dummies[feature_cols].fillna(0)
    y = np.log1p(house["price"].values)  # log-transform target

    print(f"Features: {len(feature_cols)}, Records: {len(X):,}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
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
    mape = np.median(np.abs((y_actual - y_pred) / y_actual.clip(1))) * 100
    dept_metrics = _metrics_by_department(
        house.loc[X_test.index, "department"] if "department" in house.columns else None,
        y_actual,
        y_pred,
    )

    print(f"\nModel 1 — Land Price Estimator (XGBoost)")
    print(f"  MAE:  {mae:,.0f} EUR")
    print(f"  RMSE: {rmse:,.0f} EUR")
    print(f"  R²:   {r2:.3f}")
    print(f"  Median Error: {mape:.1f}%")
    print(f"  Department validation groups: {len(dept_metrics)}")

    departments = sorted([c.replace("department_", "") for c in dept_cols])

    artifacts = {
        "model": model,
        "feature_cols": feature_cols,
        "departments": departments,
        "time_metadata": get_time_metadata(house),
        "comparable_features": [c for c in COMPARABLE_FEATURES if c in feature_cols],
        "geographic_metrics": dept_metrics,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape_pct": round(float(mape), 1),
            "type": "land",
        },
    }

    path = MODELS_DIR / "model_land.joblib"
    joblib.dump(artifacts, path)
    print(f"\nModel saved to {path}")

    # Feature importance
    importances = model.feature_importances_
    top = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]
    print("\nTop 15 features:")
    for name, imp in top:
        print(f"  {name}: {imp:.4f}")

    return artifacts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    train(args.data)
