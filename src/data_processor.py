import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import BallTree

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BASE_FEATURES = ["area_sqm", "rooms", "latitude", "longitude", "construction_cost_m2"]
CATEGORICAL_FEATURES = ["department", "property_type"]
TIME_FEATURES = ["sale_year", "sale_month", "sale_quarter", "months_since_start"]
COMPARABLE_RADII_M = (500, 1000, 2000)
COMPARABLE_FEATURES = [
    feature
    for radius in COMPARABLE_RADII_M
    for feature in (
        f"comp_{radius}m_median_price_m2",
        f"comp_{radius}m_sale_count",
        f"comp_{radius}m_median_price",
    )
]
EARTH_RADIUS_M = 6_371_000

SOCIOECONOMIC_FEATURES = [
    "log_population", "pop_density", "poverty_rate", "log_income",
]

OSM_FEATURES = [
    "poi_transit", "poi_education", "poi_healthcare",
    "poi_shopping", "poi_dining", "poi_leisure", "poi_services",
]

TARGET = "price"


DVF_COLUMNS = {
    "valeur_fonciere": "price",
    "surface_reelle_bati": "area_sqm",
    "nombre_pieces_principales": "rooms",
    "surface_terrain": "land_sqm",
    "type_local": "property_type",
    "code_postal": "postal_code",
    "code_departement": "department",
    "code_commune": "commune_code",
    "nom_commune": "city",
}

PROPERTY_TYPE_MAP = {
    "Appartement": "apartment",
    "Maison": "house",
    "Local industriel. commercial ou assimil\u00e9": "commercial",
    "D\u00e9pendance": "other",
    "Local d'usage mixte (artisanal. commercial. bureau)": "commercial",
}

ARRONDISSEMENT_TO_COMMUNE = {
    **{f"751{i:02d}": "75056" for i in range(1, 21)},  # Paris
    **{f"132{i:02d}": "13055" for i in range(1, 17)},  # Marseille
    **{f"6938{i}": "69123" for i in range(1, 10)},    # Lyon
}


def normalize_commune_code(series: pd.Series) -> pd.Series:
    """Map arrondissement INSEE codes to their parent commune code."""
    codes = series.astype(str).str.strip().str.zfill(5)
    return codes.replace(ARRONDISSEMENT_TO_COMMUNE)


def _sale_dates(df: pd.DataFrame) -> pd.Series:
    if "date_mutation" in df.columns:
        return pd.to_datetime(df["date_mutation"], errors="coerce")
    if "sale_date" in df.columns:
        return pd.to_datetime(df["sale_date"], errors="coerce")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def get_time_metadata(df: pd.DataFrame) -> dict:
    dates = _sale_dates(df).dropna()
    if dates.empty:
        return {}
    return {
        "start_date": dates.min().strftime("%Y-%m-%d"),
        "max_date": dates.max().strftime("%Y-%m-%d"),
    }


def add_time_features(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add transaction-date features, using as_of_date for live prediction rows."""
    df = df.copy()
    dates = _sale_dates(df)

    fallback = pd.to_datetime(as_of_date, errors="coerce") if as_of_date is not None else pd.NaT
    if pd.isna(fallback):
        fallback = dates.max() if dates.notna().any() else pd.Timestamp.today().normalize()
    dates = dates.fillna(fallback)

    start = pd.to_datetime(start_date, errors="coerce") if start_date is not None else pd.NaT
    if pd.isna(start):
        start = dates.min() if dates.notna().any() else fallback

    df["sale_year"] = dates.dt.year.astype(float)
    df["sale_month"] = dates.dt.month.astype(float)
    df["sale_quarter"] = dates.dt.quarter.astype(float)
    df["months_since_start"] = ((dates.dt.year - start.year) * 12 + (dates.dt.month - start.month)).clip(lower=0).astype(float)
    return df


class ComparableFeatureBuilder:
    """Build nearby comparable-sale features from historical DVF transactions."""

    def __init__(
        self,
        source_df: pd.DataFrame,
        radii_m: tuple[int, ...] = COMPARABLE_RADII_M,
        max_neighbors: int = 80,
        max_samples: int = 400_000,
    ):
        self.radii_m = tuple(radii_m)
        self.max_neighbors = max_neighbors
        source = source_df.copy()
        required = ["latitude", "longitude", "price", "area_sqm"]
        missing = [c for c in required if c not in source.columns]
        if missing:
            source = pd.DataFrame(columns=required + ["property_type"])
        else:
            source["latitude"] = pd.to_numeric(source["latitude"], errors="coerce")
            source["longitude"] = pd.to_numeric(source["longitude"], errors="coerce")
            source["price"] = pd.to_numeric(source["price"], errors="coerce")
            source["area_sqm"] = pd.to_numeric(source["area_sqm"], errors="coerce")
            source = source.dropna(subset=required)
            source = source[(source["area_sqm"] > 0) & (source["price"] > 0)]

        if len(source) > max_samples:
            source = source.sample(max_samples, random_state=42)
            print(f"  Comparable index subsampled: {len(source):,} rows")

        source["_source_index"] = source.index.to_numpy()
        if "property_type" not in source.columns:
            source["property_type"] = "all"
        source["property_type"] = source["property_type"].fillna("all").astype(str)
        source["price_per_sqm"] = source["price"] / source["area_sqm"].clip(lower=1)

        self.groups = {}
        for property_type, group in source.groupby("property_type"):
            if group.empty:
                continue
            coords = np.radians(group[["latitude", "longitude"]].to_numpy(dtype=float))
            self.groups[property_type] = {
                "tree": BallTree(coords, metric="haversine"),
                "price_per_sqm": group["price_per_sqm"].to_numpy(dtype=float),
                "price": group["price"].to_numpy(dtype=float),
                "source_index": group["_source_index"].to_numpy(),
            }

    def transform(self, df: pd.DataFrame, include_self: bool = True) -> pd.DataFrame:
        out = df.copy()
        for feature in COMPARABLE_FEATURES:
            out[feature] = 0.0

        if not self.groups or "latitude" not in out.columns or "longitude" not in out.columns:
            return out

        work = out.copy()
        work["latitude"] = pd.to_numeric(work["latitude"], errors="coerce")
        work["longitude"] = pd.to_numeric(work["longitude"], errors="coerce")
        valid = work.dropna(subset=["latitude", "longitude"])
        if valid.empty:
            return out

        if "property_type" not in valid.columns:
            valid["property_type"] = "all"
        valid["property_type"] = valid["property_type"].fillna("all").astype(str)

        for property_type, rows in valid.groupby("property_type"):
            group = self.groups.get(property_type) or self.groups.get("all")
            if group is None:
                continue

            coords = np.radians(rows[["latitude", "longitude"]].to_numpy(dtype=float))
            tree = group["tree"]
            row_idx_arr = rows.index.to_numpy()
            n_rows = len(rows)

            # Query a bounded nearest-neighbor set once. Radius queries in dense urban
            # areas can otherwise materialize thousands of candidates per prediction.
            k = min(self.max_neighbors, len(group["price"]))
            distances, neighbor_idx = tree.query(coords, k=k)
            if k == 1:
                distances = distances.reshape(-1, 1)
                neighbor_idx = neighbor_idx.reshape(-1, 1)

            for radius in self.radii_m:
                r_rad = radius / EARTH_RADIUS_M
                counts = np.zeros(n_rows, dtype=float)
                med_ppsqm = np.zeros(n_rows, dtype=float)
                med_price = np.zeros(n_rows, dtype=float)

                for i in range(n_rows):
                    n_idx = neighbor_idx[i][distances[i] <= r_rad]
                    if not include_self:
                        n_idx = n_idx[group["source_index"][n_idx] != row_idx_arr[i]]
                    counts[i] = float(len(n_idx))
                    if len(n_idx):
                        med_ppsqm[i] = float(np.median(group["price_per_sqm"][n_idx]))
                        med_price[i] = float(np.median(group["price"][n_idx]))

                out.loc[rows.index, f"comp_{radius}m_sale_count"] = counts
                out.loc[rows.index, f"comp_{radius}m_median_price_m2"] = med_ppsqm
                out.loc[rows.index, f"comp_{radius}m_median_price"] = med_price

        return out


def add_comparable_features(
    df: pd.DataFrame,
    source_df: pd.DataFrame | None = None,
    builder: ComparableFeatureBuilder | None = None,
    include_self: bool = False,
) -> pd.DataFrame:
    if builder is None:
        builder = ComparableFeatureBuilder(df if source_df is None else source_df)
    return builder.transform(df, include_self=include_self)


def add_prior_year_comparable_features(
    df: pd.DataFrame,
    target_years: tuple[int, ...] | list[int],
) -> pd.DataFrame:
    """Add comparable features using only transactions from earlier calendar years.

    This is intentionally stricter than excluding the current row: it prevents a sale
    from seeing any current-year, future, validation, or test-set price in its features.
    It is used by the release training pipeline for temporal backtesting.
    """
    if "date_mutation" not in df.columns:
        raise ValueError("date_mutation is required for point-in-time comparable features")

    out = df.copy()
    dates = pd.to_datetime(out["date_mutation"], errors="coerce")
    for feature in COMPARABLE_FEATURES:
        out[feature] = 0.0

    for year in sorted({int(year) for year in target_years}):
        target_mask = dates.dt.year.eq(year)
        if not target_mask.any():
            continue
        history = out.loc[dates < pd.Timestamp(year=year, month=1, day=1)]
        if history.empty:
            continue
        builder = ComparableFeatureBuilder(history)
        out.loc[target_mask, COMPARABLE_FEATURES] = builder.transform(
            out.loc[target_mask], include_self=True
        )[COMPARABLE_FEATURES]
    return out


def load_data(path: str | Path = None) -> pd.DataFrame:
    if path is None:
        dvf_dir = DATA_DIR / "dvf"
        for filename in ("cleaned_final.parquet", "cleaned_release.parquet", "cleaned.parquet"):
            candidate = dvf_dir / filename
            if candidate.exists():
                path = candidate
                break
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def from_dvf(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns=DVF_COLUMNS)
    if "property_type" in df.columns:
        df["property_type"] = df["property_type"].map(PROPERTY_TYPE_MAP).fillna("other")
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].astype(str).str.zfill(5)
    if "department" in df.columns:
        df["department"] = df["department"].astype(str).str.zfill(2)
    if "commune_code" in df.columns:
        df["commune_code"] = normalize_commune_code(df["commune_code"])
    if "land_sqm" in df.columns:
        df["land_sqm"] = pd.to_numeric(df["land_sqm"], errors="coerce").fillna(0)
    return df


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "price" in df.columns and "area_sqm" in df.columns:
        df["price_per_sqm"] = df["price"] / df["area_sqm"].clip(lower=1)
        lower = df["price_per_sqm"].quantile(0.01)
        upper = df["price_per_sqm"].quantile(0.99)
        df = df[(df["price_per_sqm"] >= lower) & (df["price_per_sqm"] <= upper)]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_price"] = np.log1p(df["price"])
    df["log_area"] = np.log1p(df["area_sqm"].clip(lower=1))
    df["rooms_per_area"] = df["rooms"] / df["area_sqm"].clip(lower=1)
    df = add_time_features(df)
    return df


def merge_socioeconomic(df: pd.DataFrame) -> pd.DataFrame:
    """Merge INSEE commune-level data into the property DataFrame."""
    insee_path = DATA_DIR / "insee" / "insee_communes.parquet"
    if not insee_path.exists():
        print("  WARNING: INSEE data not found — skipping socioeconomic features")
        return df
    insee = pd.read_parquet(insee_path)
    # Find commune code column in DVF data
    cc_col = None
    for candidate in ["commune_code", "code_commune", "postal_code"]:
        if candidate in df.columns:
            cc_col = candidate
            break
    if cc_col is None:
        print("  WARNING: no commune code column found — skipping INSEE merge")
        return df
    if cc_col == "postal_code":
        # Postal codes are only a fallback; take first five characters.
        df["commune_code"] = df[cc_col].astype(str).str.zfill(5).str[:5]
    else:
        df["commune_code"] = normalize_commune_code(df[cc_col])
    before = len(df)
    df = df.merge(insee[["commune_code"] + SOCIOECONOMIC_FEATURES], on="commune_code", how="left")
    unmatched = int(df[SOCIOECONOMIC_FEATURES].isna().all(axis=1).sum())
    pct = (unmatched / len(df) * 100) if len(df) else 0
    print(f"  Merged INSEE data: {len(df)} rows ({unmatched} unmatched, {pct:.1f}%)")
    return df


def merge_osm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge OSM point-of-interest counts by commune."""
    osm_path = DATA_DIR / "osm_features" / "commune_poi_counts.parquet"
    if not osm_path.exists():
        print("  WARNING: OSM features not found — skipping")
        return df
    osm = pd.read_parquet(osm_path)
    # Rename columns to match expected feature names
    rename = {
        "transit": "poi_transit", "education": "poi_education",
        "healthcare": "poi_healthcare", "shopping": "poi_shopping",
        "dining": "poi_dining", "leisure": "poi_leisure",
        "services": "poi_services",
    }
    osm = osm.rename(columns=rename)
    osm_cols = ["commune_code"] + list(rename.values())
    if "commune_code" in df.columns:
        df = df.merge(osm[osm_cols], on="commune_code", how="left")
    return df


def merge_construction_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Add construction cost per m² by department."""
    try:
        from construction_benchmarks import get_all_features as get_construction
    except ImportError:
        from src.construction_benchmarks import get_all_features as get_construction
    costs = get_construction()
    if "department" in df.columns:
        df["department"] = df["department"].astype(str)
        costs["department"] = costs["department"].astype(str)
        df = df.merge(costs, on="department", how="left")
    df["construction_cost_m2"] = df.get("construction_cost_m2", pd.Series(dtype=float)).fillna(1800)
    return df


def prepare_features(df: pd.DataFrame, return_frame: bool = False):
    df = engineer_features(df)
    df = filter_outliers(df)

    # Merge external data sources
    df = merge_socioeconomic(df)
    df = merge_osm_features(df)
    df = merge_construction_cost(df)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    df_with_dummies = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    dummy_cols = []
    for col in cat_cols:
        dummy_cols += [c for c in df_with_dummies.columns if c.startswith(f"{col}_")]

    base_cols = [c for c in BASE_FEATURES if c in df_with_dummies.columns]
    extra_cols = [c for c in SOCIOECONOMIC_FEATURES + OSM_FEATURES + TIME_FEATURES + COMPARABLE_FEATURES if c in df_with_dummies.columns]
    feature_cols = base_cols + extra_cols + dummy_cols

    X = df_with_dummies[feature_cols].fillna(0)
    y = np.log1p(df["price"].values)

    if return_frame:
        return X, y, feature_cols, df
    return X, y, feature_cols
