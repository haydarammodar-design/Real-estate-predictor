"""FastAPI server for real estate price prediction."""
import io, math, hashlib, time, asyncio, xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2
import sys
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from src.data_processor import ComparableFeatureBuilder, add_time_features, from_dvf, get_time_metadata

# XGBoost 2.0.3 compat with pandas 2.x — Int64Index removed in pandas 2.1
pd.Int64Index = pd.Index

import geopandas as gpd
from shapely.geometry import Point
from sqlalchemy import create_engine, text as sql_text
import numpy_financial as npf
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
import mercantile

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Load all models
_artifacts = None
model = None
feature_cols = []
departments = []
property_types = []
metrics = {}
main_path = MODELS_DIR / "model.joblib"
if main_path.exists():
    _artifacts = joblib.load(main_path)
    model = _artifacts["model"]
    feature_cols = _artifacts["feature_cols"]
    departments = sorted(_artifacts.get("departments", []))
    property_types = sorted(_artifacts.get("property_types", []))
    metrics = _artifacts["metrics"]
    print(f"  Loaded main model: {metrics}")

_land_artifacts = None
_land_model = None
_land_feature_cols = None
_land_departments = None
_land_metrics = None
land_path = MODELS_DIR / "model_land.joblib"
if land_path.exists():
    _land_artifacts = joblib.load(land_path)
    _land_model = _land_artifacts["model"]
    _land_feature_cols = _land_artifacts["feature_cols"]
    _land_departments = sorted(_land_artifacts.get("departments", []))
    _land_metrics = _land_artifacts["metrics"]
    print(f"  Loaded Land model: {_land_metrics}")

_apt_artifacts = None
_apt_model = None
_apt_feature_cols = None
_apt_departments = None
_apt_metrics = None
apt_path = MODELS_DIR / "model_apartment.joblib"
if apt_path.exists():
    _apt_artifacts = joblib.load(apt_path)
    _apt_model = _apt_artifacts["model"]
    _apt_feature_cols = _apt_artifacts["feature_cols"]
    _apt_departments = sorted(_apt_artifacts.get("departments", []))
    _apt_metrics = _apt_artifacts["metrics"]
    print(f"  Loaded Apartment model: {_apt_metrics}")

# Load socioeconomic data with GeoDataFrame spatial index
INSEE_PATH = Path(__file__).resolve().parent.parent / "data" / "insee" / "insee_communes.parquet"
_insee_gdf = None
if INSEE_PATH.exists():
    df = pd.read_parquet(INSEE_PATH)
    geometry = [Point(x, y) for x, y in zip(df["lon"], df["lat"])]
    _insee_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    _insee_gdf = _insee_gdf.set_geometry("geometry")
    _insee_gdf.sindex  # build spatial index
    print(f"  Loaded INSEE data: {len(_insee_gdf)} communes (GeoDataFrame)")

# Load construction benchmarks
try:
    from construction_benchmarks import get_all_features as get_construction
except ImportError:
    from src.construction_benchmarks import get_all_features as get_construction
_construction_costs = get_construction().set_index("department")["construction_cost_m2"].to_dict()

# Optional nearby-comparable index from cleaned DVF transactions. New models use these
# columns; old saved models ignore them because their feature list does not include them.
DVF_COMPARABLES_PATH = Path(__file__).resolve().parent.parent / "data" / "dvf" / "cleaned.parquet"
_comparable_builder = None
_comparable_time_metadata = {}
if DVF_COMPARABLES_PATH.exists():
    try:
        _comp_raw = pd.read_parquet(DVF_COMPARABLES_PATH)
        _comp_df = from_dvf(_comp_raw) if "valeur_fonciere" in _comp_raw.columns else _comp_raw
        _comparable_time_metadata = get_time_metadata(_comp_df)
        _comparable_builder = ComparableFeatureBuilder(_comp_df)
        print(f"  Loaded comparable DVF index: {sum(len(g['price']) for g in _comparable_builder.groups.values()):,} transactions")
        del _comp_raw, _comp_df
    except Exception as exc:
        print(f"  WARNING: comparable DVF index unavailable ({exc})")

# Socioeconomic feature columns used by the model
SOCIOECON_COLS = [c for c in feature_cols if c in (
    "log_population", "pop_density", "poverty_rate", "log_income", "construction_cost_m2"
)]

OSM_API = "https://api.openstreetmap.org/api/0.6"
OVERPASS_API = "https://overpass-api.de/api/interpreter"

# SQLAlchemy SQLite cache for nearby places
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_db_engine = create_engine(f"sqlite:///{CACHE_DIR / 'nearby_cache.db'}", connect_args={"check_same_thread": False})
with _db_engine.connect() as conn:
    conn.execute(sql_text("""CREATE TABLE IF NOT EXISTS nearby_cache (
        bbox TEXT PRIMARY KEY, data TEXT, cached_at REAL
    )"""))
    conn.commit()
_NEARBY_CACHE_TTL = 86400  # 24 hours

app = FastAPI(title="AlfaScript")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class NearbyPlace(BaseModel):
    name: str
    type: str
    lat: float
    lon: float
    distance_m: float


def _find_nearest_commune(lat: float, lon: float) -> dict:
    """Find nearest commune INSEE data using GeoDataFrame spatial index."""
    if _insee_gdf is None:
        return {}
    point = Point(lon, lat)
    result = _insee_gdf.sindex.nearest(point, return_all=False)
    if isinstance(result, tuple):
        idx = int(result[1][0])
    else:
        idx = int(result[1][0])
    row = _insee_gdf.iloc[idx]
    return {
        "log_population": row.get("log_population", 0),
        "pop_density": row.get("pop_density", 0),
        "poverty_rate": row.get("poverty_rate", 0),
        "log_income": row.get("log_income", 0),
        "commune_code": row.get("commune_code", ""),
    }


def _classify_poi(tags: dict) -> str | None:
    """Classify OSM tags into a category, or None if not a POI."""
    a = (tags.get("amenity") or "").lower()
    s = (tags.get("shop") or "").lower()
    l = (tags.get("leisure") or "").lower()
    r = (tags.get("railway") or "").lower()
    pt = (tags.get("public_transport") or "").lower()
    hw = (tags.get("highway") or "").lower()

    if r in ("station", "halt", "tram_stop", "subway_entrance") or pt == "station" or a in ("bus_station", "ferry_terminal") or hw == "bus_stop": return "transit"
    if a in ("school", "kindergarten", "university", "college"): return "school"
    if a in ("hospital", "clinic", "doctors"): return "hospital"
    if a == "pharmacy": return "pharmacy"
    if a in ("restaurant", "fast_food"): return "restaurant"
    if a in ("cafe", "pub", "bar"): return "cafe"
    if a in ("bank", "bureau_de_change"): return "bank"
    if a == "parking": return "parking"
    if a == "fuel": return "fuel"
    if a in ("place_of_worship", "church"): return "religious"
    if a == "post_office": return "post"
    if a == "library": return "library"
    if a == "police": return "police"
    if a in ("theatre", "cinema", "arts_centre"): return "entertainment"
    if a in ("townhall", "community_centre"): return "civic"
    if l in ("park", "garden", "playground", "pitch"): return "park"
    if s: return "shop"
    return None


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _bbox_around(lat, lon, radius_km):
    """Return (min_lon, min_lat, max_lon, max_lat) for a bounding box."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * cos(radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _osm_to_places(osm_xml: str, center_lat: float, center_lon: float, radius: int) -> list:
    """Parse OSM API XML response and extract named POIs (nodes + ways)."""
    root = ET.fromstring(osm_xml)
    places = []
    seen = set()
    radius_km = radius / 1000.0

    # First pass: collect node coordinates for way resolution
    node_coords = {}
    poi_nodes = []  # (lat, lon, tags, name)
    poi_way_nds = []  # list of (nd_refs, tags, name)

    for el in root:
        tag = el.tag
        if tag == "node":
            nid = el.get("id")
            lat = el.get("lat")
            lon = el.get("lon")
            if nid and lat and lon:
                node_coords[nid] = (float(lat), float(lon))

            tags = {}
            for child in el:
                if child.tag == "tag":
                    tags[child.get("k", "")] = child.get("v", "")
            name = tags.get("name", "")
            if name and _classify_poi(tags):
                poi_nodes.append((float(lat), float(lon), tags, name))

        elif tag == "way":
            tags = {}
            nd_refs = []
            for child in el:
                if child.tag == "tag":
                    tags[child.get("k", "")] = child.get("v", "")
                elif child.tag == "nd":
                    nd_refs.append(child.get("ref", ""))
            name = tags.get("name", "")
            if name and _classify_poi(tags) and nd_refs:
                poi_way_nds.append((nd_refs, tags, name))

    # Process nodes
    for lat, lon, tags, name in poi_nodes:
        ptype = _classify_poi(tags)
        if not ptype:
            continue
        dist = round(_haversine_km(center_lat, center_lon, lat, lon) * 1000)
        if dist > radius:
            continue
        key = f"{name}_{ptype}"
        if key in seen:
            continue
        seen.add(key)
        places.append({"name": name[:60], "type": ptype, "lat": lat, "lon": lon, "distance_m": dist})

    # Process ways: compute centroid from child node coords
    for nd_refs, tags, name in poi_way_nds:
        ptype = _classify_poi(tags)
        if not ptype:
            continue
        lats, lons = [], []
        for ref in nd_refs:
            if ref in node_coords:
                lats.append(node_coords[ref][0])
                lons.append(node_coords[ref][1])
        if not lats:
            continue
        lat = sum(lats) / len(lats)
        lon = sum(lons) / len(lons)
        dist = round(_haversine_km(center_lat, center_lon, lat, lon) * 1000)
        if dist > radius:
            continue
        key = f"{name}_{ptype}"
        if key in seen:
            continue
        seen.add(key)
        places.append({"name": name[:60], "type": ptype, "lat": lat, "lon": lon, "distance_m": dist})

    places.sort(key=lambda p: p["distance_m"])
    return places[:100]


@app.get("/nearby")
async def nearby(lat: float, lon: float, radius: int = 500):
    try:
        min_lon, min_lat, max_lon, max_lat = _bbox_around(lat, lon, (radius + 50) / 1000.0)
        min_lat = max(min_lat, -90.0); max_lat = min(max_lat, 90.0)
        min_lon = max(min_lon, -180.0); max_lon = min(max_lon, 180.0)
        bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

        now = time.time()
        # Check SQLite cache
        with _db_engine.connect() as conn:
            row = conn.execute(sql_text("SELECT data, cached_at FROM nearby_cache WHERE bbox = :b"), {"b": bbox_str}).fetchone()
        if row and (now - row[1]) < _NEARBY_CACHE_TTL:
            import json
            return {"places": json.loads(row[0])}

        places = []

        # Try OSM API first
        headers = {"User-Agent": "AlfaScript/1.0"}
        async with httpx.AsyncClient(timeout=20) as client:
            for attempt in range(3):
                try:
                    resp = await client.get(f"{OSM_API}/map?bbox={bbox_str}", headers=headers)
                    if resp.status_code == 200:
                        places = _osm_to_places(resp.text, lat, lon, radius)
                        break
                    elif resp.status_code == 509:
                        await asyncio.sleep(3 * (attempt + 1))
                        continue
                    else:
                        break
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(2)
                    continue

        # Fallback: Overpass if OSM API returned nothing
        if not places:
            try:
                op_query = f'[out:json][timeout:30];(node["amenity"](around:{radius},{lat},{lon});node["shop"](around:{radius},{lat},{lon});node["leisure"](around:{radius},{lat},{lon});node["railway"="station"](around:{radius},{lat},{lon});node["public_transport"="station"](around:{radius},{lat},{lon});way["amenity"](around:{radius},{lat},{lon});way["shop"](around:{radius},{lat},{lon});way["leisure"](around:{radius},{lat},{lon});way["railway"="station"](around:{radius},{lat},{lon});way["public_transport"="station"](around:{radius},{lat},{lon}););out center tags(30);'
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(OVERPASS_API, data={"data": op_query}, headers=headers)
                    if resp.status_code == 200:
                        data = resp.json()
                        for el in data.get("elements", []):
                            tags = el.get("tags", {})
                            name = tags.get("name", "")
                            if not name:
                                continue
                            ptype = _classify_poi(tags)
                            if not ptype:
                                continue
                            el_lat = el.get("lat") or el.get("center", {}).get("lat")
                            el_lon = el.get("lon") or el.get("center", {}).get("lon")
                            if not el_lat or not el_lon:
                                continue
                            dist = round(_haversine_km(lat, lon, el_lat, el_lon) * 1000)
                            if dist > radius:
                                continue
                            key = f"{name}_{ptype}"
                            if key in {p["name"] + "_" + p["type"] for p in places}:
                                continue
                            places.append({"name": name[:60], "type": ptype, "lat": el_lat, "lon": el_lon, "distance_m": dist})
            except Exception:
                pass

        places.sort(key=lambda p: p["distance_m"])
        places = places[:100]

        # Store in SQLite cache
        import json
        with _db_engine.connect() as conn:
            conn.execute(sql_text("""INSERT OR REPLACE INTO nearby_cache (bbox, data, cached_at) VALUES (:b, :d, :t)"""),
                         {"b": bbox_str, "d": json.dumps(places), "t": now})
            conn.commit()

        return {"places": places}

    except Exception as e:
        return {"places": [], "error": str(e)}


TILE_CACHE = Path(__file__).resolve().parent.parent / "data" / "tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "AlfaScript/1.0"


async def _get_tile(z: int, x: int, y: int) -> bytes:
    cache_path = TILE_CACHE / f"{z}_{x}_{y}.png"
    if cache_path.exists():
        if time.time() - cache_path.stat().st_mtime < 86400:
            return cache_path.read_bytes()
    url = TILE_URL.format(z=z, x=x, y=y)
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.content
            cache_path.write_bytes(data)
            return data
    return None


@app.get("/staticmap")
async def staticmap(
    lat: float = Query(...), lon: float = Query(...),
    zoom: int = 15, width: int = 600, height: int = 350,
):
    tiles_per_row = math.ceil(width / 256) + 1
    tiles_per_col = math.ceil(height / 256) + 1

    tile_center = mercantile.tile(lon, lat, zoom)
    cx, cy = tile_center.x, tile_center.y

    # fractional offset within the center tile
    bounds = mercantile.bounds(tile_center)
    tile_lon_span = bounds.east - bounds.west
    tile_lat_span = bounds.north - bounds.south
    frac_x = (lon - bounds.west) / tile_lon_span
    frac_y = (bounds.north - lat) / tile_lat_span

    # pixel offset of center within image
    center_px_x = int(frac_x * 256)
    center_px_y = int(frac_y * 256)

    start_col = cx - (center_px_x // 256) - (1 if center_px_x % 256 < width / 2 else 0)
    start_row = cy - (center_px_y // 256) - (1 if center_px_y % 256 < height / 2 else 0)

    img = Image.new("RGB", (width, height), (240, 240, 240))

    async def get_tile_safe(z, x, y):
        try:
            return await _get_tile(z, x, y)
        except Exception:
            return None

    for row_idx in range(tiles_per_col):
        for col_idx in range(tiles_per_row):
            tx = start_col + col_idx
            ty = start_row + row_idx
            tile_data = await _get_tile(zoom, tx, ty)
            if tile_data:
                tile_img = Image.open(io.BytesIO(tile_data))
                px = col_idx * 256 - (center_px_x - width // 2)
                py = row_idx * 256 - (center_px_y - height // 2)
                img.paste(tile_img, (px, py))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


class PredictionInput(BaseModel):
    area_sqm: float = Field(..., gt=0, description="Property area in m")
    rooms: int = Field(..., ge=1, le=50)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    department: str = Field(..., description="Department code (e.g. 75)")
    property_type: str = Field(default="apartment", description="apartment/house/commercial/other")


class PredictionOutput(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    confidence_low: float
    confidence_high: float
    confidence_range_formatted: str
    currency: str = "EUR"
    model_metrics: dict
    confidence_note: str


def _model_time_metadata(artifacts: dict | None) -> dict:
    return (artifacts or {}).get("time_metadata") or _comparable_time_metadata or {}


def _add_runtime_model_features(df: pd.DataFrame, artifacts: dict | None) -> pd.DataFrame:
    time_metadata = _model_time_metadata(artifacts)
    df = add_time_features(
        df,
        start_date=time_metadata.get("start_date"),
        as_of_date=time_metadata.get("max_date"),
    )
    if _comparable_builder is not None:
        df = _comparable_builder.transform(df, include_self=True)
    return df


def _confidence_range(value: float, model_metrics: dict | None) -> tuple[float, float, str]:
    mape = float((model_metrics or {}).get("mape_pct", 20.0)) / 100.0
    low = max(value * (1 - mape), 0)
    high = value * (1 + mape)
    return round(low, 2), round(high, 2), f"{low:,.0f} - {high:,.0f}"


def _aligned_features(
    df: pd.DataFrame,
    expected_features: list[str],
    dept: str,
    model_departments: list[str] | None = None,
    property_type: str | None = None,
    model_property_types: list[str] | None = None,
) -> pd.DataFrame:
    additions = {}
    for prop_type in model_property_types or []:
        additions[f"property_type_{prop_type}"] = int(property_type == prop_type)
    for department in model_departments or []:
        additions[f"department_{department}"] = int(dept == str(department).zfill(2))
    for col in expected_features:
        if col not in df.columns and col not in additions:
            additions[col] = 0
    if additions:
        df = pd.concat([df, pd.DataFrame(additions, index=df.index)], axis=1)
    return df[expected_features].fillna(0)


@app.get("/options")
def get_options():
    return {
        "departments": departments,
        "property_types": property_types,
        "models": {
            "original": {"type": "property", "metrics": metrics, "status": "active"},
            "land": {"type": "land", "metrics": _land_metrics, "status": "active" if _land_model else "unavailable"},
            "apartment": {"type": "apartment", "metrics": _apt_metrics, "status": "active" if _apt_model else "unavailable"},
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_metrics": metrics,
        "comparables_loaded": _comparable_builder is not None,
        "comparable_period": _comparable_time_metadata,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    dept = str(data.department).zfill(2)
    commune = _find_nearest_commune(data.latitude, data.longitude)
    construction_cost = _construction_costs.get(dept, 1800)

    row = {"area_sqm": data.area_sqm, "rooms": data.rooms,
           "latitude": data.latitude, "longitude": data.longitude,
           "department": dept, "property_type": data.property_type,
           "construction_cost_m2": construction_cost,
           "log_population": commune.get("log_population", 0),
           "pop_density": commune.get("pop_density", 0),
           "poverty_rate": commune.get("poverty_rate", 0),
           "log_income": commune.get("log_income", 0),
           "commune_code": commune.get("commune_code", ""),
    }

    df = _add_runtime_model_features(pd.DataFrame([row]), _artifacts)

    X = _aligned_features(
        df,
        feature_cols,
        dept,
        model_departments=departments,
        property_type=data.property_type,
        model_property_types=property_types,
    )
    pred_log = float(model.predict(X)[0])
    price = max(np.expm1(pred_log), 0)
    confidence_low, confidence_high, confidence_formatted = _confidence_range(price, metrics)

    return PredictionOutput(
        predicted_price=round(price, 2),
        predicted_price_formatted=f"{price:,.0f}",
        confidence_low=confidence_low,
        confidence_high=confidence_high,
        confidence_range_formatted=confidence_formatted,
        model_metrics=metrics,
        confidence_note="Prediction is an estimate based on historical data. "
                         "Actual market prices may vary significantly.",
    )


class LandPredictionInput(BaseModel):
    land_sqm: float = Field(..., gt=0, description="Plot area in m²")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    department: str = Field(..., description="Department code (e.g. 75)")
    zone_type: str = Field(default="urban", description="urban/periurban/rural")


class ApartmentPredictionInput(BaseModel):
    area_sqm: float = Field(..., gt=0, description="Living area in m²")
    rooms: int = Field(..., ge=1, le=50)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    department: str = Field(..., description="Department code (e.g. 75)")


@app.post("/predict/land")
def predict_land(data: LandPredictionInput):
    if _land_model is None:
        return {"error": "Land model not available"}
    dept = str(data.department).zfill(2)
    commune = _find_nearest_commune(data.latitude, data.longitude)
    construction_cost = _construction_costs.get(dept, 1800)

    # Model trained on houses; for empty plot use minimal building values
    row = {
        "area_sqm": 1, "rooms": 0, "land_sqm": data.land_sqm,
        "latitude": data.latitude, "longitude": data.longitude,
        "department": dept, "property_type": "house",
        "construction_cost_m2": construction_cost,
        "log_population": commune.get("log_population", 0),
        "pop_density": commune.get("pop_density", 0),
        "poverty_rate": commune.get("poverty_rate", 0),
        "log_income": commune.get("log_income", 0),
    }

    df = _add_runtime_model_features(pd.DataFrame([row]), _land_artifacts)
    X = _aligned_features(df, _land_feature_cols, dept, model_departments=_land_departments)
    pred_log = float(_land_model.predict(X)[0])
    price = max(np.expm1(pred_log), 0)
    confidence_low, confidence_high, confidence_formatted = _confidence_range(price, _land_metrics)

    price_per_land_m2 = price / data.land_sqm if data.land_sqm > 0 else 0

    return {
        "predicted_price": round(price, 2),
        "predicted_price_formatted": f"{price:,.0f}",
        "confidence_low": confidence_low,
        "confidence_high": confidence_high,
        "confidence_range_formatted": confidence_formatted,
        "price_per_land_m2": round(price_per_land_m2, 2),
        "price_per_land_m2_formatted": f"{price_per_land_m2:,.0f}",
        "model_metrics": _land_metrics,
        "department": dept,
        "zone_type": data.zone_type,
        "confidence_note": "Land value estimate based on comparable house transactions. "
                           "Actual plot values may vary significantly with zoning and local market conditions.",
    }


@app.post("/predict/apartment")
def predict_apartment(data: ApartmentPredictionInput):
    if _apt_model is None:
        return {"error": "Apartment model not available"}
    dept = str(data.department).zfill(2)
    commune = _find_nearest_commune(data.latitude, data.longitude)
    construction_cost = _construction_costs.get(dept, 1800)

    row = {
        "area_sqm": data.area_sqm, "rooms": data.rooms,
        "latitude": data.latitude, "longitude": data.longitude,
        "department": dept, "property_type": "apartment",
        "construction_cost_m2": construction_cost,
        "log_population": commune.get("log_population", 0),
        "pop_density": commune.get("pop_density", 0),
        "poverty_rate": commune.get("poverty_rate", 0),
        "log_income": commune.get("log_income", 0),
    }

    df = _add_runtime_model_features(pd.DataFrame([row]), _apt_artifacts)
    X = _aligned_features(df, _apt_feature_cols, dept, model_departments=_apt_departments)
    pred_log = float(_apt_model.predict(X)[0])
    price_per_sqm = max(np.expm1(pred_log), 0)
    total_price = price_per_sqm * data.area_sqm
    total_low, total_high, total_confidence = _confidence_range(total_price, _apt_metrics)
    psqm_low, psqm_high, psqm_confidence = _confidence_range(price_per_sqm, _apt_metrics)

    return {
        "predicted_price_per_sqm": round(price_per_sqm, 2),
        "predicted_price_per_sqm_formatted": f"{price_per_sqm:,.0f}",
        "price_per_sqm_confidence_low": psqm_low,
        "price_per_sqm_confidence_high": psqm_high,
        "price_per_sqm_confidence_range_formatted": psqm_confidence,
        "predicted_total_price": round(total_price, 2),
        "predicted_total_price_formatted": f"{total_price:,.0f}",
        "confidence_low": total_low,
        "confidence_high": total_high,
        "confidence_range_formatted": total_confidence,
        "model_metrics": _apt_metrics,
        "department": dept,
        "confidence_note": "Apartment price per m² estimate based on comparable transactions. "
                           "Floor level, standing, and year built are not available in DVF data.",
    }


class FinancialInput(BaseModel):
    purchase_price: float = Field(..., gt=0)
    down_payment_pct: float = Field(default=20, ge=0, le=100)
    loan_rate: float = Field(default=3.5, ge=0, le=30)
    loan_term_years: int = Field(default=20, ge=1, le=40)
    monthly_rent: float = Field(..., gt=0)
    monthly_expenses: float = Field(default=0, ge=0)
    annual_appreciation: float = Field(default=2.0, ge=-10, le=30)


class ParcelLookupInput(BaseModel):
    reference: str = Field(..., min_length=4, description="Cadastre reference e.g. 75056AB0045")


import re

def _parse_cadastre_ref(ref: str) -> dict | None:
    cleaned = re.sub(r'\s+', '', ref.strip()).upper()
    m = re.match(r'(\d{5})([A-Z]{1,2})(\d{1,4})$', cleaned)
    if m:
        return {
            "code_insee": m.group(1),
            "section": m.group(2).ljust(2)[:2],
            "numero": m.group(3).zfill(4),
        }
    m2 = re.match(r'(\d{2})\s*(\d{3})\s*([A-Z]{1,2})\s*(\d{1,4})$', ref.strip().upper())
    if m2:
        return {
            "code_insee": m2.group(1) + m2.group(2),
            "section": m2.group(3).ljust(2)[:2],
            "numero": m2.group(4).zfill(4),
        }
    return None


CADASTRE_API = "https://apicarto.ign.fr/api/cadastre/parcelle"


@app.post("/lookup-parcel")
async def lookup_parcel(data: ParcelLookupInput):
    import httpx
    parsed = _parse_cadastre_ref(data.reference)
    if not parsed:
        return {"error": "Format non reconnu. Attendu: 75056AB0045 ou 75 056 AB 45"}
    params = {k: v for k, v in parsed.items() if v}
    async with httpx.AsyncClient() as client:
        resp = await client.get(CADASTRE_API, params=params, headers={"accept": "application/json"}, timeout=15)
    if resp.status_code != 200:
        return {"error": f"API cadastre erreur {resp.status_code}"}
    body = resp.json()
    features = body.get("features", [])
    if not features:
        return {"error": "Aucune parcelle trouvée"}
    feat = features[0]
    props = feat.get("properties", {})
    geom = feat.get("geometry", {})
    centroid = {"lat": 48.8566, "lon": 2.3522}
    if geom and geom.get("type") in ("Polygon", "MultiPolygon"):
        from shapely.geometry import shape as shapely_shape
        shp = shapely_shape(geom)
        centroid = {"lat": shp.centroid.y, "lon": shp.centroid.x}
    area = props.get("contenance") or props.get("ssurf") or props.get("supf") or 0
    return {
        "code_insee": props.get("code_insee", ""),
        "commune": props.get("nom_com", ""),
        "section": props.get("section", ""),
        "numero": props.get("numero", ""),
        "area": float(area) if area else 0,
        "centroid": centroid,
    }


@app.post("/financial")
def financial_analysis(data: FinancialInput):
    dp = data.purchase_price * data.down_payment_pct / 100.0
    loan = data.purchase_price - dp
    monthly_rate = (data.loan_rate / 100.0) / 12.0
    n_payments = data.loan_term_years * 12

    if monthly_rate > 0:
        monthly_pmt = loan * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
    else:
        monthly_pmt = loan / n_payments

    monthly_cf = data.monthly_rent - monthly_pmt - data.monthly_expenses
    annual_cf = monthly_cf * 12
    coc_roi = (annual_cf / dp * 100) if dp > 0 else 0

    def compute_irr(years):
        cf = [-dp]
        for y in range(1, years + 1):
            cf.append(annual_cf)
        # Add sale proceeds at end of horizon
        sale_price = data.purchase_price * ((1 + data.annual_appreciation / 100.0) ** years)
        remaining_balance = 0
        if monthly_rate > 0:
            remaining = n_payments - years * 12
            if remaining > 0:
                remaining_balance = loan * ((1 + monthly_rate) ** remaining - 1) / ((1 + monthly_rate) ** n_payments - 1) * (1 + monthly_rate) ** remaining
            else:
                remaining_balance = 0
        else:
            remaining_balance = max(0, loan - (years * 12) * monthly_pmt)
        net_proceeds = sale_price - remaining_balance
        cf[-1] += net_proceeds
        try:
            return float(npf.irr(cf)) * 100
        except Exception:
            return None

    closing_costs = data.purchase_price * 0.08
    total_investment = dp + closing_costs
    irr_5 = compute_irr(5)
    irr_10 = compute_irr(10)
    irr_15 = compute_irr(15)
    irr_20 = compute_irr(20)

    return {
        "down_payment": round(dp, 2),
        "down_payment_formatted": f"{dp:,.0f}",
        "loan_amount": round(loan, 2),
        "loan_amount_formatted": f"{loan:,.0f}",
        "monthly_payment": round(monthly_pmt, 2),
        "monthly_payment_formatted": f"{monthly_pmt:,.0f}",
        "monthly_cash_flow": round(monthly_cf, 2),
        "monthly_cash_flow_formatted": f"{monthly_cf:,.0f}",
        "annual_cash_flow": round(annual_cf, 2),
        "annual_cash_flow_formatted": f"{annual_cf:,.0f}",
        "cash_on_cash_roi": round(coc_roi, 2),
        "closing_costs": round(closing_costs, 2),
        "closing_costs_formatted": f"{closing_costs:,.0f}",
        "total_investment": round(total_investment, 2),
        "total_investment_formatted": f"{total_investment:,.0f}",
        "irr_5y": round(irr_5, 2) if irr_5 is not None else None,
        "irr_10y": round(irr_10, 2) if irr_10 is not None else None,
        "irr_15y": round(irr_15, 2) if irr_15 is not None else None,
        "irr_20y": round(irr_20, 2) if irr_20 is not None else None,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host="127.0.0.1", port=8001, reload=False)
