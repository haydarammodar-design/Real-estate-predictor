"""Collect OSM points-of-interest by commune for proximity features.

Computes count of nearby POIs (transit, schools, shops, parks, etc.)
within a radius from each commune center — used as model features.
"""
import math, asyncio, time
from pathlib import Path
import pandas as pd
import numpy as np
import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "osm_features"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OVERPASS_API = "https://overpass-api.de/api/interpreter"
USER_AGENT = "AlfaScript/1.0"

POI_QUERIES = {
    "metro_station": """node["railway"="station"]["station"="subway"](around:{radius},{lat},{lon});""",
    "train_station": """node["railway"="station"](around:{radius},{lat},{lon});""",
    "tram_stop": """node["railway"="tram_stop"](around:{radius},{lat},{lon});""",
    "bus_stop": """node["highway"="bus_stop"](around:{radius},{lat},{lon});""",
    "school": """node["amenity"="school"](around:{radius},{lat},{lon});""",
    "kindergarten": """node["amenity"="kindergarten"](around:{radius},{lat},{lon});""",
    "university": """node["amenity"="university"](around:{radius},{lat},{lon});""",
    "hospital": """node["amenity"="hospital"](around:{radius},{lat},{lon});""",
    "clinic": """node["amenity"="clinic"](around:{radius},{lat},{lon});""",
    "pharmacy": """node["amenity"="pharmacy"](around:{radius},{lat},{lon});""",
    "supermarket": """node["shop"="supermarket"](around:{radius},{lat},{lon});node["shop"="convenience"](around:{radius},{lat},{lon});""",
    "restaurant": """node["amenity"="restaurant"](around:{radius},{lat},{lon});""",
    "cafe": """node["amenity"="cafe"](around:{radius},{lat},{lon});""",
    "park": """node["leisure"="park"](around:{radius},{lat},{lon});way["leisure"="park"](around:{radius},{lat},{lon});""",
    "sports_pitch": """node["leisure"="pitch"](around:{radius},{lat},{lon});way["leisure"="pitch"](around:{radius},{lat},{lon});""",
    "library": """node["amenity"="library"](around:{radius},{lat},{lon});""",
    "bank": """node["amenity"="bank"](around:{radius},{lat},{lon});""",
    "parking": """node["amenity"="parking"](around:{radius},{lat},{lon});""",
}

# Group into broader categories for features
FEATURE_GROUPS = {
    "transit": ["metro_station", "train_station", "tram_stop", "bus_stop"],
    "education": ["school", "kindergarten", "university"],
    "healthcare": ["hospital", "clinic", "pharmacy"],
    "shopping": ["supermarket"],
    "dining": ["restaurant", "cafe"],
    "leisure": ["park", "sports_pitch", "library"],
    "services": ["bank", "parking"],
}

RADIUS = 1000  # meters


async def count_pois_for_commune(lat: float, lon: float) -> dict:
    """Count POIs by category within RADIUS of a location."""
    groups = {g: 0 for g in FEATURE_GROUPS}

    for poi_name, query_template in POI_QUERIES.items():
        query = f"""[out:json][timeout:15];({query_template});out count;"""
        q = query.format(radius=RADIUS, lat=lat, lon=lon)
        try:
            async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as c:
                resp = await c.post(OVERPASS_API, data={"data": q})
                if resp.status_code == 200:
                    data = resp.json()
                    count = data.get("elements", [{}])[0].get("tags", {}).get("total", 0) if data.get("elements") else 0
                    count = int(count) if count else 0
                    for group, members in FEATURE_GROUPS.items():
                        if poi_name in members:
                            groups[group] += count
        except Exception:
            pass

    return groups


async def build_commune_features(communes_df: pd.DataFrame) -> pd.DataFrame:
    """For each commune with valid center coords, compute OSM POI counts."""
    cache = DATA_DIR / "commune_poi_counts.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        # Return only communes present in input
        mask = df["commune_code"].isin(communes_df["commune_code"].values)
        return df[mask].copy()

    valid = communes_df.dropna(subset=["lat", "lon"]).copy()
    results = []

    for idx, row in valid.iterrows():
        counts = await count_pois_for_commune(row["lat"], row["lon"])
        counts["commune_code"] = row["commune_code"]
        results.append(counts)
        if (idx + 1) % 50 == 0:
            print(f"  OSM: processed {idx+1}/{len(valid)} communes")
        await asyncio.sleep(0.3)  # rate limit

    df = pd.DataFrame(results)
    df.to_parquet(cache, index=False)
    print(f"  OSM: cached POI counts for {len(df)} communes")
    return df


if __name__ == "__main__":
    from insee_collector import fetch_commune_basics
    import asyncio

    async def main():
        communes = await fetch_commune_basics()
        print(f"Building OSM features for {len(communes)} communes...")
        features = await build_commune_features(communes)
        print(features.describe())

    asyncio.run(main())
