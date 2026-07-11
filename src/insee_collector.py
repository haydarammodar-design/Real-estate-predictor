"""Collect INSEE socioeconomic data by commune for enriching DVF features.

Sources (all public, no API key required):
  - geo.api.gouv.fr — commune boundaries, population, surface area
  - insee.fr — FILoSoFi income/poverty data (CSV, free download)
"""
import io, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "insee"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GEO_API = "https://geo.api.gouv.fr/communes"
USER_AGENT = "AlfaScript/1.0"

# FILoSoFi measures we care about
MEASURES_OF_INTEREST = {
    "MED_SL": "median_income",
    "PR_MD60": "poverty_rate",
}


async def fetch_commune_basics() -> pd.DataFrame:
    """Fetch commune-level geography + population from geo.api.gouv.fr (free, no auth)."""
    cache = DATA_DIR / "communes_basics.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}) as c:
        resp = await c.get(f"{GEO_API}?fields=code,nom,population,surface,centre,departement&format=json")
        resp.raise_for_status()
        communes = resp.json()

    rows = []
    for com in communes:
        centre = com.get("centre", {}) or {}
        dep = com.get("departement", {}) or {}
        rows.append({
            "commune_code": com["code"],
            "commune_name": com.get("nom", ""),
            "population": com.get("population"),
            "surface_km2": com.get("surface"),
            "lat": centre.get("coordinates", [None, None])[1],
            "lon": centre.get("coordinates", [None, None])[0],
            "department": dep.get("code", com["code"][:2]),
        })

    df = pd.DataFrame(rows)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["surface_km2"] = pd.to_numeric(df["surface_km2"], errors="coerce")
    df.to_parquet(cache, index=False)
    print(f"  INSEE: cached {len(df):,} communes to {cache}")
    return df


async def fetch_income_data() -> pd.DataFrame:
    """Download FILoSoFi commune-level income data from INSEE (long-format CSV zip)."""
    cache = DATA_DIR / "income.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = "https://www.insee.fr/fr/statistiques/fichier/8984752/FILOSOFI_CC_csv.zip"
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as c:
            resp = await c.get(url)
            resp.raise_for_status()
    except Exception as e:
        print(f"    Failed to download INSEE income data: {e}")
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".csv") and "data" in n][0]
            raw = pd.read_csv(io.BytesIO(z.read(csv_name)), sep=";", dtype=str, low_memory=False)
    except Exception as e:
        print(f"    Failed to parse INSEE income data: {e}")
        return pd.DataFrame()

    # Filter for COM (communes only)
    com = raw[raw["GEO_OBJECT"] == "COM"].copy()
    if com.empty:
        print("    WARNING: no commune rows found in FILoSoFi data")
        return pd.DataFrame()

    # Keep only measures we need
    com = com[com["FILOSOFI_MEASURE"].isin(MEASURES_OF_INTEREST.keys())].copy()
    if com.empty:
        print("    WARNING: no matching income/poverty measures found")
        return pd.DataFrame()

    # Pivot: one row per commune, one column per measure
    com["OBS_VALUE"] = pd.to_numeric(com["OBS_VALUE"].str.replace(",", ".", regex=False), errors="coerce")
    com = com.dropna(subset=["OBS_VALUE"])
    com["measure_name"] = com["FILOSOFI_MEASURE"].map(MEASURES_OF_INTEREST)
    pivoted = com.pivot_table(
        index="GEO", columns="measure_name", values="OBS_VALUE", aggfunc="first"
    ).reset_index()
    pivoted = pivoted.rename(columns={"GEO": "commune_code"})
    pivoted["commune_code"] = pivoted["commune_code"].str.zfill(5)

    pivoted.to_parquet(cache, index=False)
    print(f"  INSEE: cached income data ({len(pivoted):,} communes)")
    return pivoted


async def fetch_all() -> pd.DataFrame:
    basics = await fetch_commune_basics()
    income = await fetch_income_data()

    if not income.empty:
        merged = basics.merge(income, on="commune_code", how="left")
    else:
        merged = basics.copy()
        for col in ["median_income", "poverty_rate"]:
            merged[col] = np.nan

    merged["median_income"] = merged["median_income"].fillna(merged["median_income"].median())
    merged["log_population"] = np.log1p(merged["population"].fillna(0))
    merged["log_income"] = np.log1p(merged["median_income"])
    merged["pop_density"] = merged["population"] / merged["surface_km2"].clip(lower=1)
    merged["pop_density"] = merged["pop_density"].fillna(merged["pop_density"].median())
    merged["poverty_rate"] = merged["poverty_rate"].fillna(merged["poverty_rate"].median())

    out = DATA_DIR / "insee_communes.parquet"
    merged.to_parquet(out, index=False)
    print(f"  INSEE: merged dataset saved ({len(merged):,} communes)")
    return merged


if __name__ == "__main__":
    import asyncio
    asyncio.run(fetch_all())
