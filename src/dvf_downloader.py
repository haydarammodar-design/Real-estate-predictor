"""Download & clean French DVF real estate data.

Usage:
  python src/dvf_downloader.py                  # Download all departments (slow)
  python src/dvf_downloader.py --departments 75 # Just Paris
  python src/dvf_downloader.py --departments 75 92 93 94  # Paris + petite couronne
  python src/dvf_downloader.py --years 2021 2022 2023 2024 2025  # Multi-year dataset
  python src/dvf_downloader.py --raw ./data/dvf_raw/valeursfoncieres.txt  # Process your own raw txt file
"""
import argparse
import csv
import io
import os
import sys
import ssl
import zipfile
from pathlib import Path

import certifi
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Fix SSL for macOS — use certifi's trusted CA bundle
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

REPO = "https://files.data.gouv.fr/geo-dvf/latest/csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dvf"

METROPOLITAN_DEPARTMENTS = (
    [f"{i:02d}" for i in range(1, 20)]
    + ["2A", "2B"]
    + [f"{i:02d}" for i in range(21, 96)]
)


def get_latest_year() -> str:
    """Find the most recent year available in the geo-dvf repository."""
    try:
        resp = requests.get(f"{REPO}/", timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        years = []
        for link in soup.find_all("a"):
            href = link.get("href", "")
            if href.isdigit() and 2018 <= int(href) <= 2030:
                years.append(href)
        if years:
            return max(years)
    except Exception as e:
        print(f"  Could not detect latest year: {e}")
    return "2025"


def _normalize_department(dept: str) -> str:
    dept = str(dept).strip().upper()
    if dept in ("2A", "2B"):
        return dept
    return dept.zfill(2)


def _resolve_years(args) -> list[str]:
    if args.years:
        return sorted({str(year) for year in args.years})
    if args.year_start or args.year_end:
        latest = int(get_latest_year())
        start = int(args.year_start or args.year_end or latest)
        end = int(args.year_end or args.year_start or latest)
        if start > end:
            start, end = end, start
        return [str(year) for year in range(start, end + 1)]
    return [get_latest_year()]


def download_departments(departments: list[str], years: list[str]) -> pd.DataFrame:
    all_dfs = []
    for year in years:
        print(f"Downloading year {year}...")
        for dept in departments:
            df = download_department(dept, year)
            if not df.empty:
                cols = [c for c in COLUMNS_KEEP if c in df.columns]
                all_dfs.append(df[cols])
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

COLUMNS_KEEP = [
    "id_mutation",
    "date_mutation",
    "valeur_fonciere",
    "adresse_numero",
    "adresse_nom_voie",
    "code_postal",
    "code_commune",
    "nom_commune",
    "code_departement",
    "id_parcelle",
    "type_local",
    "surface_reelle_bati",
    "nombre_pieces_principales",
    "surface_terrain",
    "longitude",
    "latitude",
]

TYPE_LOCAL_MAP = {
    "Appartement": "apartment",
    "Maison": "house",
    "Local industriel. commercial ou assimilé": "commercial",
    "Dépendance": "other",
    "Local d'usage mixte (artisanal. commercial. bureau)": "commercial",
}


def download_department(dept_code: str, year: str = None) -> pd.DataFrame:
    """Download one department CSV and return a cleaned DataFrame."""
    if year is None:
        year = get_latest_year()
    url = f"{REPO}/{year}/departements/{dept_code}.csv.gz"
    print(f"  Downloading {url} ...")
    try:
        df = pd.read_csv(url, dtype={"code_postal": str, "code_commune": str}, low_memory=False)
    except Exception as e:
        print(f"  FAILED ({e})")
        return pd.DataFrame()

    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.drop_duplicates(subset=["id_mutation"])

    if "valeur_fonciere" in df.columns:
        df = df[df["valeur_fonciere"].notna()]
        df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"], errors="coerce")
        df = df[df["valeur_fonciere"] > 10000]

    if "surface_reelle_bati" in df.columns:
        df["surface_reelle_bati"] = pd.to_numeric(df["surface_reelle_bati"], errors="coerce")
        df = df[df["surface_reelle_bati"].notna()]
        df = df[(df["surface_reelle_bati"] >= 9) & (df["surface_reelle_bati"] <= 10000)]

    if "surface_terrain" in df.columns:
        df["surface_terrain"] = pd.to_numeric(df["surface_terrain"], errors="coerce").fillna(0)

    if "nombre_pieces_principales" in df.columns:
        df["nombre_pieces_principales"] = (
            pd.to_numeric(df["nombre_pieces_principales"], errors="coerce").fillna(0).astype(int)
        )
        df = df[df["nombre_pieces_principales"] > 0]

    if "type_local" in df.columns:
        df = df[df["type_local"].isin(TYPE_LOCAL_MAP.keys())]

    df["longitude"] = pd.to_numeric(df.get("longitude", pd.NA), errors="coerce")
    df["latitude"] = pd.to_numeric(df.get("latitude", pd.NA), errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"])
    # This downloader targets metropolitan departments; remove obvious geocoding errors.
    df = df[df["latitude"].between(41, 52) & df["longitude"].between(-6, 10)]

    after = len(df)
    print(f"  Cleaned: {before} -> {after} rows ({before-after} removed)")
    return df


def process_raw_txt(txt_path: Path, chunksize: int = 50000) -> pd.DataFrame:
    """Process raw DGFiP pipe-delimited txt files (for users who already have them)."""
    RAW_COLUMNS = [
        "id_service", "ref_doc", "1er_cgi", "2e_cgi", "3e_cgi", "4e_cgi", "5e_cgi",
        "no_disposition", "date_mutation", "nature_mutation", "valeur_fonciere",
        "adresse_numero", "adresse_suffixe", "adresse_code_voie", "adresse_nom_voie",
        "code_postal", "commune", "code_departement", "code_commune",
        "prefixe_section", "section", "no_plan", "no_volume",
        "lot1_no", "lot1_surface", "lot2_no", "lot2_surface",
        "lot3_no", "lot3_surface", "lot4_no", "lot4_surface",
        "lot5_no", "lot5_surface", "nombre_lots",
        "code_type_local", "type_local", "id_local",
        "surface_reelle_bati", "nombre_pieces_principales",
        "code_nature_culture", "nature_culture",
        "surface_terrain",
    ]

    chunks = []
    reader = pd.read_csv(
        txt_path,
        sep="|",
        encoding="utf-8",
        names=RAW_COLUMNS,
        dtype=str,
        on_bad_lines="skip",
        chunksize=chunksize,
        low_memory=False,
    )
    for i, chunk in enumerate(reader):
        print(f"  Chunk {i}: {len(chunk)} rows")
        chunk = _clean(chunk)
        if not chunk.empty:
            cols = [c for c in COLUMNS_KEEP if c in chunk.columns]
            chunks.append(chunk[cols])
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download & clean DVF data")
    parser.add_argument("--departments", nargs="+", default=None,
                        help="Department codes (e.g. 75 92 93). Default: all 96")
    parser.add_argument("--years", nargs="+", default=None,
                        help="One or more DVF years, e.g. --years 2021 2022 2023 2024 2025")
    parser.add_argument("--year-start", default=None,
                        help="First DVF year to download, inclusive")
    parser.add_argument("--year-end", default=None,
                        help="Last DVF year to download, inclusive")
    parser.add_argument("--output", default=str(DATA_DIR / "cleaned.parquet"),
                        help="Output file path")
    parser.add_argument("--raw", default=None,
                        help="Path to raw DVF txt file (pipe-delimited)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet",
                        help="Output format (parquet is smaller & faster)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.raw:
        print(f"Processing raw txt: {args.raw}")
        df = process_raw_txt(Path(args.raw))
    elif args.departments:
        years = _resolve_years(args)
        departments = [_normalize_department(dept) for dept in args.departments]
        print(f"Downloading departments {departments} for years {years}...")
        df = download_departments(departments, years)
    else:
        years = _resolve_years(args)
        print(f"Downloading all departments for years {years} (this will take a while)...")
        df = download_departments(list(METROPOLITAN_DEPARTMENTS), years)

    if df.empty:
        print("No data retrieved. Check your department codes or internet connection.")
        sys.exit(1)

    df = df.drop_duplicates()
    df = df.sort_values("date_mutation").reset_index(drop=True)

    string_cols = [
        "id_mutation", "date_mutation", "adresse_nom_voie", "code_postal",
        "code_commune", "nom_commune", "code_departement", "id_parcelle",
        "type_local",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    print(f"\nFinal dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Period: {df['date_mutation'].min()} -> {df['date_mutation'].max()}")
    print(f"  Price range: {df['valeur_fonciere'].min():,.0f} - {df['valeur_fonciere'].max():,.0f} EUR")
    print(f"  Property types: {df['type_local'].value_counts().to_dict()}")
    print(f"  Departments: {df['code_departement'].nunique()}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path.with_suffix(".csv"), index=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
