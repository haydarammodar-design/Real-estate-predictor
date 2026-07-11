"""Construction cost benchmarks per m² by department and property standing.

BATIPRIX is the official French reference (proprietary, paid).
This module uses freely available industry benchmarks as fallback.
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = DATA_DIR / "construction_benchmarks.parquet"

# Base construction cost (€/m²) by standing level
# Sources: FFB (Fédération Française du Bâtiment), academic papers,
#          "Prix de la construction au m² en France" bulletins
STANDING_COST = {
    "low":    1200,   # social housing, basic finish
    "medium": 1800,   # standard residential
    "high":   2800,   # good finish, architect
    "luxury": 4000,   # premium materials, custom
}

# Regional adjustment factors (by department)
# Île-de-France is ~25-40% above national average
# Mediterranean coast is ~15-25% above
# Rural areas are ~10-20% below
REGIONAL_FACTORS = {
    # Île-de-France — highest
    "75": 1.40, "77": 1.15, "78": 1.20, "91": 1.15, "92": 1.35,
    "93": 1.15, "94": 1.25, "95": 1.15,
    # Provence-Alpes-Côte d'Azur — high
    "04": 1.10, "05": 1.10, "06": 1.30, "13": 1.20, "83": 1.20, "84": 1.10,
    # Auvergne-Rhône-Alpes (Lyon, Grenoble)
    "01": 1.05, "07": 1.00, "26": 1.00, "38": 1.10, "42": 1.00,
    "63": 0.95, "69": 1.20, "73": 1.10, "74": 1.15,
    # Occitanie (Toulouse, Montpellier)
    "09": 1.00, "11": 1.00, "12": 0.95, "30": 1.05, "31": 1.10,
    "32": 0.95, "34": 1.15, "46": 0.95, "48": 0.95, "65": 0.95,
    "66": 1.10, "81": 0.95, "82": 0.95,
    # Nouvelle-Aquitaine (Bordeaux)
    "16": 1.00, "17": 1.05, "19": 0.95, "23": 0.90, "24": 0.95,
    "33": 1.15, "40": 1.00, "47": 0.95, "64": 1.05, "79": 0.95,
    "86": 0.95, "87": 0.95,
    # Hauts-de-France
    "02": 0.90, "59": 1.05, "60": 0.95, "62": 0.95, "80": 0.95,
    # Normandie
    "14": 1.05, "27": 1.00, "50": 0.95, "61": 0.90, "76": 1.05,
    # Bretagne
    "22": 0.95, "29": 1.00, "35": 1.05, "56": 0.95,
    # Pays de la Loire
    "44": 1.10, "49": 1.00, "53": 0.95, "72": 0.95, "85": 1.00,
    # Centre-Val de Loire
    "18": 0.90, "28": 0.95, "36": 0.90, "37": 1.00, "41": 0.95, "45": 1.00,
    # Bourgogne-Franche-Comté
    "21": 1.00, "25": 1.00, "39": 0.95, "58": 0.90, "70": 0.95,
    "71": 0.95, "89": 0.95, "90": 0.95,
    # Grand Est
    "08": 0.95, "10": 0.95, "51": 1.00, "52": 0.90, "54": 1.00,
    "55": 0.90, "57": 1.00, "67": 1.05, "68": 1.05, "88": 0.95,
    # Corse
    "2A": 1.20, "2B": 1.20,
}


def build_dataset() -> pd.DataFrame:
    """Build construction cost benchmark table by department and standing."""
    rows = []
    for dept_code, factor in REGIONAL_FACTORS.items():
        for standing, base_cost in STANDING_COST.items():
            cost_m2 = round(base_cost * factor)
            rows.append({
                "department": dept_code,
                "standing": standing,
                "construction_cost_m2": cost_m2,
            })
    return pd.DataFrame(rows)


def get_cost_for_property(department: str, standing: str = "medium") -> int:
    """Get construction cost per m² for a given department and standing."""
    df = build_dataset()
    match = df[(df["department"] == department) & (df["standing"] == standing)]
    if not match.empty:
        return int(match.iloc[0]["construction_cost_m2"])
    # Fallback: use medium standing for this department
    match = df[(df["department"] == department) & (df["standing"] == "medium")]
    if not match.empty:
        return int(match.iloc[0]["construction_cost_m2"])
    return 1800  # national average fallback


def get_all_features() -> pd.DataFrame:
    """Return a DataFrame with construction cost for all departments (medium standing)."""
    df = build_dataset()
    return df[df["standing"] == "medium"].drop(columns=["standing"])


if __name__ == "__main__":
    df = build_dataset()
    print(f"Construction benchmarks: {len(df)} rows")
    print(df.head(10))
    print(f"\nParis (75) medium: {get_cost_for_property('75', 'medium')} €/m²")
    print(f"Paris (75) luxury:  {get_cost_for_property('75', 'luxury')} €/m²")
    print(f"Creuse (23) medium: {get_cost_for_property('23', 'medium')} €/m²")
