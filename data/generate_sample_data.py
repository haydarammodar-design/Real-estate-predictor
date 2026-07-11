"""Generate sample real estate data for development and testing."""
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

DISTRICTS = {
    "Downtown":       {"lat_range": (30.04, 30.06), "lon_range": (31.23, 31.25), "base_price_per_sqm": 25000},
    "Zamalek":        {"lat_range": (30.06, 30.08), "lon_range": (31.21, 31.23), "base_price_per_sqm": 35000},
    "Maadi":          {"lat_range": (29.95, 29.98), "lon_range": (31.25, 31.28), "base_price_per_sqm": 18000},
    "New Cairo":      {"lat_range": (30.00, 30.03), "lon_range": (31.40, 31.50), "base_price_per_sqm": 15000},
    "Sheikh Zayed":   {"lat_range": (29.98, 30.01), "lon_range": (30.92, 31.00), "base_price_per_sqm": 16000},
    "Nasr City":      {"lat_range": (30.05, 30.08), "lon_range": (31.32, 31.36), "base_price_per_sqm": 12000},
    "Mohandeseen":    {"lat_range": (30.05, 30.07), "lon_range": (31.20, 31.22), "base_price_per_sqm": 20000},
    "Heliopolis":     {"lat_range": (30.08, 30.11), "lon_range": (31.33, 31.36), "base_price_per_sqm": 17000},
}

ROWS = 2000
records = []

for _ in range(ROWS):
    district = np.random.choice(list(DISTRICTS.keys()))
    info = DISTRICTS[district]

    area = round(np.random.uniform(60, 400), 1)
    rooms = int(np.clip(round(area / 40 + np.random.uniform(-1, 1)), 1, 8))
    bathrooms = int(np.clip(rooms - np.random.randint(0, 2), 1, 5))
    floor = np.random.randint(1, 20)
    property_age = max(0, int(np.random.exponential(15)))
    lat = round(np.random.uniform(*info["lat_range"]), 6)
    lon = round(np.random.uniform(*info["lon_range"]), 6)

    base_price = info["base_price_per_sqm"] * area
    age_factor = max(0.5, 1.0 - property_age * 0.015)
    room_factor = 1.0 + (rooms - 2) * 0.05
    floor_factor = 1.0 + (floor / 50.0)
    noise = np.random.normal(1.0, 0.12)

    price = round(base_price * age_factor * room_factor * floor_factor * noise, -3)
    price = int(max(price, 100_000))

    records.append({
        "area_sqm": area,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "floor": floor,
        "property_age": property_age,
        "latitude": lat,
        "longitude": lon,
        "district": district,
        "price": price,
    })

df = pd.DataFrame(records)
output_path = Path(__file__).parent / "sample_data.csv"
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} records -> {output_path}")
print(f"Price range: {df['price'].min():,} – {df['price'].max():,} EGP")
print(f"Districts: {df['district'].nunique()}")
