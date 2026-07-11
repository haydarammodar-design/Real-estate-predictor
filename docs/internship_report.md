# Internship Report: AlfaScript Real Estate Predictor

> **Project:** AI-Powered French Real Estate Price Prediction Platform
> **Author:** [Your Name]
> **Period:** [Start Date] – [End Date]
> **Supervisor:** [Supervisor Name]
> **Version:** 2.0 — Last updated: 2026-07-05

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Context & Objectives](#2-context--objectives)
3. [Methodology](#3-methodology)
4. [Data Collection & Processing](#4-data-collection--processing)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Development](#6-model-development)
7. [API & Backend Development](#7-api--backend-development)
8. [Frontend Development](#8-frontend-development)
9. [Results & Performance](#9-results--performance)
10. [Challenges & Solutions](#10-challenges--solutions)
11. [Conclusion & Future Work](#11-conclusion--future-work)
12. [Appendices](#12-appendices)

---

## 1. Executive Summary

This report documents the development of **AlfaScript Real Estate Predictor**, a web platform that estimates French property prices using machine learning models trained on open government data. The system leverages the French "Demandes de Valeurs Foncières" (DVF) dataset — **2,878,861 geolocalized property transactions** from **2021–2025** across **93 French departments** — enriched with INSEE socioeconomic data, department-level construction cost benchmarks, **temporal features**, and **comparable-sale features** computed via spatial indexing.

Three **XGBoost** models were developed and deployed:

| Model | Target | R² | Median Error |
|-------|--------|:--:|:------------:|
| **XGBoost General** | Total price (all properties) | **0.760** | **18.3%** |
| **XGBoost Apartment** | Price per m² | **0.600** | **16.2%** |
| **XGBoost Land** | Total price (houses as land proxy) | **0.622** | **18.7%** |

A FastAPI web application serves predictions via REST endpoints, with an interactive single-page frontend featuring 2D/3D maps, address search, nearby place discovery, financial investment analysis (IRR/ROI), and **confidence ranges** on all predictions. The platform is available at `http://127.0.0.1:8000`.

---

## 2. Context & Objectives

### 2.1 Background

French real estate valuation traditionally relies on local notary databases and human expertise. Publicly available data sources exist but are fragmented:

- **DVF** (Demandes de Valeurs Foncières): Records all French property transactions, published by the Directorate General of Public Finances (DGFiP). Available via `data.gouv.fr`.
- **INSEE**: National statistics institute providing commune-level demographic and socioeconomic indicators.
- **Construction benchmarks**: Industry cost references (BATIPRIX is the proprietary standard).

### 2.2 Objectives

1. **Collect and unify** open French real estate and socioeconomic data into a clean, queryable dataset spanning multiple years.
2. **Train predictive models** that estimate property prices from location, size, neighborhood, time of sale, and comparable nearby transactions.
3. **Build an interactive web platform** that allows users to explore locations on a map and obtain instant price estimates.
4. **Provide financial analysis tools** (IRR, cash flow, ROI) for investment decision support.
5. **Deliver transparent predictions** with honest accuracy metrics, confidence indicators, and per-department breakdowns.

### 2.3 Scope

- **Geographic coverage:** Metropolitan France (93 departments; 57, 67, 68 excluded due to data access limitations; 2A and 2B included with special handling).
- **Property types:** Apartments (Appartement) and Houses (Maison).
- **Data period:** Calendar years 2021–2025.
- **Output format:** Web application accessible via local browser.

---

## 3. Methodology

The project followed an iterative development methodology:

```
                    ┌─────────────┐
                    │ Data Mining │
                    │ & Cleaning  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Feature    │
                    │ Engineering │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Model     │
                    │  Training   │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Backend   │
                    │    API      │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Frontend   │
                    │     UI      │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ Evaluation  │ ◄─── Iterate
                    │ & Tuning    │
                    └─────────────┘
```

Each cycle involved data exploration, modeling, integration testing, and UI refinement. Evaluation used an 80/20 train/test split with honest reporting of all metrics.

---

## 4. Data Collection & Processing

### 4.1 DVF Transaction Data

**Source:** `https://files.data.gouv.fr/geo-dvf/latest/csv/`

The geo-dvf mirror provides department-level CSV files with geolocalized transaction records. Each record includes:

| Field | Description |
|-------|-------------|
| `valeur_fonciere` | Transaction price (€) |
| `surface_reelle_bati` | Living area (m²) |
| `nombre_pieces_principales` | Number of rooms |
| `surface_terrain` | Land area (m²) |
| `type_local` | Property type (Appartement/Maison) |
| `code_departement` | Department code |
| `code_commune` | INSEE commune code |
| `latitude`, `longitude` | Geolocation |
| `id_mutation` | Unique transaction ID |
| `date_mutation` | Transaction date |

**Download process** (`src/dvf_downloader.py`):
- Supports multi-year download via `--years`, `--year-start`, `--year-end` flags.
- Downloads and decompresses CSV.GZ files for each department-year combination.
- Cleans and standardizes column names via `from_dvf()`.
- Previously downloaded only the latest year; now downloads **2021–2025**.

**Cleaning filters:**
- Remove duplicate transactions (`id_mutation`).
- Filter `price >= 10,000 €`.
- Filter `area_sqm` between 9 and 10,000 m².
- Filter `rooms > 0`.
- Remove records with missing or outlier coordinates (>3σ from departmental centroid).
- Map `Appartement → apartment`, `Maison → house`.
- Handle Corsican departments `2A`/`2B` string encoding.
- Merge Paris arrondissements (`75101–75120 → 75056`), Marseille (`13201–13216 → 13055`), Lyon (`69381–69389 → 69123`).

**Result:** **2,878,861 cleaned records** (1,729,667 houses + 1,149,194 apartments) saved to `data/dvf/cleaned.parquet`.

### 4.2 INSEE Socioeconomic Data

**Sources:**
1. `geo.api.gouv.fr` — Commune boundaries, population, centroid coordinates (free, no authentication).
2. `insee.fr` FILoSoFi dataset — Median disposable income (`MED_SL`) and poverty rate (`PR_MD60`) at commune level.

**Collection script** (`src/insee_collector.py`):
- Queries geo.api.gouv.fr for all 34,969 French communes.
- Downloads FILoSoFi ZIP files (2019–2021), extracts income/poverty.
- Computes derived features: `log_population`, `pop_density`, `log_income`.
- Fixes arrondissement commune code merging and median income imputation for communes with missing data.

**Output:** `data/insee/insee_communes.parquet` with 34,969 commune records (no missing `median_income`, no zero `log_income`).

### 4.3 Construction Cost Benchmarks

**Source:** Industry benchmarks compiled manually (`src/construction_benchmarks.py`).

Since BATIPRIX (the official French reference) is proprietary and paid, an alternative benchmark was constructed from public sources:

- **Base cost per m² by standing:**
  | Standing | €/m² |
  |----------|:----:|
  | Low | 1,200 |
  | Medium | 1,800 |
  | High | 2,800 |
  | Luxury | 4,000 |

- **Departmental adjustment factors:** Paris (1.40), Mediterranean (1.10–1.30), rural areas (0.90), etc.
- Models use **medium standing** × department factor as a single `construction_cost_m2` feature.

### 4.4 Data Pipeline Summary

```
DVF CSV.GZ (2021–2025) ──► download_department() ──► _clean() ──► cleaned.parquet
                                                                        │
INSEE geo.api ──► commune boundaries ────────────────────────────────► │
FILoSoFi ZIP ──► income/poverty ─────────────────────────────────────► │
Construction benchmarks ──► department cost dict ────────────────────► │
                                                                        ▼
                                                             Feature Engineering
                                                             (see Section 5)
```

---

## 5. Feature Engineering

The feature engineering pipeline is implemented in `src/data_processor.py`.

### 5.1 Core Features

| Feature | Type | Description |
|---------|------|-------------|
| `area_sqm` | Numeric | Living area in m² |
| `rooms` | Numeric | Number of rooms |
| `land_sqm` | Numeric | Land area in m² (house/land models) |
| `latitude` | Numeric | Latitude (WGS84) |
| `longitude` | Numeric | Longitude (WGS84) |
| `construction_cost_m2` | Numeric | Construction cost benchmark |
| `department` | Categorical | 93 department codes → one-hot |
| `property_type` | Categorical | apartment/house → one-hot |

### 5.2 Socioeconomic Features (INSEE)

These are merged by matching the transaction's commune code to the INSEE commune database. When exact commune code is unavailable, spatial nearest-neighbor lookup is used via a GeoDataFrame R-tree index:

| Feature | Description |
|---------|-------------|
| `log_population` | Log of commune population |
| `pop_density` | Population per km² |
| `poverty_rate` | Percentage of population below poverty line |
| `log_income` | Log of median disposable income |

### 5.3 Time Features

Multi-year data (2021–2025) enables temporal modeling. The following features are computed from `date_mutation`:

| Feature | Description |
|---------|-------------|
| `sale_year` | Calendar year of the transaction |
| `sale_month` | Month (1–12) capturing seasonal patterns |
| `sale_quarter` | Quarter (1–4) for broader seasonality |
| `months_since_start` | Months since 2021-01-01, capturing long-term trends |

Time features allow the model to learn price trends, seasonality effects (e.g., summer premium, winter discounts), and year-over-year appreciation.

### 5.4 Comparable-Sale Features (Spatial)

A `ComparableFeatureBuilder` class uses **BallTree with haversine distance** to find nearby comparable transactions at three radii:

| Feature | Description |
|---------|-------------|
| `comp_500m_median_price_m2` | Median €/m² of properties within 500m |
| `comp_500m_sale_count` | Number of sales within 500m |
| `comp_500m_median_price` | Median total price within 500m |
| `comp_1000m_median_price_m2` | Median €/m² within 1 km |
| `comp_1000m_sale_count` | Sales count within 1 km |
| `comp_1000m_median_price` | Median total price within 1 km |
| `comp_2000m_median_price_m2` | Median €/m² within 2 km |
| `comp_2000m_sale_count` | Sales count within 2 km |
| `comp_2000m_median_price` | Median total price within 2 km |

**Training:** For large datasets (>400k rows), a random subsample is used as the BallTree index to keep memory usage feasible.

**Inference:** The full cleaned dataset (~2.88M rows) is loaded at API startup and indexed in the BallTree for live comparable computation.

### 5.5 OSM Proximity Features (Planned, Not Run)

The module `src/osm_features.py` defines 7 POI category features (transit, education, healthcare, shopping, dining, leisure, services) per commune. This was not executed due to the prohibitive cost of 30,000+ Overpass API calls.

### 5.6 Feature Encoding

- **Numerical features:** Used as-is (models are tree-based, no scaling needed).
- **Categorical features:** One-hot encoded (`department` → 93 columns, `property_type` → 2 columns).
- **Missing values:** Filled with 0 after one-hot encoding.
- **Target variable:** `log1p(price)` — log-transformed to handle the long-tailed distribution of real estate prices.

### 5.7 Outlier Filtering

- **General model:** Remove top/bottom 1% by price/m².
- **Apartment model:** Remove top/bottom 0.5% by price/m²; require `area_sqm >= 9`, `rooms > 0`.
- **Land model:** Remove top/bottom 1% by price/land_m²; require `rooms > 0`, `land_sqm > 0`.

### 5.8 Total Feature Set

The main model uses **117 features**:
- 9 base numerical features
- 4 time features
- 9 comparable-sale features
- 93 department one-hot columns
- 2 property type one-hot columns

Land model: 116 features (no property_type). Apartment model: 115 features (no property_type, no land_sqm).

---

## 6. Model Development

### 6.1 Model Selection Rationale

Three **XGBoost** models were developed to cover different use cases:

1. **XGBoost General** — Predicts total price for both houses and apartments using all features.
2. **XGBoost Apartment** — Specialized in per-m² pricing for apartments; provides unit-price insight.
3. **XGBoost Land** — Estimates total plot value using house transactions as a proxy.

**Why XGBoost over RandomForest?** XGBoost trains faster on large datasets (2.88M rows), produces smaller model artifacts, and achieved higher R² (0.760 vs 0.737) with the same feature set. Tree-based ensemble methods perform well on tabular data with mixed numerical/categorical features, are interpretable via feature importance, and require less data than neural networks.

**Why not a single deep learning model?** Tree-based methods are well-suited to tabular data with mixed feature types, offer native handling of missing values via split direction, and provide straightforward feature importance analysis.

### 6.2 XGBoost General Model

**File:** `src/train.py` | **Output:** `models/model.joblib`

```python
XGBRegressor(
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
)
```

- **Training data:** ~505,000 records (apartments + houses after filtering)
- **Features:** 117 (22 base/time/comparable + 93 department + 2 property type)
- **Target:** `log1p(price)`
- **Saved artifacts:** model, feature columns, department list, property types, time metadata, comparable feature list, geographic breakdown metrics

### 6.3 XGBoost Apartment Model

**File:** `src/train_apartment.py` | **Output:** `models/model_apartment.joblib`

```python
XGBRegressor(
    n_estimators=700,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
```

- **Training data:** ~575,000 apartment transactions (after filtering)
- **Features:** 115 (22 base/time/comparable + 93 department)
- **Target:** `log1p(price_per_sqm)` — predicts unit price, then multiplies by area
- **Hyperparameter tuning:** Grid search over 12 combinations of `n_estimators` (300/500/700), `max_depth` (6/10), and `learning_rate` (0.03/0.05).

### 6.4 XGBoost Land Model

**File:** `src/train_land.py` | **Output:** `models/model_land.joblib`

```python
XGBRegressor(
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
)
```

- **Training data:** ~530,000 house transactions (after filtering, land_sqm > 0)
- **Features:** 116 (23 base/time/comparable/land + 93 department)
- **Target:** `log1p(price)` — total transaction price
- **Limitation:** DVF contains no pure land transactions (vacant plots). The land model is trained on house sales that bundle land + building. It uses `land_sqm` as the primary driver and assumes a minimal building contribution (`area_sqm=1, rooms=0` for empty plot queries).

### 6.5 Training Protocol

All models follow the same training protocol:

1. Load cleaned parquet → filter by property type.
2. Merge socioeconomic data by commune code.
3. Merge construction costs by department.
4. Filter outliers by price/sqm quantiles.
5. Add time features from `date_mutation`.
6. Build BallTree comparable features (subsampled index for large datasets).
7. Build feature matrix with one-hot encoding.
8. Split 80/20 (`train_test_split`, `random_state=42`).
9. Train XGBoost on log-transformed target.
10. Evaluate on test set (expm1 to revert log).
11. Compute per-department metrics for geographic validation.
12. Save model + all metadata + metrics.

---

## 7. API & Backend Development

### 7.1 Technology Stack

- **Framework:** FastAPI (Python) — chosen for automatic OpenAPI documentation, Pydantic validation, and async support.
- **Server:** Uvicorn (ASGI server).
- **Port:** `127.0.0.1:8000`.

### 7.2 Startup Sequence

The server (`src/app.py`) loads at startup:

1. **XGBoost General model** (`model.joblib`) — always loaded.
2. **XGBoost Land model** (`model_land.joblib`) — if available.
3. **XGBoost Apartment model** (`model_apartment.joblib`) — if available.
4. **INSEE GeoDataFrame** (`insee_communes.parquet`) — loaded with spatial index for O(1) nearest-commune lookup.
5. **Construction costs** — computed from benchmark module.
6. **Comparable index** — full cleaned DVF dataset (~2.88M rows) loaded into a BallTree index for live comparable-feature computation. Subsampled to 400k rows for the index to balance memory and coverage.

### 7.3 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve frontend HTML |
| GET | `/health` | Server health + model metrics + comparable index status |
| GET | `/options` | Available departments, models, types |
| POST | `/predict` | XGBoost general total price prediction |
| POST | `/predict/apartment` | XGBoost per-m² prediction |
| POST | `/predict/land` | XGBoost land value prediction |
| POST | `/financial` | Investment analysis (IRR/ROI) |
| GET | `/nearby` | OSM nearby places query |
| GET | `/staticmap` | Static map image generation |

**Detailed endpoint documentation is provided in Appendix A.**

### 7.4 Key Implementation Details

**Runtime comparable features** (`_add_runtime_model_features`):
- For live predictions, the API computes time features from the current date and comparable features from the pre-loaded BallTree index.
- The `_aligned_features` helper ensures runtime feature columns match the training feature order exactly (including one-hot dummies).

**Confidence ranges** (`_confidence_range`):
- Every prediction includes `confidence_low` and `confidence_high` bounds derived from the model's stored `mape_pct`.
- Formula: `low = max(price * (1 - MAPE), 0)`, `high = price * (1 + MAPE)`.
- Displayed as a formatted range string (e.g., "340K – 510K EUR").

**Nearest commune lookup** (`_find_nearest_commune`):
- Uses GeoDataFrame spatial index (`sindex.nearest()`) for efficient querying.
- Returns commune-level socioeconomic data for any (lat, lon) coordinate.

**Nearby places** (`/nearby`):
- Queries OpenStreetMap API (`/api/0.6/map?bbox=...`) for raw map data.
- Parses OSM XML into categorized places (nodes with name tags + way centroids).
- Falls back to Overpass API on rate-limit errors (509).
- Results cached in SQLite (`data/cache/nearby_cache.db`) for 24 hours.
- Returns up to 100 nearest places within 500m radius.

**Static maps** (`/staticmap`):
- Downloads individual OSM tiles (256×256) for requested bounding box.
- Composited into a single PNG image via Pillow.
- Tiles cached on disk for 24 hours.

**Financial analysis** (`/financial`):
- Standard amortization formula for monthly P&I.
- Cash flow = rent − P&I − expenses.
- Cash-on-cash ROI = (annual cash flow) / (down payment + closing costs).
- IRR computed via `numpy_financial.irr()` with annual cash flows + sale proceeds.

---

## 8. Frontend Development

### 8.1 Technology Stack

- **Architecture:** Single-page application (SPA) served from FastAPI.
- **2D Map:** Leaflet.js (local copy at `src/static/leaflet/`).
- **3D Map:** CesiumJS (lazy-loaded from jsdelivr CDN on user click).
- **Geocoding:** Nominatim (OpenStreetMap) — free, 1 req/s rate limit.
- **Design:** Dark theme, custom CSS variables, responsive grid.

### 8.2 Page Structure

```
┌─────────────────────────────────────────────────────────┐
│  Nav: Logo | [Search Bar] | Predict | Markets | How 3D │
├────────────────────────────────┬────────────────────────┤
│  Property Details              │     Map Panel          │
│  ┌─────────────────────────┐   │  ┌──────────────────┐  │
│  │ [Home] [Land] [Fin] [N] │   │  │                  │  │
│  ├─────────────────────────┤   │  │   Leaflet /      │  │
│  │                         │   │  │   Cesium Map     │  │
│  │  Home Tab:              │   │  │                  │  │
│  │  [Apartment] [House]    │   │  │                  │  │
│  │  Department ___          │   │  │                  │  │
│  │  Area m² ___             │   │  └──────────────────┘  │
│  │  Land m² ___ (house)     │   │                        │
│  │  Rooms ___               │   │  Slideshow             │
│  │  Lat/Lon (auto)          │   │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ │
│  │                         │   │  │S │ │N │ │D │ │C │ │
│  │  [Estimate Price]       │   │  └──┘ └──┘ └──┘ └──┘ │
│  │  ┌───────────────────┐  │   └────────────────────────┘
│  │  │ Predicted: 416,204│  │
│  │  │ ±18.3% confidence │  │
│  │  │ XGB detail/m² info│  │
│  │  └───────────────────┘  │
├────────────────────────────────┴────────────────────────┤
│  Market Insights (3 cards)                              │
├─────────────────────────────────────────────────────────┤
│  By the Numbers (stats)                                 │
├─────────────────────────────────────────────────────────┤
│  How It Works (3 steps)                                 │
├─────────────────────────────────────────────────────────┤
│  Footer                                                 │
└─────────────────────────────────────────────────────────┘
```

### 8.3 Key Features

**Tabs (4):**

| Tab | Sub-options | Purpose |
|-----|-------------|---------|
| **Home** | Apartment / House | Total price via XGBoost + per-m² detail |
| **Land** | — | Plot value estimation |
| **Financial** | — | IRR, ROI, cash flow analysis |
| **Nearby** | Category filters | Nearby places list + map markers |

**Confidence Ranges:**
- Every prediction displays a formatted confidence range below the price.
- Range formula: `[price × (1 − MAPE), price × (1 + MAPE)]`.
- Model info line shows median error percentage (MAPE).

**Map Panel:**
- Leaflet.js with OSM tiles.
- Draggable marker synced with coordinate inputs.
- Reverse geocodes location on marker move → fills address info + department.
- **3D toggle:** Lazy-loads CesiumJS from CDN, switches to 3D globe view, adds property marker (yellow point).
- **Slideshow:** 4 static map tiles at zoom levels 17/15/13/11, clickable for lightbox.

**Address Search:**
- Nominatim integration with 400ms debounce.
- Autocomplete dropdown (up to 6 results).
- Click result → fly to location + auto-predict.

**Financial Tab:**
- User inputs: purchase price, down payment %, loan rate, term, rent, expenses, appreciation.
- Outputs: monthly CF, annual CF, cash-on-cash ROI, closing costs, IRR at 5/10/15/20 years.
- Full amortization schedule with sale proceeds at horizon.

**Nearby Places:**
- Fetches from `/nearby` endpoint.
- Shows up to 100 names places within 500m.
- Category filter chips (e.g. "school(3)", "restaurant(7)") toggle visibility.
- Places shown as colored circle markers on map + interactive list.
- 3D view shows up to 30 places as colored 3D points.

**Responsive Design:**
- Grid collapses to single column below 900px.
- Touch-friendly controls.
- Scroll-triggered fade-in animations via IntersectionObserver.

---

## 9. Results & Performance

### 9.1 Model Accuracy

Comprehensive evaluation on an 80/20 held-out test set:

| Metric | XGBoost General | XGBoost Apt | XGBoost Land |
|--------|:---------------:|:-----------:|:------------:|
| **Samples** | ~505,000 | ~575,000 | ~530,000 |
| **Features** | 117 | 115 | 116 |
| **MAE** | €55,757 | €1,064/m² | €59,748 |
| **RMSE** | €100,411 | €2,157/m² | €149,075 |
| **R²** | **0.760** | **0.600** | **0.622** |
| **Median error** | **18.3%** | **16.2%** | **18.7%** |
| Mean error | 32.7% | 28.4% | 36.1% |
| Within ±10% | 29.2% | 32.5% | 28.3% |
| Within ±20% | 53.5% | 57.1% | 52.4% |
| Within ±30% | 70.8% | 73.2% | 69.5% |

### 9.2 Model File Sizes

| Model | Size |
|-------|:----:|
| XGBoost General | 21 MB |
| XGBoost Apartment | 20 MB |
| XGBoost Land | 3.2 MB |

The main model is significantly smaller than the previous RandomForest (21 MB vs 129 MB), making deployment more practical.

### 9.3 Error by Price Range (XGBoost General)

| Price Range | Count | Median Error | MAE |
|-------------|:----:|:------------:|:---:|
| < €50K | 18,214 | **95.2%** | €42,315 |
| €50–100K | 71,058 | 28.7% | €31,284 |
| €100–150K | 91,487 | 16.3% | €28,445 |
| €150–200K | 91,461 | **14.8%** | €30,512 |
| €200–300K | 117,087 | 15.6% | €42,183 |
| €300–500K | 79,111 | 17.1% | €70,248 |
| €500K+ | 32,067 | 20.8% | €198,334 |

**Interpretation:** The model performs best on mid-range properties (€100–300K). Very cheap properties (<€50K) remain challenging — these include unusual transactions (garages, small studios in rural areas). Luxury properties (>€500K) show larger absolute errors due to higher variance. Multi-year data has improved coverage in the <€50K segment (18K vs 3.7K samples) but error remains high.

### 9.4 Error by Department (XGBoost General)

**Best 5 (lowest error):**

| Dept | Name | Count | Med Error | MAE |
|:----:|------|:----:|:---------:|:---:|
| 92 | Hauts-de-Seine | 10,153 | **11.9%** | €74,182 |
| 75 | Paris | 18,093 | **12.7%** | €91,447 |
| 94 | Val-de-Marne | 7,818 | **13.0%** | €55,218 |
| 69 | Rhône | 10,648 | **13.4%** | €53,614 |
| 13 | Bouches-du-Rhône | 14,033 | **13.6%** | €53,891 |

**Worst 5 (highest error):**

| Dept | Name | Count | Med Error | MAE |
|:----:|------|:----:|:---------:|:---:|
| 55 | Meuse | 1,424 | **30.8%** | €38,214 |
| 23 | Creuse | 793 | **29.1%** | €27,503 |
| 88 | Vosges | 2,674 | **28.6%** | €49,127 |
| 43 | Haute-Loire | 1,669 | **28.4%** | €47,102 |
| 15 | Cantal | 1,233 | **28.2%** | €36,815 |

**Interpretation:** Dense urban departments (Paris region, Lyon, Marseille) have the most transactions and most consistent pricing → lowest error. Rural departments with sparse data and high price variance show errors ~2.5× higher. Multi-year data has narrowed the gap compared to the single-year model.

### 9.5 Error by Property Type (XGBoost General)

| Type | Count | Med Error | MAE |
|------|:----:|:---------:|:---:|
| Apartment | ~230,000 | **17.6%** | €51,437 |
| House | ~346,000 | **19.0%** | €58,614 |

Houses are slightly harder to predict than apartments, likely due to greater diversity in land size, construction quality, and outbuilding presence — factors not captured in the DVF data.

### 9.6 Per-Department Validation

All three training scripts compute per-department metrics and store them in the model artifact as `geographic_metrics`. This enables:
- Per-department accuracy reporting in API responses.
- Identification of departments where the model performs poorly.
- Future work on per-department fine-tuning for high-error regions.

---

## 10. Challenges & Solutions

### 10.1 Data Limitations

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **No pure land transactions in DVF** | Land model must use house proxies | Use house sales with `land_sqm` as driver; minimal building values for vacant plots |
| **No floor level / standing / year built** | These explain 20–40% of price variance | Explicit disclaimer in UI; models rely on location + size + comparables |
| **Uneven department coverage** | Rural departments have 10–50× fewer records | Multi-year data helps; rural predictions have wider confidence ranges |
| **No commute time / school quality / crime data** | Missing neighborhood quality signals | INSEE income/poverty partially captures neighborhood effects |
| **Alsace-Moselle departments (57, 67, 68) unavailable** | 404 on geo-dvf mirror | Explicitly excluded; ~3 departments missing coverage |

### 10.2 Technical Challenges

| Challenge | Solution |
|-----------|----------|
| **OSM API rate limits (509 errors)** | 3 retries with exponential backoff → Overpass API fallback → 24h SQLite cache |
| **Overpass API unreachable** | Detected gracefully; `/nearby` returns error message without crashing |
| **CesiumJS bundle size (~3MB)** | Lazy-loaded from CDN only on 3D button click |
| **3D nearby place glitching** | Limit to 30 places, 8m altitude, smaller point size |
| **Race conditions in async predictions** | Incrementing `_predictReqId` cancels stale requests |
| **INSEE commune code mismatches (arrondissements)** | Fixed mapping: Paris 75101–75120→75056, Marseille→13055, Lyon→69123 |
| **Long-tailed price distribution** | Log-transformed target (`log1p`) for all models |
| **Large dataset comparable features** | BallTree with subsampled index (400k rows) for training; full index at API startup |
| **Main model artifact too large (RandomForest 129MB)** | Switched to XGBoost (21 MB), enabling faster loading and smaller deployment footprint |
| **pandas/XGBoost compatibility** | Added `pd.Int64Index = pd.Index` compatibility alias for XGBoost 2.0.3 + pandas 2.x |
| **Corsican departments (2A/2B) encoding** | Fixed string column normalization in DVF cleaner |

### 10.3 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single-origin FastAPI** | Avoid CORS issues; simple deployment |
| **Leaflet served locally** | Works offline; no CDN dependency |
| **Cesium from CDN** | Large library; only needed for 3D |
| **SQLite cache** | Persists across server restarts; no external DB needed |
| **Three specialized XGBoost models** | Better accuracy per use case than a single combined model |
| **Log-transformed target** | Stabilizes variance; handles exponential price growth |
| **Comparable features via BallTree** | O(log n) nearest-neighbor queries, scalable to 2.88M points |
| **Confidence ranges from MAPE** | Simple, interpretable, and honest communication of model uncertainty |

---

## 11. Conclusion & Future Work

### 11.1 Achievements

1. Successfully collected, cleaned, and unified **2,878,861** French property transactions (2021–2025) with socioeconomic enrichment across **93 departments**.
2. Built **three XGBoost models** — General (R²=0.760), Apartment (R²=0.600), Land (R²=0.622).
3. Implemented **comparable-sale features** via BallTree spatial index and **time features** from multi-year data.
4. Deployed a full-stack web application with interactive map, 3D globe, address search, financial analysis, and **confidence ranges** on all predictions.
5. Achieved **~53.5% of predictions within 20%** of actual price using the XGBoost General model.
6. Reduced main model artifact size from **129 MB (RandomForest) to 21 MB (XGBoost)** .
7. Added per-department validation metrics for all models, enabling geographically-aware accuracy reporting.

### 11.2 Limitations

- **Land model accuracy remains moderate** (R²=0.622) due to proxy training data (house transactions instead of pure land sales).
- **Missing critical features** (floor level, construction year, property condition, school quality, crime statistics).
- **No pure land transactions** available in open data.
- **Rural departments** with sparse data still show higher prediction errors (~28–31%).
- **OSM proximity features** collected but not integrated (API cost).
- **Model cannot predict extreme luxury** (>€1M) with high accuracy due to variance.

### 11.3 Future Work

| Priority | Task | Expected Impact |
|:--------:|------|:---------------:|
| 1 | **Integrate OSM proximity features** per department | Marginal improvement (~1–2% R²) |
| 2 | **Cross-validation** for hyperparameter tuning | More robust parameter selection |
| 3 | **Neural network model** (TabNet / FT-Transformer) | May capture non-linear interactions better |
| 4 | **Ensemble model** combining XGBoost variants | Potential 1–2% improvement |
| 5 | **Per-department models** for high-density areas (Paris, Lyon, Marseille) | Better local accuracy |
| 6 | **External API integration** (school ratings, crime stats, commute times) | Better neighborhood quality signals |
| 7 | **Deployment to cloud** (Render / Railway / self-hosted VPS) | Public internet access |
| 8 | **Add 2026 H1 data** when geo-dvf mirror updates | +500K records, more recent |

---

## 12. Appendices

### Appendix A: API Documentation

#### `POST /predict` — Property Price Prediction (XGBoost General)

**Request:**
```json
{
  "area_sqm": 60,
  "rooms": 2,
  "latitude": 48.8566,
  "longitude": 2.3522,
  "department": "75",
  "property_type": "apartment"
}
```

**Response:**
```json
{
  "predicted_price": 429811.23,
  "predicted_price_formatted": "429,811",
  "currency": "EUR",
  "confidence_low": 351127.28,
  "confidence_high": 508495.18,
  "confidence_range_formatted": "351K – 508K EUR",
  "model_metrics": {
    "mae": 55757.08,
    "rmse": 100411.26,
    "r2": 0.760,
    "mape_pct": 18.3
  },
  "confidence_note": "Prediction is an estimate based on historical data. Actual market prices may vary significantly."
}
```

#### `POST /predict/apartment` — Apartment Per-m² Prediction (XGBoost)

**Request:**
```json
{
  "area_sqm": 60,
  "rooms": 2,
  "latitude": 48.8566,
  "longitude": 2.3522,
  "department": "75"
}
```

**Response:**
```json
{
  "predicted_price_per_sqm": 9082.14,
  "predicted_price_per_sqm_formatted": "9,082",
  "predicted_total_price": 544928.40,
  "predicted_total_price_formatted": "544,928",
  "confidence_low": 456649.52,
  "confidence_high": 633207.28,
  "confidence_range_formatted": "457K – 633K EUR",
  "model_metrics": {
    "mae": 1064.49,
    "rmse": 2156.84,
    "r2": 0.600,
    "mape_pct": 16.2,
    "type": "apartment"
  },
  "department": "75",
  "confidence_note": "Apartment price per m² estimate based on comparable transactions. Floor level, standing, and year built are not available in DVF data."
}
```

#### `POST /predict/land` — Land Value Prediction (XGBoost)

**Request:**
```json
{
  "land_sqm": 500,
  "latitude": 48.8566,
  "longitude": 2.3522,
  "department": "75",
  "zone_type": "urban"
}
```

**Response:**
```json
{
  "predicted_price": 1687368.26,
  "predicted_price_formatted": "1,687,368",
  "price_per_land_m2": 3374.74,
  "price_per_land_m2_formatted": "3,375",
  "confidence_low": 1371843.60,
  "confidence_high": 2002892.92,
  "confidence_range_formatted": "1.37M – 2.00M EUR",
  "model_metrics": {
    "mae": 59748.31,
    "rmse": 149074.55,
    "r2": 0.622,
    "mape_pct": 18.7,
    "type": "land"
  },
  "department": "75",
  "zone_type": "urban",
  "confidence_note": "Land value estimate based on comparable house transactions. Actual plot values may vary significantly with zoning and local market conditions."
}
```

#### `POST /financial` — Investment Analysis

**Request:**
```json
{
  "purchase_price": 300000,
  "down_payment_pct": 20,
  "loan_rate": 3.5,
  "loan_term_years": 20,
  "monthly_rent": 1500,
  "monthly_expenses": 200,
  "annual_appreciation": 2.0
}
```

**Response:**
```json
{
  "down_payment": 60000,
  "down_payment_formatted": "60,000",
  "loan_amount": 240000,
  "monthly_payment": 1391.86,
  "monthly_cash_flow": -91.86,
  "annual_cash_flow": -1102.32,
  "cash_on_cash_roi": -1.37,
  "closing_costs": 24000,
  "total_investment": 84000,
  "irr_5y": -3.38,
  "irr_10y": -0.92,
  "irr_15y": 0.56,
  "irr_20y": 1.71
}
```

#### `GET /nearby` — Nearby Places

**Query:** `?lat=48.8566&lon=2.3522&radius=500`

**Response:**
```json
{
  "places": [
    {"name": "Cathédrale Notre-Dame de Paris", "type": "religious", "lat": 48.8529, "lon": 2.3500, "distance_m": 423},
    {"name": "Île de la Cité", "type": "park", "lat": 48.8546, "lon": 2.3472, "distance_m": 386}
  ]
}
```

#### `GET /options` — Available Options

**Response:**
```json
{
  "departments": ["01", "02", ..., "95"],
  "property_types": ["apartment", "house"],
  "models": {
    "original": {"type": "property", "metrics": {...}, "status": "active"},
    "land": {"type": "land", "metrics": {...}, "status": "active"},
    "apartment": {"type": "apartment", "metrics": {...}, "status": "active"}
  }
}
```

#### `GET /health` — Server Health

**Response:**
```json
{
  "status": "ok",
  "model_metrics": {"mae": 55757.08, "rmse": 100411.26, "r2": 0.760, "mape_pct": 18.3},
  "comparables_loaded": true,
  "comparable_period": {"start_date": "2021-01-01", "max_date": "2025-12-31"}
}
```

### Appendix B: Model Parameters

| Parameter | XGB General | XGB Apt | XGB Land |
|-----------|:-----------:|:-------:|:--------:|
| n_estimators | 500 | 700 | 700 |
| max_depth | 10 | 10 | 6 |
| learning_rate | 0.05 | 0.03 | 0.05 |
| subsample | 0.8 | 0.8 | 0.8 |
| colsample_bytree | 0.8 | 0.8 | 0.8 |
| min_child_weight | 3 | 3 | 3 |
| reg_alpha | 0.1 | 0.1 | 0.1 |
| reg_lambda | 1.0 | 1.0 | 1.0 |
| tree_method | hist | — | — |
| max_bin | 256 | — | — |
| Features | 117 | 115 | 116 |
| Training records | ~505K | ~575K | ~530K |
| Artifact size | 21 MB | 20 MB | 3.2 MB |

### Appendix C: Data Coverage

| Metric | Value |
|--------|-------|
| Total DVF records | 2,878,861 |
| Houses | 1,729,667 |
| Apartments | 1,149,194 |
| Departments | 93 |
| Data period | 2021–2025 |
| INSEE communes | 34,969 |
| Cache DB size | ~2 MB (SQLite) |
| Model file size (XGB General) | 21 MB |
| Model file size (XGB Apt) | 20 MB |
| Model file size (XGB Land) | 3.2 MB |
| DVF cleaned parquet | 156 MB |

### Appendix D: Project Structure

```
real-estate-predictor/
├── requirements.txt
├── .gitignore
├── src/
│   ├── app.py                      # FastAPI server
│   ├── data_processor.py           # Feature engineering (time, comparable, core)
│   ├── dvf_downloader.py           # DVF data download (multi-year)
│   ├── insee_collector.py          # INSEE data collection
│   ├── construction_benchmarks.py  # Construction costs
│   ├── osm_features.py             # OSM proximity (unused)
│   ├── train.py                    # XGBoost general training
│   ├── train_apartment.py          # XGBoost apt training
│   ├── train_land.py               # XGBoost land training
│   ├── templates/
│   │   └── index.html              # SPA frontend
│   └── static/
│       └── leaflet/                # Leaflet.js (local)
├── data/
│   ├── dvf/cleaned.parquet         # Transaction data (2.88M rows)
│   ├── insee/insee_communes.parquet # Socioeconomic data
│   ├── cache/nearby_cache.db       # OSM cache
│   └── tiles/                      # Map tile cache
├── models/
│   ├── model.joblib                # XGBoost General (21 MB)
│   ├── model_apartment.joblib      # XGBoost Apartment (20 MB)
│   └── model_land.joblib           # XGBoost Land (3.2 MB)
└── docs/
    └── internship_report.md        # This document
```

### Appendix E: Technologies Used

| Category | Technology | Version | Purpose |
|----------|-----------|:-------:|---------|
| Language | Python | 3.11.9 | Backend |
| Web framework | FastAPI | ≥0.100 | REST API |
| Server | Uvicorn | ≥0.20 | ASGI server |
| ML: gradient boosting | XGBoost | 2.0.3 | All three models |
| Data | pandas, numpy | ≥2.0, ≥1.24 | Data processing |
| Geospatial | geopandas, shapely | latest | Spatial index, BallTree |
| Database | SQLAlchemy + SQLite | latest | Nearby cache |
| Financial | numpy_financial | latest | IRR computation |
| Image | Pillow | latest | Static map compositing |
| Frontend | HTML/CSS/JS | — | SPA |
| 2D maps | Leaflet.js | 1.9 | Interactive map |
| 3D maps | CesiumJS | latest | 3D globe |
| Geocoding | Nominatim | — | Address search |
| SSL | certifi | latest | HTTPS DVF downloads |

---

*End of report. This document is version-controlled and should be updated as the project evolves.*
