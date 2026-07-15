# Internship Report: AlfaScript Real Estate Feasibility Platform

> **Company:** Kapitalys Conseil SAS (AlfaScript)
> **Intern:** Haidra Mohammad
> **Supervisor:** Alain Al Jebbaoui
> **Internship period:** 22 June 2026 to 21 August 2026
> **Version:** 3.0 - Final technical delivery, 15 July 2026

## 1. Executive Summary

This internship delivered a French real-estate analysis prototype for historical sale analysis, residential price estimates, parcel lookup, preliminary land reference, development-feasibility scenarios, and investment analysis.

The system uses corrected, geolocated DVF transaction data from 2021 to 2025, commune-level INSEE indicators, construction-cost benchmarks, time features, and nearby comparable transactions. It is served by FastAPI through a responsive web application.

The project is a decision-support prototype. It is not a cadastral valuation, a planning opinion, an engineering study, tax advice, or an automated permitability system.

## 2. Deliverables

| Deliverable | Implementation |
|---|---|
| Historical transaction data | Corrected 2021-2025 DVF release snapshot with 4,006,005 residential observations across 93 departments |
| Geographic and socioeconomic analysis | INSEE commune enrichment, department features, map and nearby-comparable analysis |
| Residential valuation | General and apartment XGBoost models with point-in-time validation |
| Land value reference | Experimental residual land proxy based on house transactions less benchmark replacement cost |
| Cadastre | IGN APICarto parcel-reference lookup, parcel area, and centroid |
| Simplified planning feasibility | Manual CES, gross-floor-area ratio, number-of-floors, saleable-ratio, and PLU/PLUi-source scenario |
| Development business plan | Land cost, acquisition fees, construction, additional costs, sales revenue, margin, and ROI |
| Investment analysis | Loan payment, cash flow, cash-on-cash ROI, and IRR with purchase and sale costs |
| Documentation and handoff | README, technical appendix, data-governance register, multi-country architecture, release checklist, manifest generator, and tests |

## 3. Data Pipeline

### 3.1 DVF Transactions

**Source:** geo-DVF, data.gouv.fr.

The downloader supports multi-year and multi-department collection. It uses bounded HTTP timeouts and concurrent department downloads.

The cleaner handles a DVF mutation as a transaction rather than retaining an arbitrary first row. It keeps one residential type per mutation, aggregates built surface by local where identifiers are available, aggregates terrain by parcel, and excludes mixed residential-type or multi-location mutations that cannot be apportioned reliably.

**Release snapshot:**

| Metric | Value |
|---|---:|
| Period | 2021-01-01 to 2025-12-31 |
| Residential observations | 4,006,005 |
| Houses | 2,056,018 |
| Apartments | 1,949,987 |
| Departments | 93 |
| Missing departments | 57, 67, 68 were unavailable from the geo-DVF source |

### 3.2 INSEE And Construction Features

INSEE commune data contributes population, population density, poverty rate, and median-income features. Department construction benchmarks provide an indicative cost per m2. Sources, limits, and update requirements are documented in `docs/data_sources.md`.

## 4. Modelling Methodology

### 4.1 Features

- Property size, room count, land area where relevant, latitude, longitude, department, and property type.
- INSEE socioeconomic features.
- Sale year, month, quarter, and months since January 2021.
- Bounded nearby comparable-sale features for 500m, 1km, and 2km radii.

### 4.2 Point-In-Time Validation

The earlier random split was replaced with a temporal protocol:

1. Comparable features for 2024 use only transactions before 1 January 2024.
2. Comparable features for 2025 use only transactions before 1 January 2025.
3. 2024 observations train the evaluation model.
4. 2025 is the held-out backtest year.
5. The deployment model is retrained through 2025 using each transaction's original point-in-time features.

This prevents a 2025 sale price from influencing a 2025 prediction through the comparable-sale features.

### 4.3 Model Results

| Model | Target | 2025 R2 | MAE | Median APE |
|---|---|---:|---:|---:|
| General XGBoost | Total residential transaction price | 0.762 | EUR 52,072 | 17.7% |
| Apartment XGBoost | Apartment price per m2 | 0.783 | EUR 783/m2 | 15.1% |
| Residual land proxy | House price less benchmark replacement cost | 0.312 | EUR 43,678 | 43.1% |

The general and apartment models are suitable for exploratory decision support with an explicit error range. The residual land proxy has high error and must remain an experimental reference, not a standalone land valuation.

The model artifact stores empirical 80% prediction-interval multipliers measured on the 2025 holdout. The interface describes these as a historical error range, not a statistical guarantee.

## 5. Feasibility And Business Plan

The Land tab includes a development-feasibility calculation. The user must enter the applicable planning-rule source and values from the PLU/PLUi or planning certificate.

| Formula | Calculation |
|---|---|
| Maximum footprint | Parcel area x CES |
| Maximum gross floor area | Parcel area x gross-floor-area ratio |
| Buildable footprint | Minimum of CES footprint and gross area divided by floors |
| Saleable area | Gross floor area x saleable ratio |
| Total cost | Land purchase + acquisition fees + construction + additional costs |
| Sales revenue | Saleable area x declared sale price per saleable m2 |
| Margin | Sales revenue - total cost |
| ROI | Margin / total cost |

The application does not infer zoning, CES, permit status, utilities, contamination, flood risk, title restrictions, or buildability. These require qualified professional review.

## 6. Application Architecture

| Layer | Technology |
|---|---|
| API | Python 3.11, FastAPI, Uvicorn |
| Models | XGBoost, scikit-learn, joblib |
| Data | pandas, PyArrow Parquet, SQLite cache |
| Geospatial | GeoPandas, Shapely, BallTree |
| Frontend | HTML, CSS, JavaScript, Leaflet, optional Cesium presentation |
| Deployment | Docker, long-running container platform such as Hugging Face Spaces, Cloud Run, or VPS |

Vercel is not a supported target for the full ML application because it depends on large model/data artifacts and an in-memory comparable index.

## 7. Quality Controls

- Pydantic validation restricts requests to French coordinate coverage and supported residential property types.
- External map, nearby-place, and cadastre proxy endpoints have bounded inputs and per-client request limits.
- Deployment cache files use `ALFASCRIPT_CACHE_DIR`, which is writable in container and serverless environments.
- Tests cover point-in-time comparable isolation, feasibility formulas, corrected finance calculations, and invalid coordinates.
- `models/release_manifest.json` records data/model checksums, model features, validation protocol, metrics, and compatibility versions.

## 8. Limitations And Recommendations

1. The residual land proxy is not accurate enough for acquisition decisions without a human valuation.
2. Planning values are manual inputs. Automatic PLU/PLUi validation requires a legally reliable, municipality-level planning-data source and separate implementation.
3. DVF lacks building condition, energy performance, floor level, parking, school quality, and other important price drivers.
4. Rural and low-volume departments remain less reliable than high-volume urban areas.
5. Construction benchmarks are indicative defaults and must be replaced by project-specific cost studies.
6. Public release requires company-approved privacy, terms, valuation disclaimer, data licensing, monitoring, and artifact-storage procedures.

## 9. Test And Handoff

Run:

```bash
make test
make run
```

Open `http://127.0.0.1:8000`. Use `docs/final_delivery_checklist.md` as the acceptance checklist. The release manifest must be regenerated whenever data or models change.
