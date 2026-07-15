# Final Delivery Checklist

**Project:** AlfaScript real-estate valuation and feasibility prototype
**Internship period:** 22 June 2026 to 21 August 2026
**Supervisor:** Alain Al Jebbaoui, Kapitalys Conseil SAS

## Article 3: Internship Subject

| Expected delivery | Delivery evidence | Acceptance status |
|---|---|---|
| Historical real-estate sales analysis | Multi-year DVF ingestion, cleaning, 2021–2025 release dataset, point-in-time model backtest | Complete |
| Geographic sector analysis | INSEE commune enrichment, nearby comparable transactions, map, department metrics | Implemented |
| Cadastral data use | IGN APICarto parcel lookup with parcel area and centroid | Implemented; external API availability must be tested at handoff |
| Simplified CES/urban-planning analysis | `POST /development-feasibility`; CES, floor-area ratio, floors, and PLU/PLUi reference are mandatory user inputs | Implemented as a transparent manual-rule scenario; not automatic planning verification |
| Land-acquisition value | Residual land-value proxy based on house transactions less benchmark construction cost | Implemented as an indicative reference, not a cadastral valuation |
| Commercialisable/saleable surface | CES/floor-ratio feasibility formulas calculate footprint, gross floor area, and saleable area | Implemented |
| Potential sale-price evaluation | Residential models plus user-entered projected sale price per saleable m² in feasibility scenario | Implemented as a scenario input; validate against local market study |
| Forecast business plan | Feasibility endpoint calculates acquisition, construction, additional costs, revenue, margin, and ROI | Implemented |
| Expected profitability | Development margin, margin percentage, ROI; financial tab includes corrected purchase/sale costs in IRR | Implemented |

## Article 4: Assigned Missions

| Mission | Evidence | Status |
|---|---|---|
| Collect and structure real-estate and land data | `src/dvf_downloader.py`, `src/insee_collector.py`, `data/dvf/cleaned_final.parquet` | Complete |
| Statistical and exploratory analysis | Model validation metrics, geographic metrics, `models/release_manifest.json` | Complete |
| Study historical transactions | Corrected 2021–2025 DVF release dataset | Complete |
| Analyse simplified CES rules | Feasibility workflow with declared planning source | Complete within the stated simplified/manual scope |
| Build data stores and dashboards | Parquet data layer, SQLite cache, interactive web interface | Complete for prototype; not a production data warehouse |
| Develop predictive land valuation | Residual land-value proxy, limitations disclosed | Complete as an indicative model; requires professional valuation review for decisions |
| Contribute to business plans | Development feasibility and investment calculations | Complete |
| Produce analysis reports and recommendations | `docs/internship_report.md`, this checklist, release manifest | Complete |
| Define multi-country extensible data architecture | `docs/multi_country_architecture.md`, source-specific modules, and documented adapter contract | Complete for architecture design; country adapters remain future implementation work |

## Mandatory Release Tests

- [x] Run corrected DVF ingestion for 2021–2025 and verify counts, dates, property types, and departments.
- [x] Train all three models with the point-in-time 2024/2025 protocol.
- [x] Generate `models/release_manifest.json` and preserve its SHA-256 hashes.
- [x] Run `python3.11 -m unittest discover -s tests -v`.
- [x] Run `python3.11 -m compileall -q src`.
- [x] Start the API and verify `/health`.
- [x] Test `/predict`, `/predict/apartment`, `/predict/land`, `/development-feasibility`, and `/financial`.
- [x] Verify invalid French coordinates return HTTP 422.
- [x] Test `/lookup-parcel` with a live IGN response and invalid reference.
- [x] Verify unsupported department and invalid feasibility inputs return HTTP 422.
- [ ] Test desktop and mobile UI flows: address search, map pin, cadastre lookup, property valuation, land reference, feasibility, and financial calculations.
- [ ] Build and run the Docker image on the chosen deployment platform.

## Handoff Package

- [ ] Tagged source commit and release notes.
- [ ] `requirements.lock`, Dockerfile, README, this checklist, and updated internship report.
- [ ] Release models and source data delivered from approved artifact storage.
- [ ] `models/release_manifest.json` with model/data hashes and metrics.
- [ ] Data-source, licence, and retrieval-date register for DVF, INSEE, IGN APICarto, OpenStreetMap, Nominatim, Overpass, Cesium, and construction benchmarks.
- [ ] User-facing privacy notice, terms of use, and valuation/planning disclaimer before public launch.
- [ ] Supervisor acceptance record with known limitations, especially the land-value proxy and manual PLU/PLUi inputs.
