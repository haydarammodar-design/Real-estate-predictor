# Technical Appendix

## Architecture

```text
DVF 2021-2025 + INSEE commune data + construction benchmarks
                         |
                         v
          Cleaned Parquet and point-in-time features
                         |
                         v
  XGBoost property, apartment, and residual-land-reference models
                         |
                         v
        FastAPI API and single-page Leaflet/Cesium interface
                         |
                         v
 Valuation, parcel lookup, feasibility scenario, financial analysis
```

## Release Validation Protocol

- 2024 is the model-development training year.
- 2025 is the untouched evaluation year.
- 2024 comparable features use only sales before 1 January 2024.
- 2025 comparable features use only sales before 1 January 2025.
- After evaluation, the deployment model trains on 2024 and 2025 observations using their original point-in-time features.
- Outlier limits are fitted on 2024 only and applied unchanged to 2025.

## Development Feasibility Formulas

- Maximum CES footprint = parcel area × CES.
- Maximum gross floor area = parcel area × gross-floor-area ratio.
- Buildable footprint = minimum of CES footprint and gross floor area divided by number of floors.
- Gross floor area = buildable footprint × number of floors.
- Saleable area = gross floor area × saleable ratio.
- Total cost = land purchase + land acquisition fees + construction cost + additional costs.
- Sales revenue = saleable area × declared sale price per saleable m².
- Developer margin = sales revenue − total cost.
- Margin percentage = developer margin ÷ sales revenue.
- ROI = developer margin ÷ total cost.

The calculation is a scenario using user-declared planning rules. It is not a planning, permit, structural-engineering, tax, or legal opinion.

## External Services

| Service | Use | Operational note |
|---|---|---|
| DVF / data.gouv.fr | Historic transactions | Record retrieval date and source coverage in the release manifest. |
| INSEE / geo.api.gouv.fr | Commune socioeconomic features | Version and validate data coverage for every model release. |
| IGN APICarto | Parcel lookup | Availability is external; handle request failures in the UI. |
| OpenStreetMap, Nominatim, Overpass | Maps, search, nearby places | Observe provider usage policies, caching, attribution, and rate limits. |
| Construction benchmarks | Residual land proxy and feasibility defaults | Indicative inputs only; retain source and update date. |

## Deployment Requirements

- Python 3.11.
- XGBoost OpenMP runtime (`libomp-dev` in the Docker image).
- At least 2GB available memory for model and comparable-index startup.
- Writable `ALFASCRIPT_CACHE_DIR` for SQLite and map tiles.
- Versioned data/model artifacts delivered separately from Git with hashes from `models/release_manifest.json`.
