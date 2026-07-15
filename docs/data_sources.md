# Data Sources And Governance Register

| Source | Purpose | Release control |
|---|---|---|
| DGFiP DVF via `files.data.gouv.fr/geo-dvf` | Historical residential transactions | Record source URL, retrieval date, department/year coverage, cleaner version, record count, and SHA-256 in the release manifest. Confirm the applicable licence before redistribution. |
| INSEE FILoSoFi and `geo.api.gouv.fr` | Commune income, poverty, population, and location data | Record source vintage, retrieval date, merge coverage, and missing-value strategy. |
| IGN APICarto Cadastre | Parcel reference, area, and centroid lookup | Runtime external dependency only. Parcel data does not supply zoning or planning permission. |
| OpenStreetMap tiles, Nominatim, Overpass | Base map, address search, and nearby places | Retain attribution, observe rate limits and usage policies, cache responsibly, and do not use a browser-set User-Agent as an identity mechanism. |
| CesiumJS CDN | Optional 3D presentation | No access token or private asset is embedded in the application. |
| Construction benchmarks | Residual land proxy and feasibility defaults | `src/construction_benchmarks.py` contains indicative department factors. Replace with validated quantity-surveyor quotations for decision use. |

## Data Boundaries

- DVF is transaction evidence, not a complete description of property condition, floor, energy performance, zoning, title restrictions, or development rights.
- The application must not claim to verify PLU/PLUi, CES, FAR/COS, permits, utilities, flood risk, contamination, or constructibility. Those inputs remain user-declared and require professional verification.
- Personal or confidential company data must not be inserted into public model artifacts, logs, maps, or demo inputs.
