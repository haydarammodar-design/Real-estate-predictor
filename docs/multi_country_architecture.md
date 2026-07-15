# Multi-Country Data Architecture

The current implementation is France-only because DVF, INSEE commune codes, French departments, and IGN Cadastre are country-specific. The extension boundary is explicit:

| Layer | France implementation | Country adapter responsibility |
|---|---|---|
| Transaction source | `src/dvf_downloader.py` | Download and normalise local transaction registry data. |
| Geographic identifier | INSEE commune and department codes | Supply stable municipality/region identifiers and coordinate validation. |
| Socioeconomic enrichment | `src/insee_collector.py` | Provide population, income, and deprivation fields with source metadata. |
| Parcel lookup | IGN APICarto | Provide parcel area, geometry, and source availability/error handling. |
| Planning rules | User-declared PLU/PLUi inputs | Supply verified zoning/coverage only where legally and technically available. |
| Construction benchmarks | `src/construction_benchmarks.py` | Supply local currency, tax basis, construction type, source date, and regional factors. |
| Models | Shared feature names in `src/data_processor.py` | Retrain a country-specific model; never reuse French coefficients or benchmarks abroad. |

## Minimum Adapter Contract

Each country adapter must deliver:

- Transaction fields: price, date, property type, built area, land area, rooms, latitude, longitude, local region code, and municipality code.
- Data lineage: source URL, licence, retrieval date, schema version, coverage, cleaner version, and checksum.
- Currency and tax-basis definition.
- Local validation ranges for coordinates, areas, and property categories.
- A country-specific temporal backtest and model card.
- A documented planning-data policy. No planning rights may be inferred where verified source data is unavailable.
