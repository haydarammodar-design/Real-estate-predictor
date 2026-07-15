# AlfaScript Real Estate Feasibility Platform

AlfaScript is a French real-estate analysis prototype for historical transaction analysis, residential price estimates, indicative residual land-value references, cadastral lookup, development-feasibility scenarios, and investment calculations.

## Scope And Limits

- Coverage: metropolitan French DVF transaction data, currently 2021–2025.
- Property valuations: residential house and apartment estimates from DVF, INSEE, construction-cost, time, and nearby-comparable features.
- Land reference: an indicative residual land value derived from house transaction prices less benchmark construction cost. It is not a cadastral valuation.
- Development feasibility: CES and floor-area-ratio inputs are supplied by the user from the applicable PLU/PLUi or planning certificate. The application does not verify planning rules or issue permit advice.
- Financial outputs: indicative scenarios only. They are not investment, tax, legal, engineering, or planning advice.

## Local Run

```bash
python3.11 -m pip install -r requirements.lock
python3.11 -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

Required local artifacts:

- `data/dvf/cleaned_final.parquet` (preferred release snapshot; `cleaned.parquet` is accepted for a standard rebuild)
- `data/insee/insee_communes.parquet`
- `models/model.joblib`
- `models/model_apartment.joblib`
- `models/model_land.joblib`

These files are deliberately excluded from Git. Deliver them with the release manifest or retrieve them from approved artifact storage.

## Rebuild Data And Models

```bash
python3.11 src/dvf_downloader.py --years 2021 2022 2023 2024 2025 --output data/dvf/cleaned_final.parquet
python3.11 src/train.py --rebuild-features
python3.11 src/train_apartment.py
python3.11 src/train_land.py
python3.11 src/create_release_manifest.py
```

The release training protocol is a point-in-time backtest:

- Comparable-sale features for 2024 use transactions before 2024.
- Comparable-sale features for 2025 use transactions before 2025.
- Models are evaluated on 2025, then retrained through 2025 for deployment.

## Tests

```bash
python3.11 -m unittest discover -s tests -v
python3.11 -m compileall -q src
```

Equivalent Make targets are available: `make run`, `make test`, `make data`, `make models`, and `make manifest`.

## Deployment

Use the supplied `Dockerfile` with a long-running container platform such as Hugging Face Spaces, Cloud Run, or a VPS. Set `ALFASCRIPT_CACHE_DIR` to a writable cache directory in serverless/container environments.

Vercel is not supported for the full ML service because model/data startup and in-memory comparable indexing exceed the intended serverless execution model.

### Optional Cesium 3D Tiles

To enable the detailed Cesium 3D tileset, create a **new read-only Cesium Ion token** restricted to the deployed domain and asset, then start the server with:

```bash
export CESIUM_ION_TOKEN="your-restricted-token"
export CESIUM_ION_ASSET_ID="2275207"
python3.11 -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

The token is intentionally not stored in Git. Without it, the 3D button falls back to the public OSM globe.

## Release Evidence

See `docs/final_delivery_checklist.md` for the contract delivery matrix and acceptance tests. Generate `models/release_manifest.json` before handoff to record artifact hashes, training data coverage, metrics, and compatibility versions.
