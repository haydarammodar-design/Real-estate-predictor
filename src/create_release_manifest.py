"""Create a versioned manifest for the AlfaScript data and model release."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import xgboost

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA_DIR = ROOT / "data" / "dvf"
DATA = next(
    (DATA_DIR / filename for filename in ("cleaned_final.parquet", "cleaned_release.parquet", "cleaned.parquet") if (DATA_DIR / filename).exists()),
    DATA_DIR / "cleaned.parquet",
)
OUTPUT = MODELS / "release_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Missing release dataset: {DATA}")
    df = pd.read_parquet(DATA, columns=["date_mutation", "code_departement", "type_local"])
    manifest = {
        "release_created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(DATA.relative_to(ROOT)),
            "sha256": _sha256(DATA),
            "records": int(len(df)),
            "date_start": str(pd.to_datetime(df["date_mutation"]).min().date()),
            "date_end": str(pd.to_datetime(df["date_mutation"]).max().date()),
            "departments": int(df["code_departement"].nunique()),
            "property_types": df["type_local"].value_counts().to_dict(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "xgboost": xgboost.__version__,
        },
        "models": {},
    }
    for filename in ("model.joblib", "model_apartment.joblib", "model_land.joblib"):
        path = MODELS / filename
        if not path.exists():
            continue
        artifact = joblib.load(path)
        manifest["models"][filename] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "feature_count": len(artifact.get("feature_cols", [])),
            "metrics": artifact.get("metrics", {}),
            "validation": artifact.get("validation", {}),
            "comparable_cutoff": artifact.get("comparable_cutoff"),
            "target_definition": artifact.get("target_definition", "property transaction price"),
        }
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Release manifest written to {OUTPUT}")


if __name__ == "__main__":
    main()
