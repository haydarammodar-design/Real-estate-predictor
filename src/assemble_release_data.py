"""Assemble a validated release dataset from yearly corrected DVF snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def assemble(base_path: Path, replacement_path: Path, output_path: Path) -> None:
    base = pd.read_parquet(base_path)
    replacement = pd.read_parquet(replacement_path)
    base_dates = pd.to_datetime(base["date_mutation"], errors="coerce")
    replacement_dates = pd.to_datetime(replacement["date_mutation"], errors="coerce")
    if set(replacement_dates.dt.year.dropna().unique()) != {2024}:
        raise ValueError("Replacement file must contain only 2024 transactions")
    if replacement["code_departement"].astype(str).nunique() < 90:
        raise ValueError("Replacement file does not have national department coverage")

    final = pd.concat([base.loc[base_dates.dt.year.ne(2024)], replacement], ignore_index=True)
    final = final.drop_duplicates(subset=["id_mutation"], keep="first")
    final = final.sort_values("date_mutation").reset_index(drop=True)
    final_dates = pd.to_datetime(final["date_mutation"], errors="coerce")
    coverage = final.groupby(final_dates.dt.year)["code_departement"].nunique().to_dict()
    missing_years = {year for year in range(2021, 2026)}.difference(coverage)
    if missing_years:
        raise ValueError(f"Missing release years: {sorted(missing_years)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(output_path, index=False)
    print(f"Release dataset saved to {output_path}")
    print(f"Rows: {len(final):,}")
    print(f"Coverage by year: {coverage}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=root / "data" / "dvf" / "cleaned_release.parquet")
    parser.add_argument("--replacement", type=Path, default=root / "data" / "dvf" / "cleaned_2024_retry.parquet")
    parser.add_argument("--output", type=Path, default=root / "data" / "dvf" / "cleaned_final.parquet")
    args = parser.parse_args()
    assemble(args.base, args.replacement, args.output)
