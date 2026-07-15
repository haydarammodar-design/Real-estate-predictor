.PHONY: run test data models manifest release

run:
	python3.11 -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload

test:
	python3.11 -m compileall -q src
	python3.11 -m unittest discover -s tests -v

data:
	python3.11 src/dvf_downloader.py --years 2021 2022 2023 2024 2025 --output data/dvf/cleaned_final.parquet

models:
	python3.11 src/train.py --rebuild-features
	python3.11 src/train_apartment.py
	python3.11 src/train_land.py

manifest:
	python3.11 src/create_release_manifest.py

release: test models manifest
