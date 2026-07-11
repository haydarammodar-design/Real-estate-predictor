@echo off
cd /d E:\real-estate-predictor
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
pause
