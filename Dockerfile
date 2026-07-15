FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV ALFASCRIPT_CACHE_DIR=/tmp/alfascript

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=5)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]
