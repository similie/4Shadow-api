# 4Shadow forecast service: the API, and (same image, another command) the GFS ingest.
# See docker-compose.yml to run both, sharing one data volume, and docs/deploy.md.
#
#   docker build -t 4shadow-forecast .
#   docker run -p 5003:5003 -v forecast-data:/data 4shadow-forecast                                   # API
#   docker run -v forecast-data:/data 4shadow-forecast python -m forecast.ingest --max-fhr 24        # some data
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORECAST_DATA_ROOT=/data/forecast \
    FORECAST_PLACES_DB=/data/places/places.sqlite

WORKDIR /app
COPY requirements.txt ./
COPY forecast/requirements.txt forecast/
RUN pip install -r requirements.txt

COPY forecast forecast
RUN useradd --system --uid 10001 --home-dir /app forecast \
 && mkdir -p /data/forecast /data/places \
 && chown -R forecast /data
USER forecast
VOLUME /data
EXPOSE 5003

# /v2 answers as soon as the API is up; /v2/health is 503 until the first run is ingested.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5003/v2', timeout=4)"

CMD ["uvicorn", "forecast.api:app", "--host", "0.0.0.0", "--port", "5003", "--workers", "2"]
