"""Foreshadow v2 forecast engine.

Ingests numerical weather prediction runs (GFS today) into a per-run array store,
then serves point forecasts and map tiles straight from memory-mapped arrays.

    python -m forecast.ingest --loop            # keep the store up to date
    uvicorn forecast.api:app --port 5003        # serve /v2
"""

__version__ = "2.0.0.dev0"
