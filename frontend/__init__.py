"""Frontend package — the React research app and its FastAPI JSON backend.

The browser app lives in ``frontend/web`` (Vite + React + TypeScript). The JSON API that
serves it lives in ``frontend/api`` (FastAPI). Run the API from the repo root with::

    .venv/Scripts/python.exe -m uvicorn frontend.api.server:app --reload --port 5057
"""
