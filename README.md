# ESP Predictive Health

Python and Streamlit application for ESP predictive health, failure diagnosis, and future live monitoring.

## Phase 1: Case Management

Phase 1 provides a SQLite-backed register of confirmed ESP/well cases and a searchable Case Library. Raw uploaded datasets are copied into `data/raw/` and are never overwritten when case metadata changes.

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run esp_predictive_health/app.py
```

The local SQLite database is created at `data/cases/cases.db` on first use.
