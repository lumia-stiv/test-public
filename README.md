# Anomaly Scoring Service

A small FastAPI microservice that scores feature vectors against a pre-fit
`IsolationForest` model. Part of the ingestion pipeline — flags outlier records
before they reach the warehouse.

## Requirements

- Python 3.11+
- Scientific stack: `numpy`, `scipy`, `scikit-learn`

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python server.py
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/score \
  -H 'content-type: application/json' \
  -d '{"vectors": [[0,0,0,0,0,0,0,0]]}'
```

## Endpoints

- `GET /health` — liveness check
- `POST /score` — batch anomaly scoring (8-feature vectors)
