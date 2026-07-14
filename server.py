"""Anomaly-scoring microservice.

Small FastAPI service that scores incoming feature vectors against a
pre-fit IsolationForest model. Used by the ingestion pipeline to flag
outlier records before they hit the warehouse.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anomaly-service")

app = FastAPI(title="Anomaly Scoring Service", version="0.3.1")

# Model is fit lazily on first use from a small reference sample.
_model: IsolationForest | None = None
_N_FEATURES = 8


class ScoreRequest(BaseModel):
    vectors: List[List[float]] = Field(..., description="Batch of feature vectors")


class ScoreResponse(BaseModel):
    scores: List[float]
    anomalies: List[bool]


def _get_model() -> IsolationForest:
    global _model
    if _model is None:
        logger.info("Fitting reference IsolationForest model")
        rng = np.random.default_rng(seed=42)
        reference = rng.normal(size=(512, _N_FEATURES))
        _model = IsolationForest(n_estimators=128, random_state=42)
        _model.fit(reference)
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if not req.vectors:
        raise HTTPException(status_code=400, detail="no vectors provided")

    arr = np.asarray(req.vectors, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != _N_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"each vector must have exactly {_N_FEATURES} features",
        )

    model = _get_model()
    raw = model.score_samples(arr)
    preds = model.predict(arr)
    return ScoreResponse(
        scores=[float(s) for s in raw],
        anomalies=[bool(p == -1) for p in preds],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
