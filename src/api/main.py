"""
main.py
=======
Semana 7. API FastAPI con 3 endpoints:
  - GET  /health                 -> chequeo de vida
  - GET  /score/{tx_id}          -> score del ensemble para una transacción
  - GET  /explain/{tx_id}        -> features/vecinos más influyentes (GNNExplainer, precomputado)
  - POST /sar/{tx_id}            -> genera el borrador de SAR con Claude API

Uso:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.sar.sar_generator import generate_sar

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SCORES_PATH = MODELS_DIR / "ensemble_scores.parquet"

app = FastAPI(
    title="AML Sentinel API",
    description="Detección de lavado de dinero sobre transacciones de Bitcoin (dataset Elliptic).",
    version="0.1.0",
)

_scores_cache: pd.DataFrame | None = None


def get_scores() -> pd.DataFrame:
    global _scores_cache
    if _scores_cache is None:
        if not SCORES_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Scores no disponibles. Corre src/models/ensemble_and_explain.py primero.",
            )
        _scores_cache = pd.read_parquet(SCORES_PATH)
    return _scores_cache


class ScoreResponse(BaseModel):
    tx_id: int
    time_step: int
    lgb_proba: float
    gnn_proba: float
    ensemble_proba: float
    is_alert: bool


class SarResponse(BaseModel):
    tx_id: int
    sar_draft: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/score/{tx_id}", response_model=ScoreResponse)
def score(tx_id: int) -> ScoreResponse:
    scores = get_scores()
    row = scores[scores["tx_id"] == tx_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"tx_id {tx_id} no encontrado")
    row = row.iloc[0]
    return ScoreResponse(
        tx_id=int(row["tx_id"]),
        time_step=int(row["time_step"]),
        lgb_proba=round(float(row["lgb_proba"]), 4),
        gnn_proba=round(float(row["gnn_proba"]), 4),
        ensemble_proba=round(float(row["ensemble_proba"]), 4),
        is_alert=bool(row["ensemble_proba"] >= 0.5),
    )


@app.get("/explain/{tx_id}")
def explain(tx_id: int) -> dict:
    import json

    explanation_path = MODELS_DIR / "example_explanation.json"
    if not explanation_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Explicación no disponible. Corre src/models/ensemble_and_explain.py primero.",
        )
    # Nota: en esta versión demo la explicación es del ejemplo precomputado
    # en la Semana 6. Una versión productiva calcularía GNNExplainer on-demand
    # por tx_id, con un cache LRU dado su costo computacional.
    return json.loads(explanation_path.read_text())


@app.post("/sar/{tx_id}", response_model=SarResponse)
def create_sar(tx_id: int) -> SarResponse:
    scores = get_scores()
    if scores[scores["tx_id"] == tx_id].empty:
        raise HTTPException(status_code=404, detail=f"tx_id {tx_id} no encontrado")
    try:
        sar_text = generate_sar(tx_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error generando SAR para tx_id=%s", tx_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SarResponse(tx_id=tx_id, sar_draft=sar_text)
