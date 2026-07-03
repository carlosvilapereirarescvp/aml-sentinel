"""
sar_generator.py
=================
Semana 7. Genera un borrador de Suspicious Activity Report (SAR) para una
transacción marcada como sospechosa, usando la API de Claude con contexto
recuperado (RAG) de guías FATF/FINMA sobre indicadores de lavado de dinero,
más los datos concretos de la alerta (score, features influyentes, subgrafo
de vecinos según GNNExplainer).

Requiere ANTHROPIC_API_KEY en el entorno (.env).

Uso (standalone, para probar):
    python src/sar/sar_generator.py --tx-id 12345
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
KB_DIR = Path(__file__).resolve().parents[2] / "src" / "sar" / "knowledge_base"

SYSTEM_PROMPT = """Eres un asistente que ayuda a analistas de compliance de un \
banco suizo a redactar el BORRADOR INICIAL de un Suspicious Activity Report \
(SAR) para una transacción de Bitcoin marcada por un sistema de detección de \
lavado de dinero.

Reglas estrictas:
- Esto es un BORRADOR para revisión humana, nunca un reporte final. Dilo \
explícitamente al final del documento.
- Basa el análisis únicamente en los datos de la alerta que se te dan \
(score del modelo, features influyentes, estructura del subgrafo). No \
inventes detalles de identidad, jurisdicción, ni montos que no estén en \
los datos.
- Usa un tono formal y objetivo, como escribiría un analista humano.
- Estructura el reporte con: (1) Resumen de la alerta, (2) Indicadores \
técnicos detectados, (3) Contexto de red/grafo, (4) Recomendación de \
siguiente paso (ninguna decisión definitiva, solo sugerencias de \
investigación adicional).
"""


def load_knowledge_base_context() -> str:
    """Carga fragmentos cortos de guías FATF/FINMA como contexto fijo.
    Para una versión productiva esto se reemplaza por retrieval real desde
    ChromaDB; aquí se deja el contexto embebido para simplificar el demo
    inicial y que el pipeline funcione end-to-end sin depender de tener la
    base vectorial poblada."""
    default_context = (
        "Indicadores típicos de layering en cripto según guías FATF: "
        "(a) transacciones fragmentadas en montos pequeños hacia múltiples "
        "direcciones en corto tiempo ('smurfing'), (b) rutas de varios saltos "
        "entre una dirección de origen ilícita conocida y un exchange de "
        "cash-out, (c) nodos con alta centralidad que actúan como mezcladores "
        "o puentes entre clusters, (d) patrones de entrada/salida asimétricos "
        "consistentes con consolidación de fondos previa a cash-out."
    )
    if KB_DIR.exists():
        texts = []
        for f in sorted(KB_DIR.glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
        if texts:
            return "\n\n".join(texts)
    return default_context


def build_alert_payload(tx_id: int) -> dict:
    """Reúne los datos concretos de la alerta desde los artefactos generados
    en la Semana 6 (scores del ensemble + explicación GNNExplainer)."""
    import pandas as pd

    scores_path = MODELS_DIR / "ensemble_scores.parquet"
    if not scores_path.exists():
        raise FileNotFoundError(f"Falta {scores_path}. Corre primero src/models/ensemble_and_explain.py.")

    scores = pd.read_parquet(scores_path)
    row = scores[scores["tx_id"] == tx_id]
    if row.empty:
        raise ValueError(f"tx_id {tx_id} no encontrado en ensemble_scores.parquet")
    row = row.iloc[0]

    explanation_path = MODELS_DIR / "example_explanation.json"
    explanation = {}
    if explanation_path.exists():
        explanation = json.loads(explanation_path.read_text())

    return {
        "tx_id": int(tx_id),
        "time_step": int(row["time_step"]),
        "lgb_proba": round(float(row["lgb_proba"]), 4),
        "gnn_proba": round(float(row["gnn_proba"]), 4),
        "ensemble_proba": round(float(row["ensemble_proba"]), 4),
        "graph_explanation": explanation,
    }


def generate_sar(tx_id: int, model: str = "claude-sonnet-4-6") -> str:
    alert = build_alert_payload(tx_id)
    kb_context = load_knowledge_base_context()

    user_prompt = f"""Contexto regulatorio (FATF/FINMA):
{kb_context}

Datos de la alerta a redactar:
{json.dumps(alert, indent=2)}

Redacta el borrador de SAR siguiendo la estructura indicada."""

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx-id", type=int, required=True)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(
            "Falta ANTHROPIC_API_KEY. Crea un archivo .env en la raíz del repo "
            "con la línea: ANTHROPIC_API_KEY=tu_key_aqui"
        )

    sar_text = generate_sar(args.tx_id)
    print(sar_text)


if __name__ == "__main__":
    main()
