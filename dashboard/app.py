"""
app.py
======
Semana 8. Dashboard Streamlit para que un analista de compliance revise
alertas del ensemble, vea las features más influyentes de una transacción,
y dispare la generación del borrador de SAR.

Uso:
    streamlit run dashboard/app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
SCORES_PATH = MODELS_DIR / "ensemble_scores.parquet"

st.set_page_config(page_title="AML Sentinel", layout="wide")
st.title("🔎 AML Sentinel — Panel de alertas")
st.caption("Detección de lavado de dinero en transacciones de Bitcoin (dataset Elliptic)")


@st.cache_data
def load_scores() -> pd.DataFrame:
    if not SCORES_PATH.exists():
        st.error(
            "No se encontraron scores. Corre `python src/models/ensemble_and_explain.py` "
            "antes de abrir este dashboard."
        )
        st.stop()
    return pd.read_parquet(SCORES_PATH)


scores = load_scores()

col1, col2, col3 = st.columns(3)
col1.metric("Transacciones evaluadas", f"{len(scores):,}")
col2.metric("Alertas (ensemble ≥ 0.5)", f"{(scores['ensemble_proba'] >= 0.5).sum():,}")
col3.metric("Tasa de alerta", f"{100 * (scores['ensemble_proba'] >= 0.5).mean():.2f}%")

st.divider()

threshold = st.slider("Umbral de alerta (ensemble_proba)", 0.0, 1.0, 0.5, 0.01)
sort_desc = st.checkbox("Ordenar por score descendente", value=True)

alerts = scores[scores["ensemble_proba"] >= threshold].sort_values(
    "ensemble_proba", ascending=not sort_desc
)
st.subheader(f"Alertas por encima del umbral ({len(alerts)})")
st.dataframe(
    alerts[["tx_id", "time_step", "lgb_proba", "gnn_proba", "ensemble_proba", "label"]].head(200),
    use_container_width=True,
)

st.divider()
st.subheader("Generar borrador de SAR")
st.caption(
    "Requiere que la API esté corriendo (`uvicorn src.api.main:app --port 8000`) "
    "y ANTHROPIC_API_KEY configurada."
)

tx_id_input = st.number_input("tx_id de la alerta", min_value=0, step=1)
if st.button("Generar SAR"):
    import requests

    with st.spinner("Generando borrador con Claude..."):
        try:
            resp = requests.post(f"http://localhost:8000/sar/{int(tx_id_input)}", timeout=60)
            resp.raise_for_status()
            st.text_area("Borrador de SAR", resp.json()["sar_draft"], height=400)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error llamando a la API: {exc}")
