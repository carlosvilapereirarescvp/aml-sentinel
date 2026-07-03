# AML Sentinel

Sistema de detección de lavado de dinero sobre una red de transacciones de
Bitcoin (dataset [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)),
con explicabilidad de cada alerta y generación automática de un borrador de
reporte de investigación (SAR) usando la API de Claude con RAG sobre
documentación de FATF/FINMA.

> 🚧 Proyecto en construcción activa — roadmap de 8 semanas. Estado actual: **Semana 1-2, ingesta y EDA.**

## Por qué este proyecto

Suiza está actualizando su marco de lucha contra el blanqueo de capitales
para 2026, exigiendo a bancos y aseguradoras un monitoreo de transacciones
más basado en datos y en riesgo. Este proyecto simula, con datos reales, el
tipo de sistema que un banco o regulador (FINMA) necesitaría: no solo un
modelo que puntúa transacciones, sino una herramienta end-to-end que un
analista de compliance podría usar en su día a día.

## Arquitectura

1. **Ingesta y grafo** — dataset Elliptic (203k transacciones, 234k aristas) → NetworkX + PyTorch Geometric.
2. **Modelado** — baseline LightGBM (con SHAP) + GNN (GraphSAGE) sobre el grafo, comparados con MLflow.
3. **Explicabilidad** — SHAP para el baseline, GNNExplainer para el GNN.
4. **Generación de SAR** — Claude API + RAG (ChromaDB) sobre guías FATF/FINMA, para redactar el borrador del reporte de cada alerta.
5. **Servicio** — FastAPI (scoring, explicación, generación de SAR) + Streamlit (dashboard) + Docker + CI/CD.

## Estado del roadmap

| Semana | Entregable | Estado |
|---|---|---|
| 1-2 | Ingesta, construcción del grafo, EDA | ✅ scaffold listo |
| 3 | Feature engineering, split temporal | ⏳ |
| 4 | Baseline LightGBM + SHAP, métrica AUC-PR | ⏳ |
| 5 | GNN GraphSAGE + MLflow | ⏳ |
| 6 | Ensemble + GNNExplainer | ⏳ |
| 7 | FastAPI + generación de SAR con Claude | ⏳ |
| 8 | Streamlit + Docker + CI/CD + demo | ⏳ |

## Cómo correrlo

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Descargar el dataset Elliptic siguiendo data/README.md
# 2. Construir el grafo
python src/graph/build_graph.py

# 3. Abrir el EDA
jupyter notebook notebooks/01_eda.ipynb

# 4. Correr los tests
pytest tests/ -v
```

## Estructura del repo

```
aml-sentinel/
├── data/             # instrucciones de descarga + raw/ processed/
├── notebooks/        # EDA, análisis de features, comparativa de modelos
├── src/
│   ├── graph/        # construcción del grafo PyG   <- HECHO
│   ├── features/     # feature engineering            (Semana 3)
│   ├── models/        # GNN, LightGBM, ensemble        (Semanas 4-6)
│   ├── explainability/# SHAP, GNNExplainer             (Semana 4-6)
│   ├── api/            # FastAPI                        (Semana 7)
│   └── sar/            # generador de SAR con Claude    (Semana 7)
├── dashboard/        # Streamlit app                    (Semana 8)
├── tests/
├── docker-compose.yml  (Semana 8)
├── Dockerfile           (Semana 8)
└── .github/workflows/ci.yml (Semana 8)
```
