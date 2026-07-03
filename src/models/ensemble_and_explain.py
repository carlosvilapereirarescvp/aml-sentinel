"""
ensemble_and_explain.py
========================
Semana 6. Combina las probabilidades del baseline LightGBM (features
tabulares) y de la GraphSAGE (estructura de grafo) en un ensemble simple por
promedio ponderado, y agrega GNNExplainer para poder explicar CUÁLES nodos
vecinos influyeron en la predicción de una transacción puntual — esto es
justo lo que un analista de compliance necesita ver para escribir el SAR.

Uso:
    python src/models/ensemble_and_explain.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.explain import Explainer, GNNExplainer

from src.models.train_gnn import GraphSAGE, DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES_PATH = PROCESSED_DIR / "nodes_with_graph_features.parquet"
PYG_PATH = PROCESSED_DIR / "elliptic_pyg.pt"

ENSEMBLE_WEIGHT_GNN = 0.5  # peso del GNN vs. LightGBM en el ensemble (empieza 50/50)


def load_baseline_scores() -> pd.DataFrame:
    model = joblib.load(MODELS_DIR / "lightgbm_baseline.pkl")
    df = pd.read_parquet(FEATURES_PATH)
    labeled = df[df["label"] != -1].copy()
    feature_cols = [c for c in labeled.columns if c not in ("tx_id", "time_step", "label")]
    labeled["lgb_proba"] = model.predict(labeled[feature_cols])
    return labeled[["tx_id", "time_step", "label", "lgb_proba"]]


def load_gnn_scores() -> pd.DataFrame:
    data = torch.load(PYG_PATH, weights_only=False)
    model = GraphSAGE(in_channels=data.x.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "graphsage.pt", map_location=DEVICE))
    model.eval()

    data = data.to(DEVICE)
    with torch.no_grad():
        proba = torch.sigmoid(model(data.x, data.edge_index)).cpu().numpy()

    df = pd.read_parquet(FEATURES_PATH)[["tx_id"]]
    df["gnn_proba"] = proba
    return df


def build_ensemble() -> pd.DataFrame:
    baseline = load_baseline_scores()
    gnn = load_gnn_scores()
    merged = baseline.merge(gnn, on="tx_id", how="left")
    merged["ensemble_proba"] = (
        ENSEMBLE_WEIGHT_GNN * merged["gnn_proba"] + (1 - ENSEMBLE_WEIGHT_GNN) * merged["lgb_proba"]
    )
    return merged


def evaluate_ensemble(merged: pd.DataFrame) -> dict:
    test = merged[merged["time_step"] > 42]
    y_true = test["label"].values
    y_pred = (test["ensemble_proba"] >= 0.5).astype(int)

    metrics = {
        "auc_pr": average_precision_score(y_true, test["ensemble_proba"]),
        "auc_roc": roc_auc_score(y_true, test["ensemble_proba"]),
        "f1_illicit": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    logger.info(
        "Ensemble (test) -> AUC-PR: %.4f | AUC-ROC: %.4f | F1 ilícito: %.4f",
        metrics["auc_pr"], metrics["auc_roc"], metrics["f1_illicit"],
    )
    return metrics


def explain_node(node_idx: int, top_k_neighbors: int = 5) -> dict:
    """Genera una explicación GNNExplainer para un nodo puntual: qué
    sub-grafo y qué features tuvieron más peso en la predicción. Esto
    alimenta directamente al generador de SAR de la Semana 7."""
    data = torch.load(PYG_PATH, weights_only=False).to(DEVICE)
    model = GraphSAGE(in_channels=data.x.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / "graphsage.pt", map_location=DEVICE))
    model.eval()

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
    )

    explanation = explainer(data.x, data.edge_index, index=node_idx)

    edge_mask = explanation.edge_mask.detach().cpu().numpy()
    top_edge_idx = np.argsort(edge_mask)[-top_k_neighbors:][::-1].copy()
    influential_edges = data.edge_index[:, top_edge_idx].cpu().numpy().tolist()

    feature_mask = explanation.node_mask.detach().cpu().numpy()[node_idx]
    top_features_idx = np.argsort(feature_mask)[-10:][::-1].copy().tolist()

    return {
        "node_idx": int(node_idx),
        "influential_edges": influential_edges,
        "top_feature_indices": top_features_idx,
    }


def main() -> None:
    merged = build_ensemble()
    metrics = evaluate_ensemble(merged)

    output_path = MODELS_DIR / "ensemble_scores.parquet"
    merged.to_parquet(output_path, index=False)
    logger.info("Scores del ensemble guardados en %s", output_path)

    metrics_path = MODELS_DIR / "ensemble_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Métricas guardadas en %s", metrics_path)

    # Demo: explica el nodo ilícito con mayor score del ensemble (ejemplo
    # representativo para probar que GNNExplainer funciona end-to-end)
    top_illicit = merged[merged["label"] == 1].sort_values("ensemble_proba", ascending=False).iloc[0]
    node_positional_idx = merged.index[merged["tx_id"] == top_illicit["tx_id"]][0]
    logger.info("Explicando el nodo ilícito de mayor score (tx_id=%s)...", top_illicit["tx_id"])
    explanation = explain_node(node_positional_idx)

    explanation_path = MODELS_DIR / "example_explanation.json"
    with open(explanation_path, "w") as f:
        json.dump(explanation, f, indent=2)
    logger.info("Ejemplo de explicación guardado en %s", explanation_path)


if __name__ == "__main__":
    main()
