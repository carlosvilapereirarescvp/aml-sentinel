"""
build_features.py
==================
Semana 3. Combina las features de nodo del dataset Elliptic (locales +
agregadas del vecindario a 1 salto, ya presentes) con features de grafo
calculadas directamente sobre la topología: grado, centralidad y tamaño de
la componente conexa a la que pertenece cada transacción.

La idea: las features "agregadas" que trae Elliptic ya resumen el vecindario
a 1 salto, pero no capturan estructura global (soy parte de un cluster
gigante o de uno mínimo, soy un hub o un nodo periférico). Eso es justo lo
que un LLM/analista de compliance evaluaría a ojo si mirara el grafo, y es
lo que agregamos aquí antes del baseline.

Uso:
    python src/features/build_features.py
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
GRAPH_PATH = PROCESSED_DIR / "elliptic_graph.gpickle"
NODES_PATH = PROCESSED_DIR / "nodes.parquet"
OUTPUT_PATH = PROCESSED_DIR / "nodes_with_graph_features.parquet"


def load_graph_and_nodes() -> tuple[nx.DiGraph, pd.DataFrame]:
    if not GRAPH_PATH.exists() or not NODES_PATH.exists():
        raise FileNotFoundError(
            "Faltan data/processed/elliptic_graph.gpickle o nodes.parquet. "
            "Corre primero src/graph/build_graph.py."
        )
    with open(GRAPH_PATH, "rb") as f:
        g = pickle.load(f)
    nodes = pd.read_parquet(NODES_PATH)
    logger.info("Grafo cargado: %d nodos, %d aristas", g.number_of_nodes(), g.number_of_edges())
    return g, nodes


def compute_graph_features(g: nx.DiGraph) -> pd.DataFrame:
    """Calcula features topológicas por nodo. Se evita centralidad exacta
    (betweenness/closeness) sobre los 203k nodos completos porque es O(N*E)
    y no escala; se usan aproximaciones y métricas locales que sí escalan."""
    logger.info("Calculando grados...")
    in_degree = dict(g.in_degree())
    out_degree = dict(g.out_degree())
    total_degree = {n: in_degree.get(n, 0) + out_degree.get(n, 0) for n in g.nodes()}

    logger.info("Calculando tamaño de componente débilmente conexa...")
    component_size = {}
    for component in nx.weakly_connected_components(g):
        size = len(component)
        for node in component:
            component_size[node] = size

    logger.info("Calculando PageRank (aproximación de importancia global, escala bien)...")
    pagerank = nx.pagerank(g, alpha=0.85, max_iter=100)

    logger.info("Calculando clustering local (proporción de vecinos conectados entre sí)...")
    undirected = g.to_undirected()
    clustering = nx.clustering(undirected)

    df = pd.DataFrame(
        {
            "tx_id": list(g.nodes()),
        }
    )
    df["in_degree"] = df["tx_id"].map(in_degree).fillna(0)
    df["out_degree"] = df["tx_id"].map(out_degree).fillna(0)
    df["total_degree"] = df["tx_id"].map(total_degree).fillna(0)
    df["component_size"] = df["tx_id"].map(component_size).fillna(1)
    df["pagerank"] = df["tx_id"].map(pagerank).fillna(0)
    df["clustering_coef"] = df["tx_id"].map(clustering).fillna(0)

    # Features derivadas simples que suelen ayudar en fraude/AML: ¿es un nodo
    # "puente" (mucho grado de entrada Y salida, típico de layering) o un
    # nodo terminal (solo entrada o solo salida, típico de mulas/cash-out)?
    df["is_bridge_like"] = ((df["in_degree"] > 0) & (df["out_degree"] > 0)).astype(int)
    df["is_terminal_like"] = ((df["in_degree"] == 0) | (df["out_degree"] == 0)).astype(int)
    df["degree_asymmetry"] = (df["out_degree"] - df["in_degree"]) / (df["total_degree"] + 1)

    logger.info("Features de grafo calculadas para %d nodos", len(df))
    return df


def main() -> None:
    g, nodes = load_graph_and_nodes()
    graph_features = compute_graph_features(g)

    merged = nodes.merge(graph_features, on="tx_id", how="left")
    logger.info("Nodos con features combinadas: %s", merged.shape)

    merged.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Guardado en %s", OUTPUT_PATH)

    feature_cols = [c for c in merged.columns if c not in ("tx_id", "time_step", "label")]
    logger.info("Total de features por nodo: %d (166 originales + %d de grafo)", len(feature_cols), len(feature_cols) - 166)


if __name__ == "__main__":
    main()
