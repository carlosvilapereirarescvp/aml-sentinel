"""
build_graph.py
==============
Ingesta del dataset Elliptic y construcción del grafo de transacciones.

Lee los 3 CSV crudos de data/raw/, los une, arma un grafo dirigido con
NetworkX y lo exporta en dos formatos:
  - data/processed/elliptic_graph.gpickle   -> grafo NetworkX (para EDA)
  - data/processed/elliptic_pyg.pt          -> objeto torch_geometric.data.Data
                                                (para entrenar el GNN en la
                                                Semana 5)

Uso:
    python src/graph/build_graph.py
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURES_FILE = RAW_DIR / "elliptic_txs_features.csv"
CLASSES_FILE = RAW_DIR / "elliptic_txs_classes.csv"
EDGES_FILE = RAW_DIR / "elliptic_txs_edgelist.csv"

# El paper original de Elliptic (Weber et al., 2019) no incluye nombres de
# columnas. Col 0 = txId, Col 1 = time step, cols 2-95 = features locales,
# cols 96-167 = features agregadas del vecindario a 1 salto.
N_LOCAL_FEATURES = 93
N_AGG_FEATURES = 72
FEATURE_COLUMNS = ["tx_id", "time_step"] + [f"local_{i}" for i in range(N_LOCAL_FEATURES)] + [
    f"agg_{i}" for i in range(N_AGG_FEATURES)
]

CLASS_MAP = {"1": 1, "2": 0, "unknown": -1}  # 1=ilícito, 0=lícito, -1=sin etiqueta


def _check_raw_files() -> None:
    missing = [f for f in (FEATURES_FILE, CLASSES_FILE, EDGES_FILE) if not f.exists()]
    if missing:
        names = "\n  - ".join(str(m) for m in missing)
        raise FileNotFoundError(
            f"Faltan archivos crudos del dataset Elliptic:\n  - {names}\n\n"
            f"Sigue las instrucciones en data/README.md para descargarlos "
            f"desde Kaggle y colocarlos en {RAW_DIR}/"
        )


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y une features + clases en un único DataFrame de nodos."""
    _check_raw_files()

    logger.info("Cargando features de %s", FEATURES_FILE)
    features = pd.read_csv(FEATURES_FILE, header=None, names=FEATURE_COLUMNS)

    logger.info("Cargando etiquetas de %s", CLASSES_FILE)
    classes = pd.read_csv(CLASSES_FILE)
    classes.columns = [c.strip().lower() for c in classes.columns]
    classes = classes.rename(columns={"txid": "tx_id", "class": "class_raw"})
    classes["label"] = classes["class_raw"].astype(str).map(CLASS_MAP)

    nodes = features.merge(classes[["tx_id", "label"]], on="tx_id", how="left")
    nodes["label"] = nodes["label"].fillna(-1).astype(int)

    logger.info("Cargando aristas de %s", EDGES_FILE)
    edges = pd.read_csv(EDGES_FILE)
    edges.columns = [c.strip().lower() for c in edges.columns]
    edges = edges.rename(columns={edges.columns[0]: "src", edges.columns[1]: "dst"})

    logger.info(
        "Nodos: %d | Aristas: %d | Ilícitos: %d | Lícitos: %d | Sin etiqueta: %d",
        len(nodes),
        len(edges),
        (nodes["label"] == 1).sum(),
        (nodes["label"] == 0).sum(),
        (nodes["label"] == -1).sum(),
    )
    return nodes, edges


def build_networkx_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.DiGraph:
    """Construye un grafo dirigido NetworkX, útil para EDA e inspección visual."""
    g = nx.DiGraph()
    for row in nodes.itertuples(index=False):
        g.add_node(row.tx_id, time_step=row.time_step, label=row.label)
    g.add_edges_from(edges[["src", "dst"]].itertuples(index=False, name=None))
    logger.info("Grafo NetworkX construido: %d nodos, %d aristas", g.number_of_nodes(), g.number_of_edges())
    return g


def build_pyg_data(nodes: pd.DataFrame, edges: pd.DataFrame) -> Data:
    """Construye un objeto torch_geometric.data.Data listo para entrenar un GNN."""
    tx_id_to_idx = {tx_id: i for i, tx_id in enumerate(nodes["tx_id"])}

    feature_cols = [c for c in nodes.columns if c.startswith("local_") or c.startswith("agg_")]
    x = torch.tensor(nodes[feature_cols].values, dtype=torch.float)

    y = torch.tensor(nodes["label"].values, dtype=torch.long)

    # Filtrar aristas cuyos extremos no aparezcan en la tabla de nodos (raro, pero defensivo)
    valid_edges = edges[edges["src"].isin(tx_id_to_idx) & edges["dst"].isin(tx_id_to_idx)]
    src_idx = valid_edges["src"].map(tx_id_to_idx).values
    dst_idx = valid_edges["dst"].map(tx_id_to_idx).values
    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)

    time_step = torch.tensor(nodes["time_step"].values, dtype=torch.long)

    # Máscaras de train/val/test por time_step: 1-34 train, 35-42 val, 43-49 test.
    # Split temporal, no aleatorio: evita leakage y refleja cómo se usaría en producción.
    labeled_mask = y != -1
    train_mask = labeled_mask & (time_step <= 34)
    val_mask = labeled_mask & (time_step > 34) & (time_step <= 42)
    test_mask = labeled_mask & (time_step > 42)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.time_step = time_step
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    logger.info(
        "Split temporal -> train: %d | val: %d | test: %d (nodos etiquetados)",
        train_mask.sum().item(),
        val_mask.sum().item(),
        test_mask.sum().item(),
    )
    return data


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nodes, edges = load_raw()

    nx_graph = build_networkx_graph(nodes, edges)
    nx_path = PROCESSED_DIR / "elliptic_graph.gpickle"
    with open(nx_path, "wb") as f:
        pickle.dump(nx_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Grafo NetworkX guardado en %s", nx_path)

    pyg_data = build_pyg_data(nodes, edges)
    pyg_path = PROCESSED_DIR / "elliptic_pyg.pt"
    torch.save(pyg_data, pyg_path)
    logger.info("Objeto PyG guardado en %s", pyg_path)

    nodes.to_parquet(PROCESSED_DIR / "nodes.parquet", index=False)
    edges.to_parquet(PROCESSED_DIR / "edges.parquet", index=False)
    logger.info("Tablas nodes/edges guardadas en formato parquet para el EDA")


if __name__ == "__main__":
    main()
