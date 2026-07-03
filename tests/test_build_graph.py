"""
Tests para src/graph/build_graph.py usando un mini-dataset sintético
(no requiere el Elliptic real) para validar la lógica de unión,
mapeo de etiquetas y construcción del split temporal.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.build_graph import (  # noqa: E402
    N_AGG_FEATURES,
    N_LOCAL_FEATURES,
    build_networkx_graph,
    build_pyg_data,
)


@pytest.fixture
def toy_nodes() -> pd.DataFrame:
    n_feat = N_LOCAL_FEATURES + N_AGG_FEATURES
    rows = []
    # 5 nodos: 2 ilícitos, 2 lícitos, 1 sin etiqueta; distribuidos en distintos time_steps
    labels = [1, 0, 1, 0, -1]
    steps = [1, 1, 40, 45, 45]
    for i, (label, step) in enumerate(zip(labels, steps)):
        row = {"tx_id": i, "time_step": step, "label": label}
        for j in range(n_feat):
            row[("local_" if j < N_LOCAL_FEATURES else "agg_") + str(j if j < N_LOCAL_FEATURES else j - N_LOCAL_FEATURES)] = 0.1 * j
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def toy_edges() -> pd.DataFrame:
    return pd.DataFrame({"src": [0, 1, 2], "dst": [1, 2, 3]})


def test_networkx_graph_shape(toy_nodes, toy_edges):
    g = build_networkx_graph(toy_nodes, toy_edges)
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 3
    assert g.nodes[0]["label"] == 1


def test_pyg_data_shapes(toy_nodes, toy_edges):
    data = build_pyg_data(toy_nodes, toy_edges)
    assert data.x.shape[0] == 5
    assert data.x.shape[1] == N_LOCAL_FEATURES + N_AGG_FEATURES
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] == 3


def test_temporal_split_no_leakage(toy_nodes, toy_edges):
    """El split debe respetar los cortes de time_step: train<=34, val 35-42, test>42,
    y nunca debe incluir nodos sin etiqueta (-1)."""
    data = build_pyg_data(toy_nodes, toy_edges)

    assert data.train_mask.sum().item() == 2  # steps 1,1 -> ambos etiquetados
    assert data.val_mask.sum().item() == 1    # step 40
    assert data.test_mask.sum().item() == 1   # step 45 etiquetado (el otro step 45 es -1)

    # Ningún nodo sin etiqueta debe caer en ninguna máscara
    unlabeled_idx = (data.y == -1).nonzero(as_tuple=True)[0]
    for idx in unlabeled_idx:
        assert not data.train_mask[idx]
        assert not data.val_mask[idx]
        assert not data.test_mask[idx]
