"""
train_gnn.py
============
Semana 5. Entrena una GraphSAGE de 2 capas sobre el objeto PyG construido en
build_graph.py, usando message passing real (no solo features tabulares como
el baseline) para capturar patrones de propagación típicos de layering en
lavado de dinero (una transacción "limpia" en aislamiento puede ser
sospechosa por estar conectada a un cluster ilícito).

Uso:
    python src/models/train_gnn.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.nn import SAGEConv

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PYG_PATH = PROCESSED_DIR / "elliptic_pyg.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return self.classifier(x).squeeze(-1)


def load_data():
    if not PYG_PATH.exists():
        raise FileNotFoundError(f"Falta {PYG_PATH}. Corre primero src/graph/build_graph.py.")
    data = torch.load(PYG_PATH, weights_only=False)
    return data.to(DEVICE)


def train_one_epoch(model, data, optimizer, pos_weight) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    loss = F.binary_cross_entropy_with_logits(
        out[data.train_mask],
        data.y[data.train_mask].float(),
        pos_weight=pos_weight,
    )
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask) -> dict:
    model.eval()
    out = model(data.x, data.edge_index)
    proba = torch.sigmoid(out[mask]).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    y_pred = (proba >= 0.5).astype(int)

    return {
        "auc_pr": average_precision_score(y_true, proba),
        "auc_roc": roc_auc_score(y_true, proba),
        "f1_illicit": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def main(epochs: int = 100, patience: int = 15) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("aml-sentinel-gnn")

    data = load_data()
    logger.info("Datos cargados en %s | x: %s | edge_index: %s", DEVICE, tuple(data.x.shape), tuple(data.edge_index.shape))

    n_pos = (data.y[data.train_mask] == 1).sum().item()
    n_neg = (data.y[data.train_mask] == 0).sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    logger.info("pos_weight = %.2f", pos_weight.item())

    model = GraphSAGE(in_channels=data.x.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    with mlflow.start_run(run_name="graphsage"):
        mlflow.log_params({"model_type": "graphsage", "hidden_channels": 128, "lr": 0.01, "epochs": epochs})

        best_val_auc_pr = 0.0
        epochs_no_improve = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            loss = train_one_epoch(model, data, optimizer, pos_weight)
            val_metrics = evaluate(model, data, data.val_mask)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %03d | loss %.4f | val AUC-PR %.4f | val F1 %.4f",
                    epoch, loss, val_metrics["auc_pr"], val_metrics["f1_illicit"],
                )

            if val_metrics["auc_pr"] > best_val_auc_pr:
                best_val_auc_pr = val_metrics["auc_pr"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info("Early stopping en epoch %d (sin mejora en %d epochs)", epoch, patience)
                break

        model.load_state_dict(best_state)
        test_metrics = evaluate(model, data, data.test_mask)
        logger.info("Test final -> AUC-PR: %.4f | AUC-ROC: %.4f | F1 ilícito: %.4f",
                    test_metrics["auc_pr"], test_metrics["auc_roc"], test_metrics["f1_illicit"])
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        model_path = MODELS_DIR / "graphsage.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("Modelo guardado en %s", model_path)


if __name__ == "__main__":
    main()
