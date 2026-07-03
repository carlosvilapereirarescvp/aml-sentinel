"""
train_baseline.py
==================
Semana 4. Entrena un LightGBM sobre las features combinadas (nodo + grafo),
respetando el split temporal ya definido, y evalúa con métricas apropiadas
para el fuerte desbalance de clases (AUC-PR y F1 de la clase minoritaria,
NUNCA accuracy). Registra el experimento en MLflow y calcula importancias
SHAP como base para la explicabilidad de cada alerta.

Uso:
    python src/models/train_baseline.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES_PATH = PROCESSED_DIR / "nodes_with_graph_features.parquet"

# Mismo corte temporal que en build_graph.py: train<=34, val 35-42, test>42.
TRAIN_MAX_STEP = 34
VAL_MAX_STEP = 42


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Falta {FEATURES_PATH}. Corre primero src/features/build_features.py."
        )
    df = pd.read_parquet(FEATURES_PATH)
    labeled = df[df["label"] != -1].copy()

    feature_cols = [c for c in labeled.columns if c not in ("tx_id", "time_step", "label")]

    train = labeled[labeled["time_step"] <= TRAIN_MAX_STEP]
    val = labeled[(labeled["time_step"] > TRAIN_MAX_STEP) & (labeled["time_step"] <= VAL_MAX_STEP)]
    test = labeled[labeled["time_step"] > VAL_MAX_STEP]

    logger.info("Train: %d | Val: %d | Test: %d | Features: %d", len(train), len(val), len(test), len(feature_cols))
    return train, val, test, feature_cols


def train_model(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list[str]) -> lgb.Booster:
    train_set = lgb.Dataset(train[feature_cols], label=train["label"])
    val_set = lgb.Dataset(val[feature_cols], label=val["label"], reference=train_set)

    # scale_pos_weight compensa el desbalance (clase ilícita minoritaria)
    # sin necesidad de undersampling, que perdería información del grafo.
    n_pos = (train["label"] == 1).sum()
    n_neg = (train["label"] == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info("scale_pos_weight = %.2f (n_neg=%d, n_pos=%d)", scale_pos_weight, n_neg, n_pos)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "scale_pos_weight": scale_pos_weight,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)],
    )
    return model


def evaluate(model: lgb.Booster, test: pd.DataFrame, feature_cols: list[str]) -> dict:
    y_true = test["label"].values
    y_proba = model.predict(test[feature_cols])
    y_pred = (y_proba >= 0.5).astype(int)

    auc_pr = average_precision_score(y_true, y_proba)
    auc_roc = roc_auc_score(y_true, y_proba)
    f1_illicit = f1_score(y_true, y_pred, pos_label=1)

    logger.info("AUC-PR (test): %.4f", auc_pr)
    logger.info("AUC-ROC (test): %.4f", auc_roc)
    logger.info("F1 clase ilícita (test): %.4f", f1_illicit)
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=["licito", "ilicito"]))

    return {"auc_pr": auc_pr, "auc_roc": auc_roc, "f1_illicit": f1_illicit}


def compute_shap_importance(model: lgb.Booster, test: pd.DataFrame, feature_cols: list[str], top_n: int = 20) -> pd.DataFrame:
    """SHAP sobre una muestra de test (calcularlo sobre todo el set es costoso
    y no aporta más para un reporte de importancia global)."""
    sample = test.sample(min(2000, len(test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[feature_cols])

    importance = pd.DataFrame(
        {"feature": feature_cols, "mean_abs_shap": abs(shap_values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)

    logger.info("Top %d features por SHAP:\n%s", top_n, importance.head(top_n).to_string(index=False))
    return importance


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("aml-sentinel-baseline")

    train, val, test, feature_cols = load_split()

    with mlflow.start_run(run_name="lightgbm_baseline"):
        model = train_model(train, val, feature_cols)
        metrics = evaluate(model, test, feature_cols)
        mlflow.log_metrics(metrics)
        mlflow.log_params({"model_type": "lightgbm", "n_features": len(feature_cols)})

        importance = compute_shap_importance(model, test, feature_cols)
        importance_path = MODELS_DIR / "shap_importance_baseline.csv"
        importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path))

        model_path = MODELS_DIR / "lightgbm_baseline.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        logger.info("Modelo guardado en %s", model_path)


if __name__ == "__main__":
    main()
