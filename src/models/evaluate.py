"""Evaluation helpers compatible with the notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.prepare import PROCESSED_DATA_PATH, prepare_dataset
from src.models.train import MODEL_PATH, SCALER_PATH, SPLIT_DATA_PATH

DEFAULT_REPORT_PATH = Path("reports/evaluation.json")


def load_artifacts() -> tuple[object, object, dict[str, pd.DataFrame | pd.Series]]:
    """Load the persisted model, scaler and train/test split."""

    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not SPLIT_DATA_PATH.exists():
        raise FileNotFoundError(
            "É necessário treinar o modelo antes de executar a avaliação. "
            "Use `python -m src.models.train`."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    split_data = joblib.load(SPLIT_DATA_PATH)
    return model, scaler, split_data


def compute_metrics(
    model: object,
    scaler: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """Calculate the set of metrics reported in the notebooks."""

    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "roc_auc_ovr": float(roc_auc_score(y_test, y_proba, multi_class="ovr")),
    }


def ensure_processed_dataset(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Guarantee that the processed dataset exists and load it."""

    if not path.exists():
        prepare_dataset(output_path=path)
    return pd.read_csv(path)


def main(report_path: Path = DEFAULT_REPORT_PATH) -> None:
    # Garantimos que o pré-processamento já rodou para manter a coerência com os notebooks.
    ensure_processed_dataset()
    model, scaler, split_data = load_artifacts()
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    metrics = compute_metrics(model, scaler, X_test, y_test)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_json(report_path, indent=2)

    print("Relatório de avaliação salvo em", report_path)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
