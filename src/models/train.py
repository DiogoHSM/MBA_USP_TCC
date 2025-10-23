"""Training utilities for the SDSS classification notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data.prepare import (
    FEATURE_COLUMNS,
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
    prepare_dataset,
)

MODEL_PATH = Path("models/hist_gradient_boosting.joblib")
SCALER_PATH = Path("models/minmax_scaler.joblib")
SPLIT_DATA_PATH = Path("data/processed/sdss_split.joblib")
METRICS_PATH = Path("reports/metrics.json")


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the processed dataset, creating it first if needed."""

    if not path.exists():
        prepare_dataset(output_path=path)
    return pd.read_csv(path)


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Generate a deterministic train-test split with stratification."""

    X = df[list(FEATURE_COLUMNS)]
    y = df[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[HistGradientBoostingClassifier, MinMaxScaler]:
    """Train the default HistGradientBoosting model used in the notebooks."""

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = HistGradientBoostingClassifier(random_state=1)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def save_artifacts(
    model: HistGradientBoostingClassifier,
    scaler: MinMaxScaler,
    split_data: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    """Persist models, scalers, splits and metrics to disk."""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(split_data, SPLIT_DATA_PATH)
    pd.Series(metrics).to_json(METRICS_PATH, indent=2)


def main() -> None:
    df = load_processed_data()
    X_train, X_test, y_train, y_test = split_dataset(df)
    model, scaler = train_model(X_train, y_train)

    X_test_scaled = scaler.transform(X_test)
    accuracy = float(model.score(X_test_scaled, y_test))

    artifacts = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    metrics = {"accuracy": accuracy}
    save_artifacts(model, scaler, artifacts, metrics)

    print(
        "Modelo treinado com acurácia",
        f"{accuracy:.4f}",
        "e salvo em",
        MODEL_PATH,
    )


if __name__ == "__main__":
    main()
