"""Train, evaluate, and save the five classifiers used by the Streamlit app.

Run from the project root with:
    python model/train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "model" / "artifacts"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_PATH = PROJECT_ROOT / "metrics.csv"
RANDOM_STATE = 42
TARGET_COLUMN = "target"


def build_models() -> dict[str, Pipeline]:
    """Return independently configured preprocessing-and-model pipelines."""
    def numeric_pipeline(classifier: object, scale: bool = True) -> Pipeline:
        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", classifier))
        return Pipeline(steps)

    return {
        "Logistic Regression": numeric_pipeline(
            LogisticRegression(max_iter=2_000, solver="liblinear", random_state=RANDOM_STATE)
        ),
        "Decision Tree": numeric_pipeline(
            DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE),
            scale=False,
        ),
        "K-Nearest Neighbors": numeric_pipeline(KNeighborsClassifier(n_neighbors=11)),
        "Gaussian Naive Bayes": numeric_pipeline(GaussianNB()),
        "Random Forest": numeric_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            scale=False,
        ),
    }


def evaluate(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """Compute all metrics mandated in the assignment brief."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "AUC": roc_auc_score(labels, probabilities),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def main() -> None:
    """Create the reproducible test data, trained model artifacts, and metrics table."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    labels = dataset.target.copy()
    labels.name = TARGET_COLUMN

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test.to_numpy()
    test_data.to_csv(TEST_DATA_PATH, index=False)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        scores = evaluate(model, x_test, y_test)
        metric_rows.append({"ML Model Name": model_name, **scores})
        artifact_name = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, ARTIFACT_DIR / artifact_name)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(METRICS_PATH, index=False, float_format="%.4f")
    metadata = {
        "dataset_name": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "instances": int(features.shape[0]),
        "features": features.columns.tolist(),
        "target_column": TARGET_COLUMN,
        "class_labels": {"0": "malignant", "1": "benign"},
        "test_rows": int(test_data.shape[0]),
        "random_state": RANDOM_STATE,
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(metric_rows)} model artifacts to {ARTIFACT_DIR}")
    print(f"Saved test data ({len(test_data)} rows) to {TEST_DATA_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
