"""Interactive Streamlit interface for ML Assignment 2."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "model" / "artifacts"
METRICS_PATH = PROJECT_ROOT / "metrics.csv"
DEFAULT_TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "target"


st.set_page_config(page_title="Breast Cancer Classifier Lab", page_icon="BC", layout="wide")


@st.cache_resource
def load_project_assets() -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
    """Load model artifacts and static metadata once per app session."""
    metadata_path = ARTIFACT_DIR / "metadata.json"
    if not metadata_path.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("Project artifacts are missing. Run: python model/train_models.py")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    models: dict[str, object] = {}
    for artifact_path in ARTIFACT_DIR.glob("*.joblib"):
        model_name = artifact_path.stem.replace("_", " ").title()
        # Match the artifact filename to the displayed model name from metrics.csv.
        models[model_name] = joblib.load(artifact_path)
    metrics = pd.read_csv(METRICS_PATH)
    return metadata, models, metrics


def artifact_key(model_name: str) -> str:
    """Return the display name used as the in-memory model dictionary key."""
    return model_name


def calculate_metrics(model: object, features: pd.DataFrame, labels: pd.Series) -> tuple[dict[str, object], object, object]:
    """Return metrics, predictions, and optional positive-class probabilities."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else None
    scores: dict[str, object] = {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "MCC": matthews_corrcoef(labels, predictions),
        "AUC": roc_auc_score(labels, probabilities) if probabilities is not None and labels.nunique() == 2 else None,
    }
    return scores, predictions, probabilities


def show_confusion_matrix(labels: pd.Series, predictions: object) -> None:
    """Render a readable confusion matrix for the selected model and data."""
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["malignant (0)", "benign (1)"],
        yticklabels=["malignant (0)", "benign (1)"],
        ax=axis,
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    st.pyplot(figure, use_container_width=True, clear_figure=True)


def main() -> None:
    st.title("Breast Cancer Classifier Lab")
    st.caption("ML Assignment 2: Comparative Analysis of Five Supervised Classification Models Using the UCI Breast Cancer Dataset")

    try:
        metadata, loaded_models, baseline_metrics = load_project_assets()
    except FileNotFoundError as error:
        st.error(str(error))
        st.code("python model/train_models.py")
        st.stop()

    model_names = baseline_metrics["ML Model Name"].tolist()
    model_name = st.sidebar.selectbox("Choose a classification model", model_names)
    model = loaded_models[artifact_key(model_name)]

    st.sidebar.divider()
    st.sidebar.subheader("Data input")
    uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
    st.sidebar.caption("Use the 30 feature columns in `test_data.csv`. Include `target` to calculate evaluation metrics.")
    use_example = st.sidebar.checkbox("Use bundled test_data.csv", value=uploaded_file is None)

    with st.expander("Dataset and implementation details", expanded=False):
        st.write(
            f"**Dataset:** {metadata['dataset_name']}  \n"
            f"**Rows:** {metadata['instances']} | **Features:** {len(metadata['features'])} | "
            f"**Held-out test rows:** {metadata['test_rows']}"
        )
        st.caption("Target labels: 0 = Malignant, 1 = Benign")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    elif use_example:
        data = pd.read_csv(DEFAULT_TEST_DATA_PATH)
    else:
        st.info("Upload a CSV or select the bundled test file to see predictions and model-specific results.")
        return

    feature_columns = metadata["features"]
    missing_columns = sorted(set(feature_columns) - set(data.columns))
    if missing_columns:
        st.error(f"The CSV is missing {len(missing_columns)} required feature column(s), for example: {', '.join(missing_columns[:4])}.")
        st.stop()

    features = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    labels = data[TARGET_COLUMN] if TARGET_COLUMN in data.columns else None
    st.subheader(f"Selected model: {model_name}")

    predictions = model.predict(features)
    output = data.copy()
    output["prediction"] = predictions
    output["predicted_class"] = pd.Series(predictions).map({0: "malignant", 1: "benign"})
    if hasattr(model, "predict_proba"):
        output["probability_benign"] = model.predict_proba(features)[:, 1]

    if labels is not None:
        labels = pd.to_numeric(labels, errors="coerce")
        valid_rows = labels.isin([0, 1])
        if not valid_rows.all():
            st.warning("Rows with an invalid target value were excluded from metric calculations.")
        scores, valid_predictions, _ = calculate_metrics(model, features.loc[valid_rows], labels.loc[valid_rows])
        metric_columns = st.columns(6)
        for column, (metric_name, score) in zip(metric_columns, scores.items()):
            column.metric(metric_name, "N/A" if score is None else f"{score:.4f}")

        left, right = st.columns([1.1, 0.9], gap="large", vertical_alignment="top")
        with left:
            report = classification_report(
                labels.loc[valid_rows],
                valid_predictions,
                labels=[0, 1],
                target_names=["malignant", "benign"],
                output_dict=True,
                zero_division=0,
            )
            report_table = pd.DataFrame(report).transpose().rename(
                index={
                    "malignant": "Malignant",
                    "benign": "Benign",
                    "accuracy": "Accuracy",
                    "macro avg": "Macro Average",
                    "weighted avg": "Weighted Average",
                }
            )
            report_table["support"] = report_table["support"].round().astype(int)
            with st.container(border=True):
                st.markdown("**Classification Report**")
                st.dataframe(
                    report_table.style.format(
                        {"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}
                    ),
                    use_container_width=True,
                )
        with right:
            with st.container(border=True):
                st.markdown("**Confusion Matrix**")
                show_confusion_matrix(labels.loc[valid_rows], valid_predictions)
    else:
        st.info("No `target` column was found, so the app displays predictions without evaluation metrics.")

    st.markdown("#### Predictions")
    st.dataframe(output.head(100), hide_index=True, use_container_width=True)
    st.download_button(
        "Download predictions as CSV",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="model_predictions.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Performance Comparison Across All Models")
    st.caption("Scores calculated using the held-out test dataset.")
    formatted_baseline = baseline_metrics.copy()
    for column in formatted_baseline.columns[1:]:
        formatted_baseline[column] = formatted_baseline[column].map(lambda value: f"{value:.4f}")
    st.dataframe(formatted_baseline, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
