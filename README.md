# ML Assignment 2 - Breast Cancer Classification

## Problem statement

Build and compare five supervised machine-learning classifiers for predicting whether a breast tumour is **malignant (0)** or **benign (1)** from computed diagnostic measurements. The project also provides a Streamlit interface where a user can choose a model, upload compatible test data, inspect the required evaluation metrics, view a confusion matrix and download predictions.

## Dataset description

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The dataset contains 569 labelled observations (212 malignant and 357 benign) and 30 continuous diagnostic features calculated from a digitised fine-needle aspirate image of a breast mass. It therefore exceeds the assignment requirement of at least 500 instances and 12 features.

The data is loaded reproducibly through `sklearn.datasets.load_breast_cancer`. A stratified 70:30 train/test split with `random_state=42` is used. The held-out 171-row test partition is saved in [test_data.csv](test_data.csv); it includes all 30 feature columns plus the `target` column.

> This model is prepared for an academic demonstration only. It must not be used for medical diagnosis or patient-care decisions.

## Repository structure

```text
.
|-- app.py                    # Streamlit application
|-- requirements.txt          # Deployment dependencies
|-- README.md                 # Project documentation
|-- test_data.csv             # Reproducible held-out test data
|-- metrics.csv               # Five-model comparison table
`-- model/
    |-- train_models.py       # Training, evaluation, and model persistence
    `-- artifacts/            # Generated .joblib models and metadata
```

## Models used and evaluation metrics

All models use the identical training and test partitions. Missing numeric values are median-imputed within each model pipeline; algorithms that depend on comparable feature scales also use standardisation. The required metrics are Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

| ML model name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9883 | 0.9981 | 0.9907 | 0.9907 | 0.9907 | 0.9750 |
| Decision Tree | 0.9357 | 0.9545 | 0.9444 | 0.9533 | 0.9488 | 0.8623 |
| K-Nearest Neighbors | 0.9649 | 0.9958 | 0.9469 | 1.0000 | 0.9727 | 0.9264 |
| Gaussian Naive Bayes | 0.9357 | 0.9892 | 0.9364 | 0.9626 | 0.9493 | 0.8620 |
| Random Forest (Ensemble) | 0.9415 | 0.9914 | 0.9450 | 0.9626 | 0.9537 | 0.8746 |

`metrics.csv` is produced by the training script, and the Streamlit app presents the same comparison table together with fresh metrics for the selected model on any uploaded labelled test data.

## Observations about model performance

| ML model name | Observation to include after training |
|---|---|
| Logistic Regression | This was the strongest model on this split: it achieved the best accuracy (0.9883), AUC (0.9981), F1 (0.9907) and MCC (0.9750). This suggests a well-scaled linear boundary separates the classes effectively. |
| Decision Tree | It is the most interpretable model but returned a lower MCC (0.8623) and AUC (0.9545), suggesting that a single bounded-depth tree does not capture all relevant relationships. |
| K-Nearest Neighbors | It achieved perfect recall (1.0000), so no benign cases were missed on this test split. Its slightly lower precision and MCC than logistic regression indicate some additional false-positive predictions. |
| Gaussian Naive Bayes | It achieved a high AUC (0.9892) but lower accuracy and MCC than logistic regression. The conditional-independence assumption is restrictive because several diagnostic measurements are correlated. |
| Random Forest (Ensemble) | The ensemble improved on the single decision tree in accuracy, AUC, F1 and MCC, demonstrating the benefit of aggregating many trees. It still did not exceed logistic regression on this dataset split. |
| Overall winner | **Logistic Regression** is the winner because it gives the most balanced result across all six required metrics, with the highest accuracy, AUC, F1 and MCC. |

## Run locally

1. Create and activate a virtual environment (recommended).
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train and save all five models, metrics and test data:

   ```bash
   python model/train_models.py
   ```

4. Start the interactive application:

   ```bash
   streamlit run app.py
   ```

5. In the sidebar, select a model and either upload a compatible CSV or use the bundled `test_data.csv`. A labelled file (with `target`) shows all required metrics, a confusion matrix and a classification report. An unlabelled file returns predictions only.

## Streamlit deployment

1. Push this folder, including generated `model/artifacts/`, `metrics.csv` and `test_data.csv`, to a new GitHub repository.
2. Visit [Streamlit Community Cloud](https://streamlit.io/cloud), sign in with GitHub and select **New app**.
3. Choose the repository and branch, set the main file path to `app.py`, then click **Deploy**.
4. Copy the deployed app URL below after verifying that the upload, model selector, metrics and confusion matrix all work.

## Mandatory submission links

- GitHub repository: **[replace with your GitHub repository URL]**
- Live Streamlit app: **[replace with your Streamlit app URL]**
- BITS Virtual Lab proof: add one screenshot of your own assignment execution to the final submission PDF.

## Reproducibility notes

- `random_state=42` and a stratified split make the experiment repeatable.
- The test CSV must retain the exact 30 feature names in `test_data.csv`.
- Do not upload sensitive real-world medical data to the demo application.
