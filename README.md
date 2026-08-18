# ML Assignment 2 - Breast Cancer Classification

## a. Problem Statement

The objective of this assignment is to build and compare five supervised machine-learning classification models for predicting whether a breast tumour is **malignant (0)** or **benign (1)** from diagnostic measurements. The project also includes a Streamlit web application that allows a user to select a model, upload test data, view the required evaluation metrics, inspect a confusion matrix and classification report, and download predictions.

## b. Dataset Description

This project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). It contains 569 labelled instances: 212 malignant and 357 benign. Each instance has 30 continuous diagnostic features calculated from a digitised fine-needle aspirate image of a breast mass.

The dataset satisfies the assignment requirements because it contains more than 500 instances and more than 12 features. The data is loaded reproducibly through `sklearn.datasets.load_breast_cancer`, then split into training and test sets using a stratified 70:30 split with `random_state=42`. The held-out test data contains 171 rows and is available in `test_data.csv`.

## c. GitHub Repository Link

[https://github.com/Reethu-Kondepudi/2025ac05754-ml-assignment-2](https://github.com/Reethu-Kondepudi/2025ac05754-ml-assignment-2)

**Live Streamlit App:** [https://2025ac05754-ml-assignment-2-j2vycwf5kzfdcl738cnvqx.streamlit.app/](https://2025ac05754-ml-assignment-2-j2vycwf5kzfdcl738cnvqx.streamlit.app/)

## d. Models Used

The following five classification models are implemented using the same dataset and the same train/test split:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (kNN)
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier (Ensemble)

### Comparison Table of Evaluation Metrics

All models are evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9883 | 0.9981 | 0.9907 | 0.9907 | 0.9907 | 0.9750 |
| Decision Tree Classifier | 0.9357 | 0.9545 | 0.9444 | 0.9533 | 0.9488 | 0.8623 |
| K-Nearest Neighbor Classifier (kNN) | 0.9649 | 0.9958 | 0.9469 | 1.0000 | 0.9727 | 0.9264 |
| Gaussian Naive Bayes Classifier | 0.9357 | 0.9892 | 0.9364 | 0.9626 | 0.9493 | 0.8620 |
| Random Forest Classifier (Ensemble) | 0.9415 | 0.9914 | 0.9450 | 0.9626 | 0.9537 | 0.8746 |

### Observations About Model Performance

| ML Model Name | Observation About Model Performance |
|---|---|
| Logistic Regression | This was the strongest model on the held-out test set. It achieved the highest Accuracy (0.9883), AUC (0.9981), F1 Score (0.9907), and MCC (0.9750), indicating that the two classes are well separated by a scaled linear model. |
| Decision Tree Classifier | This model is easy to interpret, but its lower AUC (0.9545) and MCC (0.8623) indicate that one bounded-depth tree did not capture all important relationships in the data. |
| K-Nearest Neighbor Classifier (kNN) | kNN achieved perfect Recall (1.0000), meaning it identified every benign case in this test split. Its Precision and MCC were slightly lower than Logistic Regression because of a small number of additional false-positive predictions. |
| Gaussian Naive Bayes Classifier | Gaussian Naive Bayes achieved a high AUC (0.9892), but lower Accuracy and MCC than Logistic Regression. This is expected because several diagnostic features are correlated, which conflicts with the model's conditional-independence assumption. |
| Random Forest Classifier (Ensemble) | The Random Forest improved on the single Decision Tree in Accuracy, AUC, F1 Score, and MCC. Combining multiple trees made it more robust, although it did not outperform Logistic Regression on this test split. |
| Overall Winner for This Dataset | **Logistic Regression** is the overall winner because it achieved the most balanced performance across all required metrics, including the highest Accuracy, AUC, F1 Score, and MCC. |
