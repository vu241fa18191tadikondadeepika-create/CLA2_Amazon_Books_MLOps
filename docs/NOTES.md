# CLA 2 Viva / Record Notes

## Lab 3
Baseline pipeline converts the notebook workflow into separate preprocessing, training and evaluation modules. The model is serialized with joblib and an error-analysis file is generated.

## Lab 4
MLflow records model parameters, MAE, RMSE, R2, data-lineage metadata and the trained model artifact. Repeated runs use `random_state=42` and the same split so reproducibility can be checked.

## Lab 5
Schema validation checks required columns, duplicates, target completeness and allowed Genre values. A reusable sklearn `Pipeline` performs imputation, scaling and one-hot encoding. Output validation verifies equal transformed feature dimensions for train and test data.

## Lab 6
The Random Forest is logged as an MLflow registered model named `AmazonBookRatingModel`. The registry provides model-version traceability and supports later promotion/rollback workflows.

## Original CLA 1 model
The original report states that Linear Regression, Decision Tree and Random Forest were compared, with Random Forest reported as the strongest model on the held-out test set (R2 0.2543, MAE 0.1390, RMSE 0.2068).
