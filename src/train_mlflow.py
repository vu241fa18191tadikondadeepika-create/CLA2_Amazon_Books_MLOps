
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np

from .preprocess import load_data, preprocess, make_model_data
from .config import RANDOM_STATE


def run_mlflow():
    print("Starting MLflow experiment tracking...")

    # Use SQLite database for MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Create or use the experiment
    mlflow.set_experiment("Amazon_Books_CLA2")

    # Load dataset
    df = load_data()

    # Preprocess dataset
    df = preprocess(df)

    # Prepare features and target
    X, y = make_model_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # Start MLflow run
    with mlflow.start_run(run_name="RandomForest"):

        # Model parameters
        n_estimators = 100

        # Log parameters
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.2)

        # Create model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE
        )

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log dataset lineage
        mlflow.log_dict(
            {
                "dataset": "Amazon Books",
                "rows": len(df),
                "features": list(X.columns)
            },
            "data_lineage.json"
        )

        # Log Random Forest model
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # Display results
        print()
        print("--- MLflow Run Results ---")
        print(f"R2   : {r2:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print("--------------------------")
        print("MLflow run successfully tracked!")

        return r2


if __name__ == "__main__":
    run_mlflow()

