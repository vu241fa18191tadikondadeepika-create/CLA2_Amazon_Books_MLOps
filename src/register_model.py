
import mlflow
from pathlib import Path


def register():
    print("Starting MLflow Model Registry...")

    # Use the same SQLite MLflow database used in Lab 4
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Use the existing experiment
    mlflow.set_experiment("Amazon_Books_CLA2")

    # Get the latest run from the experiment
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name("Amazon_Books_CLA2")

    if experiment is None:
        print("Experiment not found.")
        print("Please run Lab 4 first.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        print("No MLflow runs found.")
        print("Please run Lab 4 first.")
        return

    latest_run = runs[0]

    run_id = latest_run.info.run_id

    print(f"Latest MLflow Run ID: {run_id}")

    # Model artifact location created by Lab 4
    model_uri = f"runs:/{run_id}/model"

    print(f"Model URI: {model_uri}")

    # Register model
    model_name = "Amazon_Books_RandomForest"

    try:
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        print()
        print("--- Model Registry Results ---")
        print(f"Model Name    : {result.name}")
        print(f"Model Version : {result.version}")
        print(f"Run ID        : {run_id}")
        print("------------------------------")
        print("Model registered successfully!")

    except Exception as e:
        print(f"Model registration error: {e}")
        raise


if __name__ == "__main__":
    register()

