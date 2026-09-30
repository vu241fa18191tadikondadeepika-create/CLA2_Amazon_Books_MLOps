import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .preprocess import load_data, preprocess, make_model_data
from .config import RANDOM_STATE, ARTIFACTS, MODELS

def evaluate():
    df = preprocess(load_data())
    X, y = make_model_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    bundle = joblib.load(MODELS / "random_forest_baseline.joblib")
    pred = bundle["model"].predict(X_test)
    metrics = {"MAE": float(mean_absolute_error(y_test, pred)), "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))), "R2": float(r2_score(y_test, pred))}
    errors = X_test.copy(); errors["actual"] = y_test.values; errors["predicted"] = pred; errors["absolute_error"] = abs(errors["actual"]-errors["predicted"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    errors.sort_values("absolute_error", ascending=False).head(20).to_csv(ARTIFACTS / "error_analysis.csv", index=False)
    print("\n--- Model Evaluation Report ---")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"R2   : {metrics['R2']:.4f}")
    print("-------------------------------")
    return metrics

if __name__ == "__main__":
    print("Starting Model Evaluation...")
    evaluate(); print("Evaluation complete and error analysis files saved!")
