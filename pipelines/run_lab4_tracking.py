import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preprocess import main
from src.train_mlflow import run_mlflow
from src.validate_reproducibility import run_once
print('[INFO] ========================================='); print('[INFO] Starting Lab 4: MLflow Experiment Tracking'); print('[INFO] =========================================')
main(); run_mlflow(); a,b=run_once(),run_once(); print(f'Execution 1 R2: {a:.6f}'); print(f'Execution 2 R2: {b:.6f}'); print('[SUCCESS] Lab 4 Pipeline fully executed!')
