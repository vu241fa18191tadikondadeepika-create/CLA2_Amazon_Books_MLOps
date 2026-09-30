# CLA 2 – Amazon Top 50 Books MLOps Project

This project is the **CLA 1 Amazon Top 50 Bestselling Books ML project continued for CLA 2**. It extends the original notebook with the MLOps concepts corresponding to Labs 3, 4, 5 and 6:

- **Lab 3 – Baseline ML Pipeline:** modular preprocessing, training, evaluation, error analysis and model serialization.
- **Lab 4 – MLflow Experiment Tracking & Reproducibility:** parameters, metrics, artifacts, MLflow runs and reproducibility validation.
- **Lab 5 – Data Validation & Reproducible ML Pipeline:** schema validation and a reusable sklearn preprocessing pipeline.
- **Lab 6 – Model Registry & Version Management:** register the trained model in MLflow Model Registry and manage model versions/stages through metadata and aliases.

## Important dataset note
The original uploaded CLA 1 ZIP contained `Amazon.ipynb` and the project report, but **did not contain `Data.csv`**. The notebook expects a file named `Data.csv` with the Amazon Top 50 Bestselling Books 2009–2019 dataset (550 rows and the columns `Name, Author, User Rating, Reviews, Price, Year, Genre`).

Place the original `Data.csv` at:

```text
data/raw/Data.csv
```

The project is intentionally not replacing your original dataset with fabricated data.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Run CLA 2 labs

### Lab 3
```powershell
python .\pipelines\run_lab3_baseline.py
```

### Lab 4
```powershell
python .\pipelines\run_lab4_tracking.py
```

### Lab 5
```powershell
python .\pipelines\run_lab5_pipeline.py
```

### Lab 6
```powershell
python .\pipelines\run_lab6_registry.py
```

## MLflow UI

After Lab 4 or Lab 6:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Open the local MLflow URL shown in the terminal.

## Git workflow demonstration

```powershell
git init
git add .
git commit -m "CLA2 Amazon books MLOps project"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Use your own GitHub repository URL. Do not commit `venv/`, `mlruns/`, large generated artifacts, or secrets.
