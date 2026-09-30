# CLA 2 Lab Mapping

| CLA 2 requirement | Amazon project implementation |
|---|---|
| Lab 3 baseline ML pipeline | `src/preprocess.py`, `src/train.py`, `src/evaluate.py`, `pipelines/run_lab3_baseline.py` |
| Lab 4 MLflow tracking | `src/train_mlflow.py`, `src/validate_reproducibility.py`, `pipelines/run_lab4_tracking.py` |
| Lab 5 data validation + reproducible preprocessing | `src/validate_data.py`, `src/preprocess_pipeline.py`, `src/validate_outputs.py`, `pipelines/run_lab5_pipeline.py` |
| Lab 6 model registry | `src/register_model.py`, `pipelines/run_lab6_registry.py` |
| Git/version-control workflow | README Git commands + `.gitignore` + modular source tree |

The project keeps the original Amazon book-rating objective and extends it rather than creating a separate ML problem.
