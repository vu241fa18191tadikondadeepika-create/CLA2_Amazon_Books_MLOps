from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "Data.csv"
PROCESSED_DATA = ROOT / "data" / "processed" / "processed.csv"
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
MLRUNS = ROOT / "mlruns"
RANDOM_STATE = 42
TARGET = "User Rating"
REQUIRED_COLUMNS = ["Name", "Author", "User Rating", "Reviews", "Price", "Year", "Genre"]
