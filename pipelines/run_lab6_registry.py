import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.register_model import register
print('[INFO] ========================================='); print('[INFO] Starting Lab 6: Model Registry and Version Management'); print('[INFO] =========================================')
register(); print('[SUCCESS] Lab 6 Model Registry pipeline fully executed!')
