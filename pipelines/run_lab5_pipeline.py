import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.validate_data import validate
from src.validate_outputs import validate_outputs
print('[INFO] ========================================='); print('[INFO] Starting Lab 5: Production Data Pipeline'); print('[INFO] =========================================')
validate(); validate_outputs(); print('[SUCCESS] Lab 5 Production Pipeline fully executed!')
