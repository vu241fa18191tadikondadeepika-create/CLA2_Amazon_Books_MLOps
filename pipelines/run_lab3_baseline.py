import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preprocess import main as preprocess
from src.train import train
from src.evaluate import evaluate
print('[INFO] ========================================='); print('[INFO] Starting Lab 3: Baseline ML Pipeline'); print('[INFO] =========================================')
preprocess(); train(); evaluate(); print('[SUCCESS] Lab 3 Pipeline fully executed!')
