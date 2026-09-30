import json
from .preprocess import load_data
from .config import REQUIRED_COLUMNS, ARTIFACTS

def validate():
    df=load_data(); ARTIFACTS.mkdir(exist_ok=True)
    errors=[]
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: errors.append(f"Missing columns: {missing}")
    if df.duplicated().sum(): errors.append(f"Duplicate rows: {df.duplicated().sum()}")
    if df["User Rating"].isna().sum(): errors.append("Missing target values")
    if df["Genre"].dropna().isin(["Fiction","Non Fiction"]).all()==False: errors.append("Unexpected Genre values")
    report={"records":len(df),"columns":list(df.columns),"errors":errors,"passed":not errors}
    (ARTIFACTS/"schema_validation_report.json").write_text(json.dumps(report,indent=2))
    if errors: raise ValueError("Schema Validation FAILED: " + "; ".join(errors))
    print(f"[SUCCESS] Schema Validation PASSED. Records: {len(df)}")
    return df
if __name__=='__main__': validate()
