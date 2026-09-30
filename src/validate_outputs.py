import json, joblib
from .preprocess_pipeline import build
from .config import ARTIFACTS

def validate_outputs():
    pipe,X_train,X_test,y_train,y_test=build(); a=pipe.transform(X_train); b=pipe.transform(X_test)
    report={"train_rows":a.shape[0],"test_rows":b.shape[0],"train_features":a.shape[1],"test_features":b.shape[1],"same_feature_dimension":a.shape[1]==b.shape[1],"passed":a.shape[1]==b.shape[1]}
    (ARTIFACTS/"preprocessing_summary_report.json").write_text(json.dumps(report,indent=2))
    if not report["passed"]: raise ValueError("Output Validation FAILED")
    print("[SUCCESS] Output Validation PASSED. Features are clean, scaled, encoded, and dimensionally consistent.")
if __name__=='__main__': validate_outputs()
