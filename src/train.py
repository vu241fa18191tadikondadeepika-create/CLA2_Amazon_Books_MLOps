import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from .preprocess import load_data, preprocess, make_model_data
from .config import RANDOM_STATE, MODELS

def train():
    df = preprocess(load_data())
    X, y = make_model_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns)}, MODELS / "random_forest_baseline.joblib")
    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Starting Model Training...")
    train()
    print("Model training complete and saved to disk!")
