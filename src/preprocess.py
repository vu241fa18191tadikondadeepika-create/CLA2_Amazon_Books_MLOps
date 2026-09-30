import pandas as pd
from pathlib import Path
from .config import RAW_DATA, PROCESSED_DATA, REQUIRED_COLUMNS

def load_data(path=RAW_DATA):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Place Data.csv in data/raw/")
    return pd.read_csv(path)

def preprocess(df):
    df = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.drop_duplicates().reset_index(drop=True)
    df["Book_Age"] = 2026 - df["Year"]
    df["Author_Book_Count"] = df.groupby("Author")["Name"].transform("count")
    try:
        df["Review_Category"] = pd.qcut(df["Reviews"], q=3, labels=["Low", "Medium", "High"], duplicates="drop")
    except ValueError:
        df["Review_Category"] = "Medium"
    df["Price_Category"] = pd.cut(df["Price"], bins=[-1, 10, float("inf")], labels=["Budget", "Expensive"])
    df["Name_Length"] = df["Name"].astype(str).str.len()
    df["Author_Frequency"] = df["Author"].map(df["Author"].value_counts())
    return df

def make_model_data(df):
    X = df[["Reviews", "Price", "Year", "Genre"]].copy()
    y = df["User Rating"].copy()
    X = pd.get_dummies(X, columns=["Genre"], drop_first=True, dtype=int)
    return X, y

def main():
    print("Starting Preprocessing Pipeline...")
    df = load_data()
    processed = preprocess(df)
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATA, index=False)
    print("Preprocessing completed successfully!")
    return processed

if __name__ == "__main__": main()
