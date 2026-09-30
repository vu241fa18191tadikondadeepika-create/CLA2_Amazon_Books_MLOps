import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from .validate_data import validate
from .config import MODELS, RANDOM_STATE

def build():
    df=validate(); X=df[["Reviews","Price","Year","Genre"]]; y=df["User Rating"]
    num=["Reviews","Price","Year"]; cat=["Genre"]
    pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    pipe=Pipeline([("preprocessor",pre)])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE)
    pipe.fit(X_train,y_train); MODELS.mkdir(exist_ok=True); joblib.dump(pipe,MODELS/"preprocessor.pkl")
    print("[SUCCESS] Preprocessing pipeline saved as preprocessor.pkl")
    return pipe,X_train,X_test,y_train,y_test
if __name__=='__main__': build()
