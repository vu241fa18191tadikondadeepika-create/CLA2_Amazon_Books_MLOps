from .preprocess import load_data, preprocess, make_model_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from .config import RANDOM_STATE

def run_once():
    df=preprocess(load_data()); X,y=make_model_data(df)
    a,b,c,d=train_test_split(X,y,test_size=.2,random_state=RANDOM_STATE)
    m=RandomForestRegressor(n_estimators=100,random_state=RANDOM_STATE); m.fit(a,c)
    return r2_score(d,m.predict(b))
if __name__=='__main__':
    x,y=run_once(),run_once(); print(f"Execution 1 R2: {x:.6f}"); print(f"Execution 2 R2: {y:.6f}"); print("SUCCESS: Pipeline is 100% reproducible." if x==y else "WARNING: Results differ.")
