import joblib
import pandas as pd
from sklift.models import SoloModel
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from sklift.datasets import fetch_x5

data = fetch_x5()
X = data.data
y = data.target
treatment = data.treatment

features = X.columns
X_train, X_test, y_train, y_test, treat_train, treat_test = train_test_split(
    X, y, treatment, test_size=0.2, random_state=42
)

model = CatBoostClassifier(verbose=0, random_state=42)
uplift_model = SoloModel(estimator=model)
uplift_model.fit(X_train, y_train, treat_train)

joblib.dump(uplift_model, "model/uplift_model.pkl")
