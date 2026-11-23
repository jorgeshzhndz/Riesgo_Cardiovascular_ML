import sys, pickle
import pandas as pd

sys.path.append("../src")
from riesgo import ClasificadorRiesgo
from sklearn.metrics import roc_auc_score

with open("../models/final_model.pkl", "rb") as f:
    modelo = pickle.load(f)

test = pd.read_csv("../data/test/test.csv")
X_test, y_test = test.drop(columns="ecv"), test["ecv"]

res = modelo.predict_riesgo(X_test)
res["ECV_Real"] = y_test.values

print("ROC AUC:", round(roc_auc_score(y_test, res["score"]), 3))
print(res.groupby("nivel")["ECV_Real"].agg(["count", "sum", "mean"]))