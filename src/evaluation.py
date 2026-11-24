import sys
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
TEST_PATH = BASE_DIR / "data" / "test" / "test.csv"

SRC_DIR = str(BASE_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from riesgo import ClasificadorRiesgo  # noqa: E402

with open(MODEL_PATH, "rb") as f:
    modelo = pickle.load(f)

test = pd.read_csv(TEST_PATH)
X_test, y_test = test.drop(columns="ecv"), test["ecv"]

res = modelo.predict_riesgo(X_test)
res["ECV_Real"] = y_test.values

print("ROC AUC:", round(roc_auc_score(y_test, res["score"]), 3))
print(res.groupby("nivel")["ECV_Real"].agg(["count", "sum", "mean"]))