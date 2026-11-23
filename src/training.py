"""Entrenamiento de los 5 modelos supervisados y guardado de resultados.

Lee data/processed/ecv_procesado.csv, hace el split train/test, entrena y
afina los modelos, guarda train/test en data/, los modelos en models/ y la
tabla de resultados en docs/. El modelo final es la regresion logistica
envuelta en ClasificadorRiesgo (niveles bajo / moderado / alto).

Uso (desde la raiz del proyecto):
    python src/training.py
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from riesgo import ClasificadorRiesgo

RAIZ = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
SCORING = "f1"
CV = 5
CORTES = (0.2, 0.5)
MODELO_FINAL = "Regresión Logística"

# ---------------------------------------------------------------- columnas
COLS_RIESGO = ["tension_alta", "colesterol_alto", "diabetes", "epoc", "problemas_rinon",
               "migrana", "tumor_maligno", "depresion", "ansiedad", "otros_problemas_mentales"]
COLS_DIETA = ["dieta_fruta", "dieta_verdura", "dieta_legumbres", "dieta_pescado",
              "dieta_carne_procesada", "dieta_dulces", "dieta_refrescos",
              "dieta_comida_rapida", "dieta_aperitivos"]
COLS_NUM = ["edad", "imc", "sentado_h", "cigarrillos_dia",
            "dias_camina", "dias_deporte", "dias_fuerza"]
COLS_BIN = ["mujer"]
COLS_CAT = (COLS_RIESGO + COLS_DIETA +
            ["nivel_estudios", "clase_social", "ejercicio_ocio", "severidad_depresiva",
             "apoyo_personas", "fuma", "frec_alcohol", "frec_atracon_alcohol",
             "actividad_trabajo", "vapeo", "humo_ajeno", "imc_categoria", "ccaa"])


# ------------------------------------------------------------------- datos
def cargar_y_dividir():
    """Carga los datos procesados, divide en train/test y guarda ambos."""
    df = pd.read_csv(RAIZ / "data" / "processed" / "ecv_procesado.csv")
    print("Datos:", df.shape)
    print(df["ecv"].value_counts(normalize=True).round(3))

    # 'factor' es el peso muestral de la encuesta: no es un predictor
    X = df.drop(columns=["ecv", "factor"])
    y = df["ecv"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print("Train:", X_train.shape, "| Test:", X_test.shape)

    (RAIZ / "data" / "train").mkdir(parents=True, exist_ok=True)
    (RAIZ / "data" / "test").mkdir(parents=True, exist_ok=True)
    X_train.assign(ecv=y_train).to_csv(RAIZ / "data" / "train" / "train.csv", index=False)
    X_test.assign(ecv=y_test).to_csv(RAIZ / "data" / "test" / "test.csv", index=False)
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------ preprocesado
def construir_preprocesador():
    rama_num = Pipeline([("imputar", SimpleImputer(strategy="median")),
                         ("escalar", StandardScaler())])
    rama_cat = Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", rama_num, COLS_NUM),
                              ("bin", "passthrough", COLS_BIN),
                              ("cat", rama_cat, COLS_CAT)])


# ----------------------------------------------------------------- modelos
def definir_modelos(y_train):
    """Devuelve {nombre: estimador sin entrenar} en el orden de trained_model_1..5."""
    prep = construir_preprocesador()
    peso_positivo = (y_train == 0).sum() / (y_train == 1).sum()

    pipe_lr = Pipeline([("prep", prep),
                        ("modelo", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                      random_state=RANDOM_STATE))])

    pipe_dt = Pipeline([("prep", prep),
                        ("modelo", DecisionTreeClassifier(class_weight="balanced",
                                                          random_state=RANDOM_STATE))])
    grid_dt = GridSearchCV(pipe_dt,
                           {"modelo__max_depth": [3, 5, 8, 12],
                            "modelo__min_samples_leaf": [5, 20, 50]},
                           cv=CV, scoring=SCORING, n_jobs=-1)

    pipe_rf = Pipeline([("prep", prep),
                        ("modelo", RandomForestClassifier(class_weight="balanced",
                                                          random_state=RANDOM_STATE))])
    grid_rf = GridSearchCV(pipe_rf,
                           {"modelo__n_estimators": [100, 200],
                            "modelo__max_depth": [5, 10, None],
                            "modelo__min_samples_leaf": [1, 5, 10]},
                           cv=CV, scoring=SCORING, n_jobs=-1)

    pipe_xgb = Pipeline([("prep", prep),
                         ("modelo", XGBClassifier(scale_pos_weight=peso_positivo,
                                                  eval_metric="logloss",
                                                  random_state=RANDOM_STATE))])
    rand_xgb = RandomizedSearchCV(pipe_xgb,
                                  {"modelo__n_estimators": [100, 200, 300],
                                   "modelo__max_depth": [3, 5, 7],
                                   "modelo__learning_rate": [0.01, 0.05, 0.1],
                                   "modelo__subsample": [0.7, 0.9, 1.0]},
                                  n_iter=10, cv=CV, scoring=SCORING,
                                  random_state=RANDOM_STATE, n_jobs=-1)

    pipe_knn = Pipeline([("prep", prep),
                         ("modelo", KNeighborsClassifier())])
    grid_knn = GridSearchCV(pipe_knn,
                            {"modelo__n_neighbors": [5, 15, 31],
                             "modelo__weights": ["uniform", "distance"]},
                            cv=CV, scoring=SCORING, n_jobs=-1)

    return {"Regresión Logística": pipe_lr,
            "Árbol de decisión": grid_dt,
            "Random Forest": grid_rf,
            "XGBoost": rand_xgb,
            "KNN": grid_knn}


def entrenar(modelos, X_train, y_train):
    """Entrena cada modelo y devuelve {nombre: mejor estimador ya ajustado}."""
    entrenados = {}
    for nombre, m in modelos.items():
        print(f"\nEntrenando: {nombre}")
        if nombre == "Regresión Logística":
            f1 = cross_val_score(m, X_train, y_train, cv=CV, scoring="f1").mean()
            rec = cross_val_score(m, X_train, y_train, cv=CV, scoring="recall").mean()
            print(f"  F1 CV: {f1:.3f} | Recall CV: {rec:.3f}")
            entrenados[nombre] = m.fit(X_train, y_train)
        else:
            m.fit(X_train, y_train)
            print(f"  Mejores parámetros: {m.best_params_} | {SCORING} CV: {m.best_score_:.3f}")
            entrenados[nombre] = m.best_estimator_
    return entrenados


# -------------------------------------------------------------- evaluacion
def comparar(entrenados, X_train, y_train, X_test, y_test):
    """Tabla de metricas en test, con un baseline (DummyClassifier)."""
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    candidatos = {"Baseline (prior)": baseline, **entrenados}

    filas = []
    for nombre, m in candidatos.items():
        pred = m.predict(X_test)
        proba = m.predict_proba(X_test)[:, 1]
        filas.append({"Modelo": nombre,
                      "Accuracy": accuracy_score(y_test, pred),
                      "Precision": precision_score(y_test, pred, zero_division=0),
                      "Recall": recall_score(y_test, pred),
                      "F1": f1_score(y_test, pred),
                      "ROC AUC": roc_auc_score(y_test, proba)})
    return pd.DataFrame(filas).set_index("Modelo").round(3)


# ----------------------------------------------------------------- guardado
def guardar(entrenados, tabla):
    (RAIZ / "models").mkdir(exist_ok=True)
    (RAIZ / "docs").mkdir(exist_ok=True)

    for i, (nombre, m) in enumerate(entrenados.items(), start=1):
        with open(RAIZ / "models" / f"trained_model_{i}.pkl", "wb") as f:
            pickle.dump(m, f)
        print(f"trained_model_{i}.pkl -> {nombre}")

    clasificador = ClasificadorRiesgo(entrenados[MODELO_FINAL], cortes=CORTES)
    with open(RAIZ / "models" / "final_model.pkl", "wb") as f:
        pickle.dump(clasificador, f)
    print(f"final_model.pkl -> {MODELO_FINAL} + cortes {CORTES}")

    tabla.to_csv(RAIZ / "docs" / "resultados_modelos.csv")


def main():
    X_train, X_test, y_train, y_test = cargar_y_dividir()
    entrenados = entrenar(definir_modelos(y_train), X_train, y_train)
    tabla = comparar(entrenados, X_train, y_train, X_test, y_test)
    print("\nComparación en test:\n", tabla)
    guardar(entrenados, tabla)


if __name__ == "__main__":
    main()