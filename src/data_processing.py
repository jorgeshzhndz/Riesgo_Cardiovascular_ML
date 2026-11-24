"""
Procesamiento de datos (Data Loading y Data Cleansing)
Proyecto: Predicción de Riesgo de Enfermedad Cardiovascular
Autor: Jorge Sánchez Hernández
Datos: Encuesta de Salud España 2025 (Ministerio de Sanidad)
"""

import os
import numpy as np
import pandas as pd


# Diccionario de mapeo de variables originales a nombres descriptivos
NOMBRES_VARIABLES = {
    # Variable objetivo (luego se combinan en 'ecv' -> enfermedad cardiovascular)
    "C5a_2": "infarto",
    "C5a_3": "angina_y_otras_coronarias",
    "C5a_4": "otra_cardiopatia",
    "C5a_24": "ictus",

    # Peso muestral (utilizado en EDA, no es predictor)
    "FACTORADULTO": "factor",

    # Variables demográficas y socioeconómicas
    "EDADa": "edad",
    "SEXOa": "sexo",
    "NIVEST": "nivel_estudios",
    "CLASE_PR": "clase_social",

    # Cálculo del IMC
    "N1": "altura_cm",
    "N2": "peso_kg",
    "IMC": "imc_categoria",

    # Antecedentes clínicos (factores de riesgo cardiovascular)
    "C5a_1": "tension_alta",
    "C5a_15": "colesterol_alto",
    "C5a_12": "diabetes",
    "C5a_11": "epoc",
    "C5a_30": "problemas_rinon",
    "C5a_25": "migrana",
    "C5a_27": "tumor_maligno",

    # Salud mental
    "C5a_21": "depresion",
    "C5a_22": "ansiedad",
    "C5a_23": "otros_problemas_mentales",
    "SEVERIDAD_DEPRESIVA": "severidad_depresiva",

    # Tabaco
    "Q1": "fuma",
    "Q6": "humo_ajeno",
    "Q8": "vapeo",

    # Alcohol
    "R1": "frec_alcohol",

    # Ejercicio y sedentarismo
    "O1": "actividad_trabajo",
    "O2": "ejercicio_ocio",
    "O3": "dias_camina",
    "O7": "dias_deporte",
    "O9": "dias_fuerza",
    "O10_1": "sentado_h",
    "O10_2": "sentado_min",

    # Dieta
    "P1_1": "dieta_fruta",
    "P1_9": "dieta_verdura",
    "P1_10": "dieta_legumbres",
    "P1_6": "dieta_pescado",
    "P1_2": "dieta_carne_procesada",
    "P1_12": "dieta_dulces",
    "P1_13": "dieta_refrescos",
    "P1_14": "dieta_comida_rapida",
    "P1_15": "dieta_aperitivos",
}


def cargar_datos(ruta_sav: str, dir_salida: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos crudos desde el archivo SPSS (.sav), renombra las columnas
    y guarda las versiones iniciales de códigos y etiquetas en data/processed.
    """
    columnas = ["PROXY_0"] + list(NOMBRES_VARIABLES.keys())

    df_codigo = pd.read_spss(ruta_sav, usecols=columnas, convert_categoricals=False)
    df_etiquetas = pd.read_spss(ruta_sav, usecols=columnas)

    # Convertir códigos a tipo numérico
    df_codigo = df_codigo.apply(pd.to_numeric, errors="coerce")

    # Renombrar columnas
    df_codigo = df_codigo.rename(columns=NOMBRES_VARIABLES)
    df_etiquetas = df_etiquetas.rename(columns=NOMBRES_VARIABLES)

    os.makedirs(dir_salida, exist_ok=True)
    df_codigo.to_csv(os.path.join(dir_salida, "ESdEadulto_2025_codigos.csv"), index=False)
    df_etiquetas.to_csv(os.path.join(dir_salida, "ESdEadulto_2025_etiquetas.csv"), index=False)

    return df_codigo, df_etiquetas


def limpiar_datos(
    df_codigo: pd.DataFrame,
    df_etiquetas: pd.DataFrame,
    dir_salida: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica el proceso de limpieza, imputación de flujos, definición de la variable
    objetivo ('ecv'), recodificación e ingeniería de características básica.
    """
    df = df_codigo.copy()
    df_etiq = df_etiquetas.copy()

    # 1. Selección de cohorte: personas que responden por sí mismas (PROXY_0 == 1)
    cohort = df["PROXY_0"] == 1
    df = df[cohort].drop(columns="PROXY_0")
    df_etiq = df_etiq[cohort].drop(columns="PROXY_0")

    # 2. Definición de grupos de columnas y códigos de "No contesta" (NC)
    cols_ecv = ["infarto", "angina_y_otras_coronarias", "otra_cardiopatia", "ictus"]
    cols_riesgo = [
        "tension_alta", "colesterol_alto", "diabetes", "epoc", "problemas_rinon",
        "migrana", "tumor_maligno", "depresion", "ansiedad", "otros_problemas_mentales",
    ]
    cols_dieta = [
        "dieta_fruta", "dieta_verdura", "dieta_legumbres", "dieta_pescado",
        "dieta_carne_procesada", "dieta_dulces", "dieta_refrescos",
        "dieta_comida_rapida", "dieta_aperitivos",
    ]

    cols_9 = (
        cols_ecv
        + cols_riesgo
        + cols_dieta
        + [
            "clase_social", "severidad_depresiva", "fuma", "humo_ajeno", "vapeo",
            "actividad_trabajo", "ejercicio_ocio", "dias_camina", "dias_deporte",
            "dias_fuerza", "apoyo_personas", "imc_categoria",
        ]
    )
    cols_99 = [
        "nivel_estudios", "frec_atracon_alcohol", "sentado_h", "sentado_min",
    ]
    cols_999 = ["edad", "altura_cm", "peso_kg"]

    # 4. Sustitución de códigos "No contesta" (9, 99, 999) por NaN
    df[cols_9] = df[cols_9].replace(9, np.nan)
    df[cols_99] = df[cols_99].replace(99, np.nan)
    df[cols_999] = df[cols_999].replace(999, np.nan)

    # 5. Definición de la variable objetivo binaria 'ecv'
    df = df[df[cols_ecv].notna().all(axis=1)].copy()
    df["ecv"] = (df[cols_ecv] == 1).any(axis=1).astype(int)
    df = df.drop(columns=cols_ecv)

    # 6. Recodificación de variables y creación de características
    df["mujer"] = df["sexo"].map({1: 0, 2: 1})
    df["imc"] = df["peso_kg"] / (df["altura_cm"] / 100) ** 2
    df = df.drop(columns=["sexo", "altura_cm", "peso_kg", "sentado_min"])

    # 7. Realineación del dataset de etiquetas con el índice limpio
    df_etiq = df_etiq.loc[df.index]

    # 8. Guardado de datasets procesados finales
    os.makedirs(dir_salida, exist_ok=True)
    df.to_csv(os.path.join(dir_salida, "ecv_procesado.csv"), index=False)
    df_etiq.to_csv(os.path.join(dir_salida, "ecv_procesado_etiquetas.csv"), index=False)

    return df, df_etiq


def ejecutar_pipeline(
    ruta_sav: str = "../data/raw/ESdEadulto_2025.sav",
    dir_salida: str = "../data/processed",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el pipeline completo de carga y limpieza de datos."""
    print("Iniciando carga de datos desde SPSS...")
    df_codigo, df_etiquetas = cargar_datos(ruta_sav, dir_salida)
    print(f"Datos cargados correctamente: {df_codigo.shape}")

    print("Ejecutando limpieza y transformación de variables...")
    df_procesado, df_etiquetas_proc = limpiar_datos(df_codigo, df_etiquetas, dir_salida)
    print(f"Proceso finalizado. Dimensiones finales: {df_procesado.shape}")
    print(f"Prevalencia ECV en muestra: {df_procesado['ecv'].mean():.3%}")

    return df_procesado, df_etiquetas_proc


if __name__ == "__main__":
    ejecutar_pipeline()