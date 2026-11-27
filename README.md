# 🩺 Predicción de riesgo cardiovascular en España

Proyecto final de Machine Learning supervisado del bootcamp de Data Science de **The Bridge**.
Estima, a partir de hábitos de vida, antecedentes de salud y datos demográficos, la probabilidad de que una persona adulta tenga una enfermedad cardiovascular (ECV) y la clasifica en **riesgo bajo, moderado o alto**.

**Autor:** Jorge Sánchez Hernández
**App:** [riesgocardiovascular.streamlit.app](https://riesgocardiovascular.streamlit.app/)

> ⚠️ Es una herramienta de apoyo con fines formativos y de cribado orientativo. **No es un diagnóstico médico.**

---

## 📋 El problema

Las enfermedades cardiovasculares son una de las principales causas de morbilidad en España. Identificar qué perfiles tienen más probabilidad de padecerlas permite **priorizar la prevención y el cribado** (atención primaria, campañas dirigidas, autoevaluación ciudadana).

- **Tipo de problema:** clasificación binaria supervisada.
- **Variable objetivo (`ecv`):** vale 1 si la persona declara haber padecido alguna vez **infarto de miocardio, angina de pecho o enfermedad coronaria, otra enfermedad del corazón o ictus**, y 0 en caso contrario.
- **Desbalanceo:** solo el **10,0 %** de la muestra tiene ECV (1.995 de 20.016 personas). La cifra ponderada para la población adulta es del **8,6 %**.

## 📊 Datos

| | |
|---|---|
| **Fuente** | Encuesta de Salud de España 2025 (ESdE), Ministerio de Sanidad e INE. Cuestionario del adulto |
| **Formato original** | SPSS (`ESdEadulto_2025.sav`) con diccionario de registro (`dr_ESdEadulto_2025.xlsx`) |
| **Tamaño original** | 21.641 adultos y 446 variables |
| **Dataset final** | 20.016 personas × 42 columnas (`ecv`, `factor` y 40 predictores) |

> El fichero original pesa unos 77 MB y **no se incluye en el repositorio**. Hay que descargarlo de la web oficial de la encuesta y colocarlo en `data/raw/`.

### Variables utilizadas (40 predictores)

| Grupo | Nº | Variables |
|---|---|---|
| Demografía | 5 | edad, mujer, nivel de estudios, clase social, comunidad autónoma |
| Cuerpo | 2 | IMC calculado y categoría de IMC de la encuesta |
| Antecedentes clínicos | 10 | tensión alta, colesterol alto, diabetes, EPOC, problemas de riñón, migraña, tumor maligno, depresión, ansiedad, otros problemas mentales |
| Salud mental | 1 | severidad de síntomas depresivos |
| Tabaco | 4 | fuma, cigarrillos al día, humo ajeno, vapeo |
| Alcohol | 2 | frecuencia y consumo excesivo |
| Ejercicio y sedentarismo | 6 | actividad en el trabajo, ejercicio en el ocio, días que camina, días de deporte, días de fuerza, horas sentado |
| Dieta | 9 | fruta, verdura, legumbres, pescado, carne procesada, dulces, refrescos, comida rápida, aperitivos |
| Apoyo social | 1 | personas con las que podría contar |

**Variables excluidas a propósito** (son consecuencia de estar enfermo y producirían fuga de información): medicación, consultas, hospitalizaciones y mediciones de tensión, colesterol o azúcar. El peso muestral (`factor`) solo se usa en el EDA, no como predictor.

## 🧹 Limpieza de datos

1. **Cohorte:** se mantienen las 20.357 personas que **responden por sí mismas** (se excluyen 1.284 cuestionarios contestados por terceros, que tienen un 33 % de ECV y datos menos fiables).
2. **Objetivo conocido:** se eliminan 341 personas sin respuesta en alguna de las cuatro enfermedades del objetivo.
3. **«No contesta» (códigos 9, 99 y 999):** pasan a dato faltante (`NaN`) y se **imputan dentro del pipeline** con la mediana en las numéricas y la moda en las categóricas. El 29 % de las personas tiene al menos un «No contesta»; borrarlas habría supuesto perder ese porcentaje de muestra y a personas con algo más de ECV.
4. **Blancos por filtro:** quien no bebe no responde a la pregunta de consumo excesivo de alcohol; se rellena con su respuesta de frecuencia (lo ha dejado / nunca) en lugar de tratarlo como un dato perdido.
5. **Variables nuevas:** IMC (peso y altura), `mujer` (0/1) y `ecv` (combinación de las cuatro enfermedades).
6. **Dos versiones de los datos:** con códigos numéricos (para el modelo) y con etiquetas de texto (para gráficos e interpretación).

## 🤖 Modelado

Todos los modelos usan un **único `Pipeline`** (imputación, escalado de numéricas y one-hot de categóricas ajustados solo con los datos de entrenamiento) y se evalúan en un test del 20 % estratificado (`random_state=42`).

| Modelo | Búsqueda de hiperparámetros |
|---|---|
| Regresión Logística | Sin búsqueda (baseline), `class_weight="balanced"` |
| Árbol de decisión | `GridSearchCV`: `max_depth`, `min_samples_leaf` |
| Random Forest | `GridSearchCV`: `n_estimators`, `max_depth`, `min_samples_leaf` |
| XGBoost | `RandomizedSearchCV` (10 combinaciones): `n_estimators`, `max_depth`, `learning_rate`, `subsample` |
| KNN | `GridSearchCV`: `n_neighbors`, `weights` |

Validación cruzada de 5 particiones y **F1** como métrica de selección. El desbalanceo se compensa con pesos de clase (`class_weight` y `scale_pos_weight`).

## 🎯 Métricas elegidas

Con un 10 % de casos, un modelo que dijera «nadie tiene ECV» acertaría un 90 % (*accuracy*), por lo que esa métrica no sirve. Se usan:

- **Recall:** qué porcentaje de las personas con ECV detecta el modelo. En un cribado, perder un caso es más grave que revisar de más.
- **F1:** equilibrio entre recall y precisión, usado para elegir hiperparámetros.
- **ROC AUC:** capacidad de separar personas con y sin ECV, independiente del umbral y robusta al desbalanceo.

## 📈 Resultados (conjunto de test)

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Regresión Logística** | 0,738 | 0,232 | **0,704** | 0,349 | **0,783** |
| Árbol de decisión | 0,722 | 0,218 | 0,689 | 0,331 | 0,749 |
| Random Forest | 0,807 | 0,270 | 0,551 | **0,363** | 0,781 |
| XGBoost | 0,767 | 0,248 | 0,657 | 0,360 | 0,777 |
| KNN | 0,891 | 0,235 | 0,040 | 0,069 | 0,624 |

- Regresión Logística, Random Forest y XGBoost quedan **prácticamente empatados** (ROC AUC ≈ 0,78).
- **Modelo final: Regresión Logística.** Tiene el mejor ROC AUC y el mejor recall (detecta 7 de cada 10 casos), y es el más sencillo de explicar y desplegar.
- KNN parece el mejor por *accuracy*, pero detecta solo el 4 % de los casos: es un ejemplo de por qué esa métrica engaña aquí.

### Estratificación del riesgo

El modelo final (`models/final_model.pkl`) es la Regresión Logística envuelta en `ClasificadorRiesgo` (`src/riesgo.py`). Cada persona recibe un *score* y se compara con **dos cortes fijos, iguales para todas** (0,2 y 0,5):

| Nivel | Score | Personas del test | % real con ECV |
|---|---|:---:|:---:|
| Riesgo bajo | < 0,2 | ≈ 33 % | ≈ 2 % |
| Riesgo moderado | 0,2 a 0,5 | ≈ 37 % | ≈ 6 % |
| Riesgo alto | ≥ 0,5 | ≈ 30 % | ≈ 23 % (concentra ≈ 71 % de los casos) |

El porcentaje real de ECV **sube de nivel en nivel**, lo que valida el *score*. Los resultados exactos se obtienen con `python evaluation.py`.

## ⚠️ Limitaciones

- Encuesta **autodeclarada y transversal**: estima si una persona *tiene* el antecedente, no predice el futuro, y hay causalidad inversa (por ejemplo, muchos exfumadores dejaron de fumar tras enfermar).
- No hay tensión ni colesterol medidos, solo si existe diagnóstico; faltan factores clásicos como antecedentes familiares o analíticas.
- Se excluyen las respuestas por terceros, que concentran más ECV.
- Con el balanceo de clases, el *score* **no es una probabilidad calibrada** (el score medio es mucho mayor que el 10 % real): los niveles son relativos.
- La imputación con la moda hace que una persona que no contesta se parezca a la más frecuente (sesgo pequeño: entre el 1 % y el 3 % por variable).

## 🚀 Próximos pasos

- Calibrar las probabilidades y validar con otra edición de la encuesta.
- Incorporar datos clínicos (historia familiar, analíticas) desde registros sanitarios.
- Entrenar un segundo modelo para diabetes con las mismas variables.

## 📁 Estructura del repositorio

```text
├── data/
│   ├── raw/          # microdatos originales (el .sav no se incluye)
│   ├── processed/    # datos limpios (ecv_procesado.csv y versión con etiquetas)
│   ├── train/        # train.csv
│   └── test/         # test.csv
├── notebooks/
│   ├── 01_Fuentes.ipynb
│   ├── 02_LimpiezaEDA.ipynb
│   └── 03_Entrenamiento_Evaluacion.ipynb
├── src/
│   ├── data_processing.py   # carga y limpieza de datos
│   ├── training.py          # entrenamiento de los 5 modelos
│   ├── evaluation.py        # evaluación del modelo final en test
│   └── riesgo.py            # clase ClasificadorRiesgo (niveles bajo / moderado / alto)
├── models/
│   ├── trained_model_1.pkl ... trained_model_5.pkl
│   └── final_model.pkl
├── app_streamlit/    # aplicación web (app.py y requirements.txt)
├── docs/
│   ├── presentacion.pptx
│   ├── memoria.md
│   └── resultados_modelos.csv
└── README.md
```

## 💻 Cómo ejecutar el proyecto

1. Clona el repositorio:
   ```bash
   git clone https://github.com/jorgeshzhndz/Riesgo_Cardiovascular_ML
   cd Riesgo_Cardiovascular_ML
   ```
2. Instala las dependencias (`pyreadstat` lee el archivo SPSS):
   ```bash
   pip install pandas numpy scikit-learn xgboost pyreadstat matplotlib seaborn
   ```
3. Descarga `ESdEadulto_2025.sav` de la web oficial de la encuesta y colócalo en `data/raw/`.
4. Ejecuta los notebooks en orden numérico (`notebooks/01`, `02` y `03`), **o** los scripts desde la carpeta `src/`:
   ```bash
   cd src
   python data_processing.py   # carga y limpieza -> data/processed/
   python training.py          # entrena y guarda train/test, modelos y resultados
   python evaluation.py        # evalúa el modelo final en test
   ```
5. Para usar la aplicación en local:
   ```bash
   pip install -r app_streamlit/requirements.txt
   streamlit run app_streamlit/app.py
   ```

> El modelo guardado en `final_model.pkl` es un objeto `ClasificadorRiesgo`: para cargarlo, `src/riesgo.py` debe estar en el path de Python.

---

*Proyecto desarrollado como entregable final del módulo de Machine Learning del bootcamp de Data Science de The Bridge.*