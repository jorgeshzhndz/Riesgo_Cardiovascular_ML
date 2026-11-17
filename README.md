# 🩺 Predicción de Riesgo de Diabetes y Enfermedades Cardíacas

Proyecto de Machine Learning enfocado en salud pública que clasifica el riesgo de desarrollar diabetes o enfermedades cardiovasculares a partir de hábitos de vida e indicadores demográficos.

## 📋 Descripción del Proyecto
El objetivo principal es construir un modelo predictivo que identifique qué factores de riesgo tienen mayor impacto en la salud metabólica y cardiovascular de la población. Además de predecir el riesgo individual, el proyecto busca generar visualizaciones claras sobre la importancia de cada variable para apoyar estrategias de prevención temprana.

## 📊 Conjunto de Datos (Dataset)
Los datos provienen de las encuestas telefónicas anuales del sistema **BRFSS** (*Behavioral Risk Factor Surveillance System*) de los **Centros para el Control y la Prevención de Enfermedades (CDC)**.
* **Variables principales:** Hábitos de alimentación, actividad física, consumo de tabaco y alcohol, índice de masa corporal (IMC), presión arterial, colesterol, edad y nivel socioeconómico.
* **Variable objetivo:** Diagnóstico positivo o negativo de diabetes / enfermedad cardíaca (clasificación binaria).

## 🛠️ Tecnologías y Herramientas
* **Lenguaje:** Python 3.x
* **Análisis y Manipulación de Datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost

## Estructura del Repositorio
```text
├── data/               
├── notebooks/          
├── src/                
├── models/             
├── requirements.txt    
└── README.md           
```

## 📈 Metodología
1. **Análisis Exploratorio (EDA):** Limpieza de datos, gestión de valores nulos y análisis de correlaciones entre hábitos y diagnóstico.
2. **Preprocesamiento:** Codificación de variables categóricas, escalado de datos y balanceo de clases.
3. **Modelado y Evaluación:** Entrenamiento de modelos base (Regresión Logística) frente a modelos de ensamble (Random Forest, XGBoost), priorizando métricas como **Recall** y **F1-Score** para minimizar falsos negativos.
4. **Interpretabilidad:** Análisis de importancia de variables (*Feature Importance*) para explicar los resultados desde una perspectiva de salud pública.

## 💻 Instalación y Uso
1. Clona este repositorio:
   ```bash
   git clone https://github.com
   ```
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta los notebooks en orden numérico dentro de la carpeta `notebooks/`.

---
*Proyecto desarrollado como primer entregable práctico para Bootcamp de Data Science & Machine Learning.*
