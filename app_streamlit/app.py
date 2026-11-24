"""
Aplicación Web Interactivo - Predicción de Riesgo de Enfermedad Cardiovascular
Proyecto: Predicción de Riesgo de Enfermedad Cardiovascular (ESdE 2025)
Autor: Jorge Sánchez Hernández
"""
import os
import sys
import pickle
from pathlib import Path

# Añade la carpeta padre (Riesgo_Cardiovascular_ML) a la ruta de búsqueda de Python
ruta_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ruta_raiz not in sys.path:
    sys.path.insert(0, ruta_raiz)

import numpy as np
import pandas as pd
import streamlit as st

# Configuración de rutas para importar ClasificadorRiesgo desde src/
RAIZ = Path(__file__).resolve().parent
if (RAIZ / "src").exists():
    sys.path.append(str(RAIZ / "src"))
elif (RAIZ.parent / "src").exists():
    sys.path.append(str(RAIZ.parent / "src"))
else:
    sys.path.append(str(RAIZ))

try:
    from riesgo import ClasificadorRiesgo
except ImportError:
    class ClasificadorRiesgo:
        NIVELES = ["Riesgo Bajo", "Riesgo Moderado", "Riesgo Alto"]

        def __init__(self, pipeline, cortes=(0.2, 0.5)):
            self.pipeline = pipeline
            self.cortes = cortes

        def predict_score(self, X):
            return self.pipeline.predict_proba(X)[:, 1]

        def predict(self, X):
            p = self.predict_score(X)
            return np.select(
                [p < self.cortes[0], p < self.cortes[1]],
                self.NIVELES[:2],
                default=self.NIVELES[2],
            )

        def predict_riesgo(self, X):
            return pd.DataFrame(
                {"score": self.predict_score(X), "nivel": self.predict(X)},
                index=X.index,
            )

# Configuración de la página
st.set_page_config(
    page_title="Evaluador de Riesgo Cardiovascular (ESdE 2025)",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .risk-card {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 8px solid #2e7d32;
        color: #1b5e20;
    }
    .risk-mod {
        background-color: #fff8e1;
        border-left: 8px solid #f57f17;
        color: #e65100;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 8px solid #c62828;
        color: #b71c1c;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_modelo():
    """Carga el modelo serializado final_model.pkl"""
    rutas_posibles = [
        RAIZ / "models" / "final_model.pkl",
        RAIZ.parent / "models" / "final_model.pkl",
        Path("models/final_model.pkl"),
    ]
    for ruta in rutas_posibles:
        if ruta.exists():
            with open(ruta, "rb") as f:
                return pickle.load(f)
    return None


def calcular_categoria_imc(imc: float) -> int:
    """Codificación estándar ESdE: 1=Peso insuficiente, 2=Normopeso, 3=Sobrepeso, 4=Obesidad"""
    if imc < 18.5:
        return 1
    elif imc < 25.0:
        return 2
    elif imc < 30.0:
        return 3
    else:
        return 4


# Diccionarios de mapeo según códigos de la Encuesta de Salud de España (ESdE)
MAP_SI_NO = {"No": 2, "Sí": 1}

MAP_ESTUDIOS = {
    "Sin estudios / Primaria incompleta": 2,
    "Educación Primaria": 3,
    "ESO / EGB / Bachillerato Elemental": 4,
    "Bachillerato / BUP / COU": 5,
    "FP Grado Medio": 6,
    "FP Grado Superior": 7,
    "Estudios Universitarios (Grado, Máster, Doctorado)": 8,
}

MAP_CLASE_SOCIAL = {
    "I - Directivos y profesionales universitarios": 1,
    "II - Ocupaciones intermedias y técnicos": 2,
    "III - Trabajadores cualificados no manuales": 3,
    "IV - Trabajadores cualificados manuales": 4,
    "V - Trabajadores semicualificados": 5,
    "VI - Trabajadores no cualificados": 6,
}

MAP_SEVERIDAD_DEP = {
    "Ninguna / Mínima": 1,
    "Leve": 2,
    "Moderada": 3,
    "Moderadamente grave": 4,
    "Grave": 5,
}

MAP_FUMA = {
    "Sí, fumo a diario": 1,
    "Sí, fumo pero no a diario": 2,
    "No fumo actualmente, pero he fumado antes (Exfumador)": 3,
    "No fumo ni he fumado nunca": 4,
}

MAP_HUMO_AJENO = {
    "Nunca o casi nunca": 1,
    "Menos de 1 hora al día": 2,
    "De 1 a 5 horas al día": 3,
    "Más de 5 horas al día": 4,
}

MAP_VAPEO = {
    "A diario": 1,
    "De forma ocasional": 2,
    "En el pasado (ya no)": 3,
    "Nunca": 4,
}

MAP_ALCOHOL = {
    "A diario o casi a diario": 1,
    "5-6 días por semana": 2,
    "3-4 días por semana": 3,
    "1-2 días por semana": 4,
    "2-3 días al mes": 5,
    "Una vez al mes": 6,
    "Menos de una vez al mes": 7,
    "No en los últimos 12 meses / Nunca": 8,
}

MAP_ACT_TRABAJO = {
    "Sentado la mayor parte de la jornada": 1,
    "De pie la mayor parte del tiempo sin grandes desplazamientos": 2,
    "Caminando, llevando algún peso o desplazándose con frecuencia": 3,
    "Realizando tareas que requieren gran esfuerzo físico": 4,
}

MAP_EJERCICIO_OCIO = {
    "No hago ejercicio (tiempo libre casi completamente sedentario)": 1,
    "Hago alguna actividad física o deportiva ocasional": 2,
    "Hago actividad física varias veces al mes": 3,
    "Hago entrenamiento deportivo o físico varias veces a la semana": 4,
}

MAP_DIETA = {
    "A diario (1 o más veces al día)": 1,
    "De 4 a 6 veces por semana": 2,
    "De 1 a 3 veces por semana": 3,
    "Menos de una vez a la semana": 4,
    "Nunca": 5,
}

# --- CABECERA ---
st.title("🫀 Predictor de Riesgo de Enfermedad Cardiovascular")
st.markdown(
    "Herramienta clínica de cribado basada en modelos de aprendizaje automático entrenados con la "
    "**Encuesta de Salud de España 2025 (Ministerio de Sanidad)**."
)
st.divider()

modelo = cargar_modelo()
if modelo is None:
    st.error(
        "⚠️ **No se encontró el archivo del modelo (`models/final_model.pkl`).** "
        "Ejecuta primero `python src/training.py` desde la raíz del proyecto para entrenar y guardar el modelo."
    )
    st.stop()

# --- SIDEBAR: DATOS DEMOGRÁFICOS Y ANTROPOMÉTRICOS ---
with st.sidebar:
    st.header("👤 Perfil Demográfico y Físico")
    edad = st.slider("Edad (años)", min_value=18, max_value=100, value=45, step=1)
    sexo_label = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
    mujer = 1 if sexo_label == "Mujer" else 0

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        peso_kg = st.number_input("Peso (kg)", min_value=35.0, max_value=220.0, value=74.0, step=0.5)
    with col_p2:
        altura_cm = st.number_input("Altura (cm)", min_value=120, max_value=220, value=172, step=1)

    imc = peso_kg / ((altura_cm / 100) ** 2)
    imc_cat = calcular_categoria_imc(imc)
    etiquetas_imc = {1: "Bajo peso", 2: "Normopeso", 3: "Sobrepeso", 4: "Obesidad"}
    st.info(f"**IMC calculado:** {imc:.1f} kg/m² ({etiquetas_imc[imc_cat]})")

    estudios_sel = st.selectbox("Nivel de estudios", list(MAP_ESTUDIOS.keys()), index=4)
    clase_sel = st.selectbox("Clase social ocupacional", list(MAP_CLASE_SOCIAL.keys()), index=2)

# --- FORMULARIO PRINCIPAL EN PESTAÑAS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 1. Antecedentes Clínicos",
    "🚬 2. Tabaco y Alcohol",
    "🏃 3. Actividad Física",
    "🥗 4. Hábitos Alimentarios",
])

with tab1:
    st.subheader("Patologías crónicas diagnosticadas y salud mental")
    c1, c2, c3 = st.columns(3)
    with c1:
        tension_alta = st.selectbox("Hipertensión arterial", ["No", "Sí"])
        colesterol_alto = st.selectbox("Colesterol alto", ["No", "Sí"])
        diabetes = st.selectbox("Diabetes", ["No", "Sí"])
        epoc = st.selectbox("EPOC / Bronquitis crónica", ["No", "Sí"])
    with c2:
        problemas_rinon = st.selectbox("Problemas crónicos de riñón", ["No", "Sí"])
        migrana = st.selectbox("Migraña o dolor de cabeza frecuente", ["No", "Sí"])
        tumor_maligno = st.selectbox("Tumor maligno / Cáncer", ["No", "Sí"])
    with c3:
        depresion = st.selectbox("Depresión diagnosticada", ["No", "Sí"])
        ansiedad = st.selectbox("Ansiedad crónica", ["No", "Sí"])
        otros_mentales = st.selectbox("Otros problemas de salud mental", ["No", "Sí"])
        sev_dep_sel = st.selectbox("Severidad sintomatología depresiva", list(MAP_SEVERIDAD_DEP.keys()))

with tab2:
    st.subheader("Consumo de tabaco, vapeo y alcohol")
    c1, c2 = st.columns(2)
    with c1:
        fuma_sel = st.selectbox("Consumo de tabaco", list(MAP_FUMA.keys()), index=3)
        humo_sel = st.selectbox("Exposición diaria al humo de tabaco ajeno", list(MAP_HUMO_AJENO.keys()))
        vapeo_sel = st.selectbox("Uso de cigarrillos electrónicos / vapeo", list(MAP_VAPEO.keys()), index=3)
    with c2:
        alcohol_sel = st.selectbox("Frecuencia de consumo de alcohol", list(MAP_ALCOHOL.keys()), index=4)

with tab3:
    st.subheader("Actividad física y sedentarismo")
    c1, c2 = st.columns(2)
    with c1:
        act_trabajo_sel = st.selectbox("Actividad física principal (trabajo/estudios/tareas)", list(MAP_ACT_TRABAJO.keys()))
        ej_ocio_sel = st.selectbox("Ejercicio físico en tiempo libre", list(MAP_EJERCICIO_OCIO.keys()), index=1)
        sentado_h = st.slider("Horas al día que pasa sentado", min_value=0.0, max_value=18.0, value=6.0, step=0.5)
    with c2:
        dias_camina = st.slider("Días a la semana que camina al menos 10 min seguidos", 0, 7, 5)
        dias_deporte = st.slider("Días a la semana de deporte / actividad moderada-intensa", 0, 7, 2)
        dias_fuerza = st.slider("Días a la semana de ejercicios de fortalecimiento muscular", 0, 7, 1)

with tab4:
    st.subheader("Frecuencia de consumo de grupos de alimentos")
    c1, c2, c3 = st.columns(3)
    with c1:
        dieta_fruta = st.selectbox("Fruta fresca", list(MAP_DIETA.keys()), index=0)
        dieta_verdura = st.selectbox("Verduras, ensaladas y hortalizas", list(MAP_DIETA.keys()), index=0)
        dieta_legumbres = st.selectbox("Legumbres", list(MAP_DIETA.keys()), index=2)
    with c2:
        dieta_pescado = st.selectbox("Pescado", list(MAP_DIETA.keys()), index=2)
        dieta_carne_proc = st.selectbox("Carne procesada / embutidos", list(MAP_DIETA.keys()), index=2)
        dieta_dulces = st.selectbox("Dulces / bollería / repostería", list(MAP_DIETA.keys()), index=3)
    with c3:
        dieta_refrescos = st.selectbox("Refrescos con azúcar", list(MAP_DIETA.keys()), index=3)
        dieta_comida_rapida = st.selectbox("Comida rápida (fast food)", list(MAP_DIETA.keys()), index=3)
        dieta_aperitivos = st.selectbox("Aperitivos salados / snacks", list(MAP_DIETA.keys()), index=3)

st.divider()

# --- BOTÓN DE PREDICCIÓN Y RESULTADOS ---
if st.button("🔍 Calcular Riesgo Cardiovascular", type="primary", use_container_width=True):
    datos_paciente = pd.DataFrame([{
        # Numéricas
        "edad": float(edad),
        "imc": float(imc),
        "sentado_h": float(sentado_h),
        "dias_camina": float(dias_camina),
        "dias_deporte": float(dias_deporte),
        "dias_fuerza": float(dias_fuerza),
        # Binaria
        "mujer": int(mujer),
        # Riesgo clínico
        "tension_alta": MAP_SI_NO[tension_alta],
        "colesterol_alto": MAP_SI_NO[colesterol_alto],
        "diabetes": MAP_SI_NO[diabetes],
        "epoc": MAP_SI_NO[epoc],
        "problemas_rinon": MAP_SI_NO[problemas_rinon],
        "migrana": MAP_SI_NO[migrana],
        "tumor_maligno": MAP_SI_NO[tumor_maligno],
        "depresion": MAP_SI_NO[depresion],
        "ansiedad": MAP_SI_NO[ansiedad],
        "otros_problemas_mentales": MAP_SI_NO[otros_mentales],
        # Dieta
        "dieta_fruta": MAP_DIETA[dieta_fruta],
        "dieta_verdura": MAP_DIETA[dieta_verdura],
        "dieta_legumbres": MAP_DIETA[dieta_legumbres],
        "dieta_pescado": MAP_DIETA[dieta_pescado],
        "dieta_carne_procesada": MAP_DIETA[dieta_carne_proc],
        "dieta_dulces": MAP_DIETA[dieta_dulces],
        "dieta_refrescos": MAP_DIETA[dieta_refrescos],
        "dieta_comida_rapida": MAP_DIETA[dieta_comida_rapida],
        "dieta_aperitivos": MAP_DIETA[dieta_aperitivos],
        # Otras categóricas
        "nivel_estudios": MAP_ESTUDIOS[estudios_sel],
        "clase_social": MAP_CLASE_SOCIAL[clase_sel],
        "ejercicio_ocio": MAP_EJERCICIO_OCIO[ej_ocio_sel],
        "severidad_depresiva": MAP_SEVERIDAD_DEP[sev_dep_sel],
        "fuma": MAP_FUMA[fuma_sel],
        "frec_alcohol": MAP_ALCOHOL[alcohol_sel],
        "actividad_trabajo": MAP_ACT_TRABAJO[act_trabajo_sel],
        "vapeo": MAP_VAPEO[vapeo_sel],
        "humo_ajeno": MAP_HUMO_AJENO[humo_sel],
        "imc_categoria": int(imc_cat),
    }])

    resultado = modelo.predict_riesgo(datos_paciente).iloc[0]
    score = float(resultado["score"])
    nivel = str(resultado["nivel"])

    if nivel == "Riesgo Bajo":
        css_class = "risk-low"
        icono = "✅"
        recomendacion = (
            "El perfil del paciente presenta una probabilidad estimada baja de enfermedad cardiovascular. "
            "Se recomienda mantener hábitos saludables de alimentación, actividad física regular y controles preventivos periódicos."
        )
    elif nivel == "Riesgo Moderado":
        css_class = "risk-mod"
        icono = "⚠️"
        recomendacion = (
            "Se observa un nivel de riesgo intermedio. Conviene revisar factores modificables (tensión arterial, perfil lipídico, "
            "sedentarismo o consumo de tabaco) y realizar seguimiento en atención primaria."
        )
    else:
        css_class = "risk-high"
        icono = "🚨"
        recomendacion = (
            "El modelo detecta una probabilidad elevada asociada a factores de riesgo cardiovascular. "
            "Se aconseja valoración médica prioritaria para control estricto de comorbilidades y prevención secundaria/primaria intensiva."
        )

    col_res1, col_res2 = st.columns([1.2, 1.8])
    with col_res1:
        st.markdown(
            f"""
            <div class="risk-card {css_class}">
                <h2 style="margin:0;">{icono} {nivel}</h2>
                <p style="font-size: 2.2rem; font-weight: bold; margin: 10px 0;">{score:.1%}</p>
                <p style="margin:0; font-size: 0.95rem;">Índice de riesgo estimado (Score)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(score, 0.0), 1.0))

    with col_res2:
        st.subheader("Interpretación Clínica Orientativa")
        st.write(recomendacion)
        st.caption(
            f"Umbrales del clasificador: Riesgo Bajo (< {modelo.cortes[0]:.0%}) | "
            f"Riesgo Moderado ({modelo.cortes[0]:.0%} – {modelo.cortes[1]:.0%}) | "
            f"Riesgo Alto (≥ {modelo.cortes[1]:.0%})."
        )
        st.warning(
            "**⚠️Estimación orientativa (no es un diagnóstico médico):** Este porcentaje refleja el grado "
                "de similitud de tu perfil con casos de enfermedad cardiovascular en la Encuesta de "
                "Salud de España 2025. Debido al ajuste estadístico del modelo, debe interpretarse "
                "como un indicador relativo para comparar perfiles y **NO** como una probabilidad clínica "
                "real ni una predicción a futuro. Consulta siempre con un profesional sanitario."
        )

st.caption(
    "**Aviso:** Esta aplicación es una herramienta estadística de apoyo basada en datos poblacionales (ESdE 2025) "
    "y **no sustituye** el criterio ni el diagnóstico de un profesional sanitario. "
    "La aplicación sigue en desarrollo y el resultado puede ser **incorrecto**."
)