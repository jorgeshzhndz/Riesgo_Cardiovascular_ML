import numpy as np
import pandas as pd

class ClasificadorRiesgo:
    """Envuelve un pipeline con predict_proba y devuelve niveles de riesgo."""
    NIVELES = ["Riesgo Bajo", "Riesgo Moderado", "Riesgo Alto"]

    def __init__(self, pipeline, cortes=(0.2, 0.5)):
        self.pipeline = pipeline
        self.cortes = cortes

    def predict_score(self, X):
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X):
        p = self.predict_score(X)
        return np.select([p < self.cortes[0], p < self.cortes[1]],
                         self.NIVELES[:2], default=self.NIVELES[2])

    def predict_riesgo(self, X):
        """DataFrame con score y nivel para cada paciente."""
        return pd.DataFrame({"score": self.predict_score(X), "nivel": self.predict(X)},
                            index=X.index)