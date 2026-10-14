"""Inferencia, métricas y latencia de los modelos de sentimiento de 3 clases."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ETIQUETAS = ["NEG", "NEU", "POS"]

# Los tres modelos tienen el mismo espacio de salida nativo: negativo, neutral y positivo.
MODELOS = {
    "RoBERTuito": "pysentimiento/robertuito-sentiment-analysis",
    "XLM-R Twitter": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "DistilBERT multilingüe": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
}

_MAPEO = {
    "pos": "POS", "positive": "POS", "label_2": "POS",
    "neu": "NEU", "neutral": "NEU", "label_1": "NEU",
    "neg": "NEG", "negative": "NEG", "label_0": "NEG",
}


def normalizar_etiqueta(etiqueta: str) -> str:
    return _MAPEO[etiqueta.lower()]


def cargar_clasificador(ruta_modelo: str, device: int | None = None):
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=ruta_modelo, device=device, top_k=None)


def predecir_probabilidades(clasificador, textos: list[str]) -> pd.DataFrame:
    """Probabilidades por clase (columnas NEG, NEU, POS) para cada texto."""
    salida = clasificador(list(textos), truncation=True, max_length=128)
    filas = [{normalizar_etiqueta(d["label"]): d["score"] for d in scores} for scores in salida]
    return pd.DataFrame(filas)[ETIQUETAS]


def decidir(probs: pd.DataFrame, umbral_neg: float | None = None) -> np.ndarray:
    """Etiqueta final. Sin umbral: argmax. Con umbral: NEG si P(NEG) >= umbral; si no, argmax entre NEU y POS."""
    if umbral_neg is None:
        return probs.idxmax(axis=1).to_numpy()
    resto = probs[["NEU", "POS"]].idxmax(axis=1)
    return np.where(probs["NEG"] >= umbral_neg, "NEG", resto)


def elegir_umbral(y_dev, probs_dev: pd.DataFrame, candidatos=None) -> float:
    """Umbral de P(NEG) que maximiza el F1 macro en dev (desempate: mayor recall de NEG)."""
    if candidatos is None:
        candidatos = np.round(np.arange(0.05, 0.96, 0.05), 2)
    mejor, mejor_clave = 0.5, (-1.0, -1.0)
    for u in candidatos:
        pred = decidir(probs_dev, u)
        f1m = f1_score(y_dev, pred, labels=ETIQUETAS, average="macro", zero_division=0)
        rec_neg = precision_recall_fscore_support(y_dev, pred, labels=["NEG"], zero_division=0)[1][0]
        if (f1m, rec_neg) > mejor_clave:
            mejor, mejor_clave = float(u), (f1m, rec_neg)
    return mejor


def metricas(y_real, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_real, y_pred, labels=ETIQUETAS, zero_division=0)
    m = {
        "Accuracy": accuracy_score(y_real, y_pred),
        "F1 macro": f1_score(y_real, y_pred, labels=ETIQUETAS, average="macro", zero_division=0),
    }
    for i, et in enumerate(ETIQUETAS):
        m[f"Precision {et}"], m[f"Recall {et}"], m[f"F1 {et}"] = p[i], r[i], f[i]
    return m


def reporte(y_real, y_pred) -> str:
    return classification_report(y_real, y_pred, labels=ETIQUETAS, digits=3, zero_division=0)


def matriz_confusion(y_real, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_real, y_pred, labels=ETIQUETAS)
    return pd.DataFrame(cm, index=[f"real {e}" for e in ETIQUETAS], columns=[f"pred {e}" for e in ETIQUETAS])


def medir_latencia(clasificador, textos: list[str], repeticiones: int = 5, calentamiento: int = 3) -> dict:
    """Latencia por comentario (una predicción a la vez), en milisegundos.

    Se reporta la mediana y el percentil 95 de todas las mediciones.
    """
    for t in textos[:calentamiento]:
        clasificador(t, truncation=True, max_length=128)
    tiempos = []
    for _ in range(repeticiones):
        for t in textos:
            inicio = time.perf_counter()
            clasificador(t, truncation=True, max_length=128)
            tiempos.append((time.perf_counter() - inicio) * 1000)
    tiempos = np.array(tiempos)
    return {
        "Latencia mediana (ms)": float(np.median(tiempos)),
        "Latencia p95 (ms)": float(np.percentile(tiempos, 95)),
        "Mediciones": len(tiempos),
    }
