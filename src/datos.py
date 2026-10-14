"""Carga y limpieza de datos para WODPredict-AI.

Fuentes, en orden de preferencia:
1. Hugging Face Datasets (dataset privado del proyecto), si hay token con acceso.
2. Copia local del repositorio (cuando se trabaja desde un clon).
3. URL raw de GitHub (Colab sin clonar el repositorio).
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
GITHUB_RAW = "https://raw.githubusercontent.com/rick-lastra/WODPredictAI/main"
HF_DATASET_ID = "rick-lastra/wodpredict-notas-coach"
HF_DATASET_RAW_ID = "rick-lastra/wodpredict-sugarwod-raw"

RUTA_WORKOUTS_RAW = "data/raw/workouts_sugarwod.csv"
RUTA_ETIQUETADAS = "data/notas_etiquetadas_coach.csv"

ETIQUETAS = ["NEG", "NEU", "POS"]


def _leer_csv(ruta_relativa: str) -> tuple[pd.DataFrame, str]:
    local = RAIZ / ruta_relativa
    if local.exists():
        return pd.read_csv(local), f"local:{ruta_relativa}"
    url = f"{GITHUB_RAW}/{ruta_relativa}"
    return pd.read_csv(url), f"github:{url}"


def cargar_workouts_raw(usar_hf: bool = True) -> pd.DataFrame:
    """Exportación original de SugarWOD (792 filas, 11 columnas).

    Intenta primero Hugging Face Datasets; si no hay acceso, usa el CSV del repositorio.
    """
    df = None
    if usar_hf:
        try:
            from datasets import load_dataset

            # En HF todas las columnas se guardan como texto; se vuelven a leer como CSV
            # para obtener los mismos tipos y valores vacíos que el archivo original.
            texto = load_dataset(HF_DATASET_RAW_ID, split="train").to_pandas().to_csv(index=False)
            df, origen = pd.read_csv(io.StringIO(texto)), f"huggingface:{HF_DATASET_RAW_ID}"
        except Exception as e:  # sin token, sin red o dataset no disponible
            print(f"Hugging Face no disponible ({type(e).__name__}); se usa el CSV del repositorio.")
    if df is None:
        df, origen = _leer_csv(RUTA_WORKOUTS_RAW)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df.attrs["origen"] = origen
    return df


def cargar_etiquetadas(usar_hf: bool = True) -> pd.DataFrame:
    """Conjunto de 84 comentarios validados por el coach (columna etiqueta_coach).

    Intenta primero Hugging Face Datasets; si no hay acceso, usa el CSV del repositorio.
    """
    if usar_hf:
        try:
            from datasets import load_dataset

            ds = load_dataset(HF_DATASET_ID, split="train")
            df = ds.to_pandas()
            df.attrs["origen"] = f"huggingface:{HF_DATASET_ID}"
            return df
        except Exception as e:  # sin token, sin red o dataset aún no publicado
            print(f"Hugging Face no disponible ({type(e).__name__}); se usa el CSV del repositorio.")
    df, origen = _leer_csv(RUTA_ETIQUETADAS)
    df.attrs["origen"] = origen
    return df


# Una nota es "técnica" cuando solo registra cargas, repeticiones, tiempos o texto
# no relacionado con cómo se sintió el atleta (por ejemplo, devocionales).
_PATRON_SUBJETIVO = re.compile(
    r"\b(?:tired|fatigue|sore|hurt|pain|felt|feel|feeling|good|great|bad|hard|easy|heavy|"
    r"cansad\w*|fatiga\w*|dolor\w*|duel\w*|me sent\w*|sentí|bien|mal|pesad\w*|fácil|lesi\w*)\b",
    re.IGNORECASE,
)
_PATRON_BIBLICO = re.compile(
    r"\b(?:lord|god|jesus|christ|psalm|proverbs|isaiah|john|matthew|romans|corinthians|"
    r"genesis|exodus|job|señor|dios|salmo\w*)\b",
    re.IGNORECASE,
)


def es_nota_subjetiva(texto: str) -> bool:
    """True si la nota parece describir la percepción del atleta (y no solo datos técnicos)."""
    if not isinstance(texto, str) or not texto.strip():
        return False
    if _PATRON_BIBLICO.search(texto):
        return False
    return bool(_PATRON_SUBJETIVO.search(texto))


def clasificar_notas(df_workouts: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las filas con notas y una columna tipo_nota: subjetiva o tecnica."""
    notas = df_workouts.dropna(subset=["notes"]).copy()
    notas["tipo_nota"] = notas["notes"].map(lambda t: "subjetiva" if es_nota_subjetiva(t) else "tecnica")
    return notas


def split_dev_test(df: pd.DataFrame, test_size: float = 0.5, semilla: int = 42):
    """Split estratificado por etiqueta del coach e idioma.

    dev se usa para elegir umbrales; test solo para reportar.
    """
    estrato = df["etiqueta_coach"] + "_" + df["idioma"]
    dev, test = train_test_split(df, test_size=test_size, stratify=estrato, random_state=semilla)
    return dev.reset_index(drop=True), test.reset_index(drop=True)
