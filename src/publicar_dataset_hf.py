"""Publica los dos conjuntos de datos del proyecto como datasets privados en Hugging Face.

- data/notas_etiquetadas_coach.csv  -> rick-lastra/wodpredict-notas-coach
- data/raw/workouts_sugarwod.csv    -> rick-lastra/wodpredict-sugarwod-raw (todas las columnas como texto)

Requiere un token con permiso de escritura en la variable de entorno HF_TOKEN:
    HF_TOKEN=... python src/publicar_dataset_hf.py
"""

import os

import pandas as pd
from datasets import Dataset

from datos import HF_DATASET_ID, HF_DATASET_RAW_ID, RAIZ, RUTA_ETIQUETADAS, RUTA_WORKOUTS_RAW

if __name__ == "__main__":
    token = os.environ["HF_TOKEN"]
    conjuntos = [
        (HF_DATASET_ID, pd.read_csv(RAIZ / RUTA_ETIQUETADAS)),
        (HF_DATASET_RAW_ID, pd.read_csv(RAIZ / RUTA_WORKOUTS_RAW, dtype=str, keep_default_na=False)),
    ]
    for repo_id, df in conjuntos:
        Dataset.from_pandas(df, preserve_index=False).push_to_hub(repo_id, private=True, token=token)
        print(f"Publicado: https://huggingface.co/datasets/{repo_id} ({len(df)} filas, privado)")
