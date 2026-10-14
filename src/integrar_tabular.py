"""Integra prob_fatiga (P(NEG) del modelo de texto) con las variables tabulares del entrenamiento.

La exportación de SugarWOD solo trae rx_or_scaled, pr, score_type y best_result_raw.
RPE y ACWR por carga no existen en estos datos; se dejan como columnas pendientes
y, mientras tanto, se calcula un proxy de ACWR por frecuencia (días entrenados en 7 días / promedio semanal de 28 días).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluar import predecir_probabilidades


def _variables_de_frecuencia(df: pd.DataFrame) -> pd.DataFrame:
    """Días entrenados en la última semana frente al promedio semanal de las últimas 4 semanas.

    La exportación tiene una fila por componente del WOD (varias por día), por eso se cuentan días y no filas.
    """
    dias = pd.Series(1, index=pd.DatetimeIndex(df["date"].unique()).sort_values())
    diario = dias.asfreq("D", fill_value=0)
    agudo = diario.rolling(7, min_periods=1).sum()
    cronico = diario.rolling(28, min_periods=1).sum() / 4
    return pd.DataFrame({
        "dias_entrenados_7d": agudo,
        "dias_entrenados_28d_prom_semanal": cronico,
        "acwr_frecuencia_proxy": agudo / cronico,
    })


def construir_features(df_workouts: pd.DataFrame, clasificador, solo_subjetivas: bool = False) -> pd.DataFrame:
    """Una fila por entrenamiento con prob_fatiga y variables tabulares.

    prob_fatiga queda vacía en los entrenamientos sin nota (o con nota técnica si solo_subjetivas=True).
    """
    df = df_workouts.sort_values("date").reset_index(drop=True).copy()
    con_nota = df["notes"].notna()
    if solo_subjetivas and "tipo_nota" in df:
        con_nota &= df["tipo_nota"].eq("subjetiva")

    df["prob_fatiga"] = np.nan
    if con_nota.any():
        probs = predecir_probabilidades(clasificador, df.loc[con_nota, "notes"].tolist())
        df.loc[con_nota, "prob_fatiga"] = probs["NEG"].to_numpy()

    df["es_rx"] = df["rx_or_scaled"].astype(str).str.upper().eq("RX").astype(int)
    df["es_pr"] = df["pr"].fillna(False).astype(bool).astype(int)
    fechas = pd.Series(df["date"].unique())
    previa = dict(zip(fechas, fechas.diff().dt.days))
    df["dias_desde_sesion_previa"] = df["date"].map(previa)

    frecuencia = _variables_de_frecuencia(df)
    df = df.join(frecuencia, on="date")

    # Variables pedidas por el docente que aún no existen en la fuente (Wodify / registro de RPE).
    df["rpe"] = np.nan
    df["acwr_carga"] = np.nan

    columnas = [
        "date", "title", "score_type", "best_result_raw", "es_rx", "es_pr",
        "dias_desde_sesion_previa", "dias_entrenados_7d", "dias_entrenados_28d_prom_semanal",
        "acwr_frecuencia_proxy", "rpe", "acwr_carga", "notes", "prob_fatiga",
    ]
    return df[columnas]
