# Ficha de datos

| Archivo | Qué contiene | Filas |
|---|---|---|
| `raw/workouts_sugarwod.csv` | Exportación original de SugarWOD (también en el dataset privado `rick-lastra/wodpredict-sugarwod-raw`) de un atleta (30/10/2022 a 22/04/2025), sin modificaciones | 792 registros (componentes de WOD en 248 días de entrenamiento), 143 con nota |
| `workouts.csv` | Versión usada en la primera entrega (3 títulos acortados y tiempos con otro formato). Se conserva por trazabilidad | 792 |
| `notas_etiquetadas_coach.csv` | Muestra experimental para evaluar los modelos (ground truth; también en el dataset privado `rick-lastra/wodpredict-notas-coach`) | 84 comentarios |

## Por qué hay un conjunto etiquetado aparte

Las 143 notas reales son técnicas: pesos, repeticiones, tiempos, sustituciones de ejercicio o reflexiones devocionales.
Ninguna describe cómo se sintió el atleta, y con la regla de palabras clave de la primera entrega la muestra de 50 notas
quedó con 0 casos NEG. No se puede medir la detección de fatiga sin ejemplos de fatiga.

## Origen y etiquetado de `notas_etiquetadas_coach.csv`

- **Redacción:** 84 comentarios cortos, en español (46) e inglés (38), escritos con el estilo de las notas de un box de CrossFit:
  sesiones normales, fatiga, dolor o mala recuperación, y casos ambiguos (por ejemplo, "el WOD estuvo pesado, pero me sentí bien").
  Se redactaron con apoyo de IA a partir de situaciones reales del box.
- **Validación:** Ricardo Lastra, coach CF-L1/L2, revisó cada comentario y asignó la etiqueta final el 14/10/2026.
  Cambió 2 de las 84 etiquetas propuestas (ids 7 y 9, de NEU a NEG); concordancia del 97.6 %.
- **Criterio de etiquetas:**
  - `NEG`: fatiga, dolor, mala recuperación o señales de riesgo de lesión.
  - `NEU`: logística o comentario sin señal clara de estado.
  - `POS`: buena recuperación o buen desempeño.
- **Distribución final:** 30 NEG, 26 NEU, 28 POS.

| Columna | Descripción |
|---|---|
| `id` | Identificador del comentario |
| `texto` | Comentario del atleta |
| `idioma` | `es` o `en` |
| `etiqueta_propuesta` | Etiqueta sugerida al redactar (solo trazabilidad, no se usa para evaluar) |
| `etiqueta_coach` | **Ground truth** asignado por el coach |
| `comentario_coach` | Observaciones del coach (vacío si no hubo) |

## Limitaciones

- Un solo anotador; falta medir la concordancia con un segundo coach.
- Los comentarios son redactados, no notas reales de atletas; antes de producción hay que validar con notas reales que incluyan percepción de esfuerzo.
- Muestra pequeña: con 42 comentarios en test, cada error cambia el recall de NEG en unos 7 puntos.
