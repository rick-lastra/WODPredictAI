# 🏋️ WODPredict-AI: Detección de Fatiga en Atletas usando NLP

## Título y Descripción
**Nombre del proyecto:** WODPredict-AI (Módulo Análisis de Sentimientos)
Este repositorio contiene la prueba de concepto para integrar Inteligencia Artificial en la red de centros **UnderCover Fitness**. A través del ecosistema de Hugging Face, se utilizan técnicas de Procesamiento de Lenguaje Natural (NLP) para analizar los comentarios de los atletas tras sus entrenamientos (WODs) y detectar riesgos de sobreentrenamiento o fatiga aguda.

**Objetivo General:**
Seleccionar, evaluar e implementar un modelo preentrenado de análisis de texto capaz de clasificar los comentarios de los atletas en español e inglés como **POS** (estado óptimo), **NEU** (sin señal) o **NEG** (fatiga o riesgo), proporcionando una alerta temprana a los coaches para adaptar las cargas de entrenamiento.

**Contexto Profesional:**
Iniciativa alineada al *Business Case* de UnderCover Fitness para reducir la deserción de atletas en un 25% derivada de lesiones por acumulación de fatiga.

## 🤗 Ecosistema Hugging Face
| Componente | Uso en el proyecto |
|---|---|
| **Models** | Los 3 modelos evaluados se descargan del Hub con `transformers.pipeline` |
| **Datasets** | Los dos conjuntos se publican como datasets privados y se cargan con `datasets.load_dataset`: la exportación de SugarWOD (`rick-lastra/wodpredict-sugarwod-raw`) y el conjunto etiquetado por el coach (`rick-lastra/wodpredict-notas-coach`). Si no hay acceso, se leen los CSV del repositorio |
| **Spaces** | Despliegue futuro: demo en Gradio para que el coach pegue un comentario y vea la clase y `prob_fatiga` |

## 🛠️ Tecnologías Utilizadas
*   **Lenguaje:** Python 3.10+
*   **Entorno:** Google Colab conectado a un entorno de ejecución local (GPU NVIDIA GeForce RTX 4050 Laptop en un contenedor Docker). Abre siempre la versión de `main`: https://colab.research.google.com/github/rick-lastra/WODPredictAI/blob/main/notebooks/Actividad_4.ipynb (también corre en CPU)
*   **Librerías principales:** `transformers`, `datasets`, `huggingface_hub`, `torch`, `scikit-learn`, `pandas`, `matplotlib`
*   **Modelos evaluados (los tres de 3 clases: negativo, neutral y positivo):**
    * `pysentimiento/robertuito-sentiment-analysis`
    * `cardiffnlp/twitter-xlm-roberta-base-sentiment`
    * `lxyuan/distilbert-base-multilingual-cased-sentiments-student`

## 📁 Estructura del Proyecto
*   `data/raw/workouts_sugarwod.csv`: exportación original de SugarWOD (792 filas, 143 con nota).
*   `data/notas_etiquetadas_coach.csv`: 84 comentarios con etiqueta validada por un coach CF-L1/L2 (ground truth).
*   `data/workouts.csv`: versión usada en la primera entrega (se conserva por trazabilidad).
*   `data/README.md`: ficha de origen, etiquetado y limitaciones de los datos.
*   `src/datos.py`: carga (Hugging Face, local o GitHub), filtro de notas técnicas y split dev/test.
*   `src/evaluar.py`: inferencia, umbral, métricas por clase, matriz de confusión y latencia.
*   `src/integrar_tabular.py`: `prob_fatiga` + variables tabulares para el modelo de rendimiento.
*   `src/publicar_dataset_hf.py`: publica los dos conjuntos como datasets privados en Hugging Face (requiere token de escritura).
*   `notebooks/Actividad_4.ipynb`: experimento completo con conclusiones parciales y generales.
*   `resultados/`: matrices de confusión, comparativa, métricas en CSV y features tabulares.
*   `requirements.txt`: dependencias para reproducir el entorno.

## ⚙️ Instrucciones de Uso

1. **Clonar el repositorio e instalar dependencias:**
   ```bash
   git clone https://github.com/rick-lastra/WODPredictAI.git
   cd WODPredictAI
   pip install -r requirements.txt
   ```
2. **Ejecutar en local:** abre `notebooks/Actividad_4.ipynb` y ejecuta las celdas en orden. Las rutas son relativas al repositorio.
3. **Ejecutar en Google Colab:** abre el notebook en Colab y ejecuta las celdas en orden. La celda del punto 2 clona el repositorio (requirements, scripts de `src/` y CSV de respaldo) e instala las dependencias; no hay que subir archivos a mano. Esta entrega se ejecutó con Colab conectado a un entorno de ejecución local (GPU RTX 4050 en un contenedor Docker); también funciona con una GPU hospedada de Colab o en CPU.
4. **Datasets privados en Hugging Face (opcional):** guarda tu token como secreto `HF_TOKEN` en Colab (o como variable de entorno). Sin token, el notebook usa los CSV del repositorio.

## 🧪 Metodología de evaluación
*   **Ground truth:** las notas reales de la exportación son técnicas (pesos, repeticiones, devocionales) y no describen fatiga. En la primera entrega las etiquetas salían de palabras clave y la muestra quedó con 0 casos NEG. Ahora se usan 84 comentarios en español e inglés, etiquetados por el coach (30 NEG, 26 NEU, 28 POS). Detalle en `data/README.md`.
*   **Split:** 42 comentarios para dev y 42 para test, estratificados por etiqueta e idioma. El umbral de `P(NEG)` se elige en dev y se reporta en test.
*   **Métricas:** F1 macro, precisión, recall y F1 por clase, con **recall de NEG** como métrica principal, y matriz de confusión por modelo.
*   **Latencia:** por comentario, una predicción a la vez, mediana y percentil 95 de 5 corridas sobre test (210 mediciones por modelo), con un umbral de < 500 ms.
*   **Estructura del notebook:** una celda por cada uno de los 9 puntos de la actividad (GPU, dependencias, autenticación, carga de modelos y datasets, inferencia, métricas, latencia, tablas y conclusiones parciales).

## 📊 Resultados Obtenidos (test, 42 comentarios, regla argmax)

| Modelo | F1 macro | Recall NEG | F1 NEG |
|---|---|---|---|
| XLM-R Twitter | 0.713 | 1.000 | 0.882 |
| RoBERTuito | 0.678 | 0.800 | 0.727 |
| DistilBERT multilingüe | 0.578 | 0.733 | 0.733 |

**Latencia por comentario** en la RTX 4050 (mediana y p95 de 210 mediciones por modelo):

| Modelo | Latencia mediana | Latencia p95 |
|---|---|---|
| XLM-R Twitter | 18 ms | 23 ms |
| RoBERTuito | 20 ms | 32 ms |
| DistilBERT multilingüe | 10 ms | 13 ms |

*   Ningún modelo llega al F1 macro de 0.80 fijado como umbral, así que ninguno está listo para producción. Los tres cumplen la latencia de < 500 ms por comentario. Como referencia para un microservicio sin GPU, en CPU la mediana fue de 39 a 76 ms.
*   **Candidato:** `cardiffnlp/twitter-xlm-roberta-base-sentiment`, que detectó todos los comentarios de fatiga de test. RoBERTuito queda como segundo candidato. El F1 de 0.859 de la primera entrega estaba inflado por las etiquetas de palabras clave.
*   La clase más difícil es NEU (falsas alarmas), por lo que la salida es una señal para el coach (*human-in-the-loop*), no un diagnóstico.
*   **Integración con el modelo tabular:** `P(NEG)` se usa como `prob_fatiga` junto con Rx/Scaled, PRs, tipo de puntuación y resultado; como no hay RPE ni cargas, se calcula un proxy de ACWR por frecuencia (ver `src/integrar_tabular.py` y `resultados/features_tabulares.csv`).

## ⚠️ Limitaciones
*   Comentarios redactados y validados por un solo coach; falta concordancia con un segundo coach y validación con notas reales.
*   Muestra pequeña: en test, cada error cambia el recall de NEG unos 7 puntos, así que las cifras son orientativas.
*   Las notas reales de la exportación son técnicas; antes de producción hay que pedir a los atletas un comentario breve sobre cómo se sintieron.
*   La exportación de SugarWOD no trae RPE ni cargas para calcular ACWR.

## 🔗 Enlaces
*   Dataset SugarWOD (privado): https://huggingface.co/datasets/rick-lastra/wodpredict-sugarwod-raw
*   Dataset etiquetado por el coach (privado): https://huggingface.co/datasets/rick-lastra/wodpredict-notas-coach
*   Notebook en Google Colab: https://colab.research.google.com/github/rick-lastra/WODPredictAI/blob/main/notebooks/Actividad_4.ipynb

## 👥 Contacto y Créditos
*   **Autor:** Ricardo Alejandro Lastra Dueñas
*   **Empresa (Caso de Uso):** UnderCover Fitness
*   **Fecha:** Octubre 2026
*   **Institución:** Universidad Tecmilenio (Gestión de Proyectos de IA)
