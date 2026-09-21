"""Análisis de la distribución de tokens del conjunto de entrenamiento.

Etapa de diagnóstico previa al ajuste fino. Tokeniza cada muestra del conjunto
de entrenamiento con el tokenizer del modelo base y reporta estadísticos
descriptivos (mínimo, máximo, promedio, mediana y percentiles 90/95/99).

A partir de la muestra más larga, sugiere un valor de ``max_length`` redondeado
a la siguiente potencia de dos, que es el parámetro a configurar en el
``SFTConfig`` de :mod:`06_ajuste_fino_QLoRA` para evitar truncamientos.

Attributes:
    RUTA_BASE (Path): Directorio raíz del proyecto, resuelto a partir de la
        ubicación de este archivo.
    ENTRADA_DATA (Path): Conjunto de entrenamiento en formato JSONL a analizar.
    MODEL_ID (str): Identificador del modelo en Hugging Face Hub del cual se
        toma el tokenizer.

Example:
    Ejecución desde la raíz del proyecto::

        $ python 05_analizar_tokens.py
"""

import numpy as np

from pathlib      import Path
from datasets     import load_dataset
from transformers import AutoTokenizer

# 1. Configuración de rutas y modelo
RUTA_BASE    = Path(__file__).resolve().parent
ENTRADA_DATA = RUTA_BASE / "01_datos_procesados" / "datos_train.jsonl"
MODEL_ID     = "Qwen/Qwen2.5-3B-Instruct"

def main():
    """Calcula y reporta la distribución de longitudes en tokens del dataset.

    Carga el conjunto de entrenamiento, tokeniza el campo ``"text"`` de cada
    muestra incluyendo tokens especiales, e imprime por consola una tabla con
    los estadísticos descriptivos y los percentiles de cobertura.

    Finalmente sugiere el valor de ``max_length`` a usar en el entrenamiento:
    la menor potencia de dos (entre 64 y 2048) que cubre la muestra más larga
    sin recortes.

    Returns:
        None: Los resultados se imprimen por consola; no se persiste nada en
        disco.

    Raises:
        FileNotFoundError: Si :data:`ENTRADA_DATA` no existe.
    """
    if not ENTRADA_DATA.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {ENTRADA_DATA}")

    print(f"Cargando dataset desde: {ENTRADA_DATA.name}...")
    dataset = load_dataset("json", data_files={"train": str(ENTRADA_DATA)})["train"]

    print(f"Cargando tokenizer ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_ID, trust_remote_code=True)

    print("Calculando longitud de tokens para cada muestra...\n")

    # Obtener el número de tokens por cada caampo "text"
    token_counts = [
        len(tokenizer.encode(ejemplo["text"], add_special_tokens=True)) 
        for ejemplo in dataset
    ]

    total_muestras = len(token_counts)
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    promedio = np.mean(token_counts)
    mediana = np.median(token_counts)
    p90 = np.percentile(token_counts, 90)
    p95 = np.percentile(token_counts, 95)
    p99 = np.percentile(token_counts, 99)

    print("=" * 50)
    print("        ANÁLISIS DE DISTRIBUCIÓN DE TOKENS       ")
    print("=" * 50)
    print(f"Total de muestras analizadas : {total_muestras}")
    print(f"Mínimo de tokens por muestra : {min_tokens}")
    print(f"Máximo de tokens por muestra : {max_tokens}")
    print(f"Promedio de tokens           : {promedio:.2f}")
    print(f"Mediana de tokens            : {mediana:.2f}")
    print("-" * 50)
    print("PERCENTILES (Cobertura del dataset):")
    print(f" - Percentil 90 (90% del dataset): <= {int(p90)} tokens")
    print(f" - Percentil 95 (95% del dataset): <= {int(p95)} tokens")
    print(f" - Percentil 99 (99% del dataset): <= {int(p99)} tokens")
    print("=" * 50)

    # Sugerencia automática de max_length en potencias de 2 (fácil de procesar por GPU)
    sugerencia_ideal = int(max_tokens)
    potencias = [64, 128, 256, 512, 1024, 2048]
    val_sugerido = next((p for p in potencias if p >= sugerencia_ideal), 2048)

    print(f"\nRECOMENDACIÓN:")
    print(f"-> Tu muestra más larga tiene {max_tokens} tokens.")
    print(f"-> Para conservar el 100% de tus textos sin recortes, ajusta `max_length = {val_sugerido}` en `SFTConfig`.")
    print("=" * 50)

if __name__ == "__main__":
    main()