"""Partición del dataset de fragmentos en conjuntos de train, validación y test.

Segunda etapa del pipeline de Domain Adaptation. Toma el archivo JSONL
generado por :mod:`03_procesar_documentos`, baraja los registros con una
semilla fija y los reparte en tres archivos independientes.

La semilla se fija a nivel de módulo para garantizar que la partición sea
reproducible entre ejecuciones y entre máquinas.

Attributes:
    SEED (int): Semilla del generador aleatorio, aplicada al importar el
        módulo para asegurar reproducibilidad.
    RUTA_BASE (Path): Directorio raíz del proyecto, resuelto a partir de la
        ubicación de este archivo.
    ENTRADA_JSONL (Path): Dataset completo de entrada en formato JSONL.
    DIR_SALIDA (Path): Directorio donde se escriben los tres archivos
        resultantes.

Example:
    Ejecución desde la raíz del proyecto::

        $ python 04_particion_datos.py
"""

import json
import random
from pathlib import Path

# Configuración de semillas para reproducibilidad
SEED = 42
random.seed(SEED)

RUTA_BASE     = Path(__file__).resolve().parent
ENTRADA_JSONL = RUTA_BASE / "01_datos_procesados" / "dataset_completo.jsonl"
DIR_SALIDA    = RUTA_BASE / "01_datos_procesados"

def dividir_dataset(archivo_entrada, train_ratio=0.8, val_ratio=0.1):
    """Baraja un dataset JSONL y lo divide en train, validación y test.

    Los registros se mezclan usando la semilla global :data:`SEED` y se cortan
    secuencialmente según las proporciones indicadas. El conjunto de prueba
    recibe el remanente, es decir, aproximadamente
    ``1 - train_ratio - val_ratio`` del total.

    Escribe tres archivos en :data:`DIR_SALIDA`: ``datos_train.jsonl``,
    ``datos_val.jsonl`` y ``datos_test.jsonl``.

    Args:
        archivo_entrada (Path | str): Ruta del archivo JSONL de entrada. Las
            líneas en blanco se ignoran.
        train_ratio (float, optional): Fracción del total destinada a
            entrenamiento. Por defecto ``0.8``.
        val_ratio (float, optional): Fracción del total destinada a
            validación. Por defecto ``0.1``.

    Returns:
        None: Los conjuntos se persisten en disco y se informa por consola el
        tamaño de cada partición.

    Raises:
        FileNotFoundError: Si ``archivo_entrada`` no existe.
        json.JSONDecodeError: Si alguna línea no es JSON válido.
    """
    # Cargar todos los registros
    with open(archivo_entrada, "r", encoding="utf-8") as f:
        lineas = [json.loads(linea) for linea in f if linea.strip()]
    
    # Mezclar aleatoriamente
    random.shuffle(lineas)
    
    total = len(lineas)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = lineas[:train_end]
    val_data = lineas[train_end:val_end]
    test_data = lineas[val_end:]
    
    # Guardar los tres archivos
    def guardar_jsonl(datos, ruta):
        """Serializa una lista de registros a un archivo JSONL en UTF-8.

        Args:
            datos (list[dict]): Registros a escribir, uno por línea.
            ruta (Path | str): Ruta del archivo de salida. Se sobrescribe si
                ya existe.

        Returns:
            None
        """
        with open(ruta, "w", encoding="utf-8") as f:
            for item in datos:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    guardar_jsonl(train_data, DIR_SALIDA / "datos_train.jsonl")
    guardar_jsonl(val_data, DIR_SALIDA / "datos_val.jsonl")
    guardar_jsonl(test_data, DIR_SALIDA / "datos_test.jsonl")
    
    print(f"Total registros: {total}")
    print(f" -> Entrenar (Train): {len(train_data)} muestras")
    print(f" -> Validación (Val): {len(val_data)} muestras")
    print(f" -> Prueba (Test):    {len(test_data)} muestras")

if __name__ == "__main__":
    dividir_dataset(ENTRADA_JSONL)