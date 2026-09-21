"""Preprocesamiento de documentos Markdown a un dataset JSONL de fragmentos.

Este módulo constituye la primera etapa del pipeline de Domain Adaptation. Lee
el documento fuente en formato Markdown, lo divide en fragmentos (*chunks*) de
longitud acotada y antepone a cada fragmento la sección jerárquica a la que
pertenece, de forma que el modelo conserve el contexto estructural del manual
durante el ajuste fino.

El resultado se escribe en formato JSONL, con un objeto ``{"text": ...}`` por
línea, listo para ser consumido por :mod:`04_particion_datos`.

Attributes:
    RUTA_BASE (Path): Directorio raíz del proyecto, resuelto a partir de la
        ubicación de este archivo.
    ENTRADA_DOC (Path): Ruta del documento Markdown de entrada.
    SALIDA_JSONL (Path): Ruta del archivo JSONL generado con todos los
        fragmentos.
    MAX_CHARS_PER_CHUNK (int): Longitud máxima en caracteres de cada fragmento,
        incluyendo el prefijo de sección.

Example:
    Ejecución desde la raíz del proyecto::

        $ python 03_procesar_documentos.py
"""

import json
import re
from pathlib import Path

RUTA_BASE    = Path(__file__).resolve().parent
ENTRADA_DOC  = RUTA_BASE / "00_documentos" / "01_documento.md"
SALIDA_JSONL = RUTA_BASE / "01_datos_procesados" / "dataset_completo.jsonl"
MAX_CHARS_PER_CHUNK = 400

def limpiar_dividir_texto_con_contexto(ruta_archivo: Path, max_chars: int):
    """Divide un documento Markdown en fragmentos contextualizados por sección.

    Recorre el documento párrafo a párrafo. Las líneas que comienzan con ``#``
    se interpretan como títulos y actualizan la sección activa en lugar de
    generar un fragmento. El resto de párrafos se emiten precedidos por el
    prefijo ``"[<sección>] "``.

    Si un párrafo con su prefijo excede ``max_chars``, se subdivide por
    oraciones mediante una expresión regular que corta únicamente tras ``.``,
    ``!`` o ``?`` seguidos de espacio y mayúscula, evitando así romper números
    decimales o abreviaturas. Cada subfragmento resultante conserva el prefijo
    de sección.

    Args:
        ruta_archivo (Path): Ruta del documento Markdown de entrada. Se lee con
            codificación UTF-8.
        max_chars (int): Longitud máxima en caracteres de cada fragmento,
            contando el prefijo de sección.

    Returns:
        list[str]: Lista de fragmentos de texto, cada uno prefijado con su
        sección de origen y sin espacios sobrantes en los extremos.

    Raises:
        FileNotFoundError: Si ``ruta_archivo`` no existe.
    """
    if not ruta_archivo.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {ruta_archivo}")

    with open(str(ruta_archivo), "r", encoding="utf-8") as f:
        contenido = f.read()

    parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]
    chunks   = []
    seccion_actual = "General"

    # Regex para dividir oraciones sin romper decimales ni perder el punto final
    patron_oraciones = r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])'

    for parrafo in parrafos:
        # Si el párrafo es un título, actualizamos el contexto
        if parrafo.startswith("#"):
            seccion_actual = parrafo.lstrip("#").strip()
            continue

        prefix = f"[{seccion_actual}] "

        # Si el párrafo cabe completo con el prefijo
        if len(prefix) + len(parrafo) <= max_chars:
            chunks.append(f"{prefix}{parrafo}")
        else:
            # División robusta por oraciones
            oraciones = [o.strip() for o in re.split(patron_oraciones, parrafo) if o.strip()]
            chunk_actual = prefix

            for oracion in oraciones:
                if len(chunk_actual) + len(oracion) + 1 <= max_chars:
                    chunk_actual += (" " + oracion) if chunk_actual != prefix else oracion
                else:
                    if chunk_actual != prefix:
                        chunks.append(chunk_actual.strip())
                    chunk_actual = f"{prefix}{oracion}"

            if chunk_actual != prefix:
                chunks.append(chunk_actual.strip())

    return chunks

def main():
    """Ejecuta el preprocesamiento completo del documento fuente.

    Fragmenta :data:`ENTRADA_DOC`, crea el directorio de salida si no existe y
    escribe cada fragmento como una línea JSON con la clave ``"text"`` en
    :data:`SALIDA_JSONL`. Los caracteres no ASCII se preservan literalmente
    (``ensure_ascii=False``).

    Returns:
        None: El resultado se persiste en disco y se informa por consola el
        número de fragmentos generados.

    Raises:
        FileNotFoundError: Propagada desde
            :func:`limpiar_dividir_texto_con_contexto` si el documento de
            entrada no existe.
    """
    print(f"Procesando archivo con Regex y contexto: {ENTRADA_DOC.name}...")
    chunks = limpiar_dividir_texto_con_contexto(ENTRADA_DOC, MAX_CHARS_PER_CHUNK)

    SALIDA_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(str(SALIDA_JSONL), "w", encoding="utf-8") as f:
        for chunk in chunks:
            linea = json.dumps({"text": chunk}, ensure_ascii=False)
            f.write(linea + "\n")

    print(f"¡Éxito! Se generaron {len(chunks)} fragmentos robustos en '{SALIDA_JSONL.name}'.")

if __name__ == "__main__":
    main()