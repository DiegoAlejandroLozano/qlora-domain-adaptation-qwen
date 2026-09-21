"""Inferencia interactiva y comparación cualitativa del adaptador LoRA.

Etapa final del pipeline. Complementa la evaluación cuantitativa de
:mod:`07_evaluar_lora` con una inspección cualitativa: dado un conjunto de
prompts que replican el formato del dataset de entrenamiento
(``"[<sección>] <texto inicial>"``), genera el autocompletado con el adaptador
LoRA activo y con el modelo base puro, e imprime ambas salidas una junto a
otra.

La generación usa *greedy decoding* (``do_sample=False``) para que las salidas
sean deterministas y la comparación entre ambos modelos sea reproducible.

Attributes:
    MODEL_ID (str): Identificador del modelo base en Hugging Face Hub.
    RUTA_BASE (Path): Directorio raíz del proyecto, resuelto a partir de la
        ubicación de este archivo.
    RUTA_LORA (Path): Directorio del adaptador LoRA entrenado.

Example:
    Ejecución desde la raíz del proyecto::

        $ python 08_inferencia_modelo.py
"""

import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"
RUTA_BASE = Path(__file__).resolve().parent
RUTA_LORA = RUTA_BASE / "02_modelos_entrenadatos" / "adaptador_Domain_Adaptation"

def cargar_modelo():
    """Carga el modelo base cuantizado a 4 bits con el adaptador LoRA acoplado.

    Configura la cuantización NF4 con cómputo en ``bfloat16``, carga el
    tokenizer fijando el token de relleno al de fin de secuencia, distribuye
    el modelo automáticamente entre los dispositivos disponibles
    (``device_map="auto"``) y acopla el adaptador almacenado en
    :data:`RUTA_LORA`. El modelo se devuelve en modo evaluación.

    Returns:
        tuple[PeftModel, PreTrainedTokenizer]: Par ``(modelo, tokenizer)``
        listo para generar texto. El modelo permite activar o desactivar el
        adaptador mediante ``disable_adapter()``.

    Raises:
        FileNotFoundError: Si el adaptador LoRA no existe en
            :data:`RUTA_LORA`.
    """
    print("Cargando tokenizer y modelo base cuantizado en 4 bits...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit           = True,
        bnb_4bit_quant_type    = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16
    )
    
    tokenizer           = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    modelo_base = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = MODEL_ID,
        quantization_config           = bnb_config,
        device_map                    = "auto",
        trust_remote_code             = True
    )

    print("Acoplando adaptador LoRA...")
    modelo = PeftModel.from_pretrained(modelo_base, RUTA_LORA)
    modelo.eval()

    return modelo, tokenizer

def probar_autocompletado(modelo, tokenizer, prompt: str, max_new_tokens: int = 100):
    """Genera y compara el autocompletado con y sin el adaptador LoRA.

    Ejecuta dos generaciones sobre el mismo prompt: la primera con el
    adaptador activo y la segunda con el adaptador desactivado mediante
    ``disable_adapter()``, que restituye el comportamiento del modelo base
    puro. Ambas usan *greedy decoding*, por lo que las diferencias observadas
    son atribuibles únicamente al adaptador.

    Args:
        modelo (PeftModel): Modelo con adaptador LoRA acoplado, en modo
            evaluación.
        tokenizer (PreTrainedTokenizer): Tokenizer correspondiente al modelo,
            con ``pad_token_id`` definido.
        prompt (str): Texto inicial a completar. Debe seguir el formato del
            dataset de entrenamiento, es decir ``"[<sección>] <texto>"``.
        max_new_tokens (int, optional): Número máximo de tokens a generar en
            cada una de las dos pasadas. Por defecto ``100``.

    Returns:
        None: Ambas continuaciones se imprimen por consola.
    """
    print(f"\nPROMPT DE ENTRADA:\n{prompt}")
    print("=" * 65)

    inputs = tokenizer(prompt, return_tensors="pt").to(modelo.device)

    # 1. Generación CON adaptador LoRA activado
    with torch.no_grad():
        out_lora = modelo.generate(
            **inputs, 
            max_new_tokens = max_new_tokens, 
            do_sample      = False, # Greedy decoding para evaluar la predicción más probable
            pad_token_id   = tokenizer.pad_token_id
        )
    texto_lora = tokenizer.decode(out_lora[0], skip_special_tokens=True)
    print(f"CON LORA:\n{texto_lora}\n")
    print("-" * 65)

    # 2. Generación DESACTIVANDO el adaptador LoRA (Modelo Base Puro)
    with modelo.disable_adapter():
        with torch.no_grad():
            out_base = modelo.generate(
                **inputs, 
                max_new_tokens = max_new_tokens, 
                do_sample      = False, 
                pad_token_id   = tokenizer.pad_token_id
            )
    texto_base = tokenizer.decode(out_base[0], skip_special_tokens=True)
    print(f"MODELO BASE (SIN LORA):\n{texto_base}\n")
    print("=" * 65)

def main():
    """Carga el modelo y ejecuta la batería de pruebas de autocompletado.

    Recorre una lista de prompts representativos del dominio técnico del
    documento fuente (embeddings, chunking, restricciones de grafo y reglas de
    consulta) y muestra, para cada uno, la comparación entre el modelo
    adaptado y el modelo base.

    Returns:
        None: Los resultados se imprimen por consola.

    Raises:
        FileNotFoundError: Propagada desde :func:`cargar_modelo` si el
            adaptador LoRA no existe.
    """
    modelo, tokenizer = cargar_modelo()

    # Prompts de prueba extraídos directamente del formato de tu dataset de entrenamiento
    prompts_prueba = [
        "[4.1 Especificación del Modelo de Embeddings] - Proveedor y Modelo:",
        "[5.1 Especificación del Fragmentado de Texto (Chunking)] - Estrategia de Ventana:",
        "[3.3 Restricciones e Índices] CREATE CONSTRAINT constraint_entidad_nombre IF NOT EXISTS",
        "[4.3 Reglas de Consulta e Inferencia] Filtro de Umbral de Similitud:"
    ]

    print("\nIniciando pruebas de autocompletado directo...")
    for i, prompt in enumerate(prompts_prueba, 1):
        print(f"\n--- PRUEBA {i} DE {len(prompts_prueba)} ---")
        probar_autocompletado(modelo, tokenizer, prompt)

if __name__ == "__main__":
    main()