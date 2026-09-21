"""Evaluación comparativa del adaptador LoRA frente al modelo base.

Cuarta etapa del pipeline. Carga el modelo base cuantizado a 4 bits con el
adaptador LoRA acoplado y calcula la *loss* y la perplejidad sobre los
conjuntos de validación y prueba en dos condiciones:

1. **Modelo base puro**: el adaptador se desactiva temporalmente mediante
   ``PeftModel.disable_adapter()``.
2. **Modelo adaptado**: el adaptador LoRA permanece activo.

Comparar ambas condiciones sobre el mismo objeto de modelo elimina cualquier
diferencia atribuible a la cuantización o al hardware, de modo que la
variación observada se debe exclusivamente al adaptador. El script reporta
además la mejora porcentual en la *loss* de prueba.

Attributes:
    MODEL_BASE_ID (str): Identificador del modelo base en Hugging Face Hub.
    RUTA_BASE (Path): Directorio raíz del proyecto, resuelto a partir de la
        ubicación de este archivo.
    RUTA_LORA (Path): Directorio del adaptador LoRA entrenado.
    ARCHIVO_VAL (Path): Conjunto de validación en formato JSONL.
    ARCHIVO_TEST (Path): Conjunto de prueba en formato JSONL.

Example:
    Ejecución desde la raíz del proyecto::

        $ python 07_evaluar_lora.py
"""

import torch, math, json
from pathlib import Path
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

MODEL_BASE_ID = "Qwen/Qwen2.5-3B-Instruct"
RUTA_BASE     = Path(__file__).resolve().parent
RUTA_LORA     = RUTA_BASE / "02_modelos_entrenadatos" / "adaptador_Domain_Adaptation"
ARCHIVO_VAL   = RUTA_BASE / "01_datos_procesados" / "datos_val.jsonl"
ARCHIVO_TEST  = RUTA_BASE / "01_datos_procesados" / "datos_test.jsonl"

def calcular_perplejidad(modelo, tokenizer, ruta_jsonl, device):
    """Calcula la loss promedio y la perplejidad sobre un dataset JSONL.

    Evalúa el modelo muestra a muestra en modo de lenguaje causal, usando los
    propios ``input_ids`` como etiquetas. La *loss* de cada muestra se pondera
    por su número de tokens antes de promediar, de modo que el resultado es
    una media por token y no una media por muestra: los fragmentos largos
    pesan proporcionalmente más, tal como corresponde a la definición de
    perplejidad.

    La inferencia se realiza bajo ``torch.no_grad()``, sin acumular gradientes.

    Args:
        modelo (PreTrainedModel | PeftModel): Modelo de lenguaje causal ya
            cargado y en modo evaluación.
        tokenizer (PreTrainedTokenizer): Tokenizer correspondiente al modelo.
        ruta_jsonl (Path | str): Archivo JSONL cuyos registros contienen la
            clave ``"text"``. Las líneas en blanco se ignoran.
        device (str): Dispositivo de cómputo al que se mueven los tensores de
            entrada, por ejemplo ``"cuda"`` o ``"cpu"``.

    Returns:
        tuple[float, float]: Par ``(loss_promedio, perplejidad)``, donde la
        perplejidad es la exponencial de la *loss* promedio por token.

    Raises:
        FileNotFoundError: Si ``ruta_jsonl`` no existe.
        ZeroDivisionError: Si el dataset no contiene ninguna muestra válida.
    """
    with open(ruta_jsonl, "r", encoding="utf-8") as f:
        muestras = [json.loads(linea) for linea in f if linea.strip()]

    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for muestra in muestras:
            texto         = muestra["text"]
            inputs        = tokenizer(texto, return_tensors="pt").to(device)
            outputs       = modelo(**inputs, labels=inputs["input_ids"])
            loss          = outputs.loss.item()
            num_tokens    = inputs["input_ids"].size(1)
            total_loss   += loss * num_tokens
            total_tokens += num_tokens
            
    loss_promedio = total_loss / total_tokens
    perplejidad   = math.exp(loss_promedio)
    return loss_promedio, perplejidad

def main():
    """Ejecuta la evaluación comparativa y reporta las métricas por consola.

    Selecciona automáticamente GPU si está disponible, carga el modelo base
    cuantizado a 4 bits, acopla el adaptador LoRA y evalúa validación y prueba
    con el adaptador desactivado y activado. Imprime una tabla comparativa de
    *loss* y perplejidad junto con la mejora porcentual en la *loss* de prueba
    (un valor positivo indica que el adaptador reduce la *loss*).

    Returns:
        None: Las métricas se imprimen por consola; no se persiste nada en
        disco.

    Raises:
        FileNotFoundError: Si el adaptador LoRA o alguno de los conjuntos de
            evaluación no existe.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit           = True,
        bnb_4bit_quant_type    = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16 # Se recomienda bfloat16 para Qwen
    )

    print("Cargando tokenizer y modelo base...")
    tokenizer   = AutoTokenizer.from_pretrained(MODEL_BASE_ID, trust_remote_code=True)
    modelo_base = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = MODEL_BASE_ID,
        quantization_config           = bnb_config,
        device_map                    = "auto",
        trust_remote_code             = True
    )

    print("Acoplando adaptador LoRA...")
    modelo_lora = PeftModel.from_pretrained(modelo_base, RUTA_LORA)
    modelo_lora.eval()

    # 1. EVALUACIÓN CON LORA DESACTIVADO (MODELO BASE PURO)
    print("\nEvaluando Modelo Base Puro (LoRA desactivado)...")
    with modelo_lora.disable_adapter():
        val_loss_base, val_ppl_base   = calcular_perplejidad(modelo_lora, tokenizer, ARCHIVO_VAL, device)
        test_loss_base, test_ppl_base = calcular_perplejidad(modelo_lora, tokenizer, ARCHIVO_TEST, device)

    # 2. EVALUACIÓN CON LORA ACTIVADO (MODELO ADAPTADO)
    print("Evaluando Modelo con Adaptador LoRA...")
    val_loss_lora, val_ppl_lora   = calcular_perplejidad(modelo_lora, tokenizer, ARCHIVO_VAL, device)
    test_loss_lora, test_ppl_lora = calcular_perplejidad(modelo_lora, tokenizer, ARCHIVO_TEST, device)

    # 3. IMPRESIÓN DE RESULTADOS
    print("\n" + "="*65)
    print(f"{'MÉTRICA':<20} | {'MODELO BASE':<18} | {'MODELO LORA':<18}")
    print("="*65)
    print(f"{'Val Loss':<20} | {val_loss_base:<18.4f} | {val_loss_lora:<18.4f}")
    print(f"{'Val Perplejidad':<20} | {val_ppl_base:<18.4f} | {val_ppl_lora:<18.4f}")
    print("-" * 65)
    print(f"{'Test Loss':<20} | {test_loss_base:<18.4f} | {test_loss_lora:<18.4f}")
    print(f"{'Test Perplejidad':<20} | {test_ppl_base:<18.4f} | {test_ppl_lora:<18.4f}")
    print("="*65)

    # Cálculo de mejora porcentual en Test Loss
    mejora = ((test_loss_base - test_loss_lora) / test_loss_base) * 100
    print(f"\nMejora en Test Loss: {mejora:+.2f}%")

if __name__ == "__main__":
    main()