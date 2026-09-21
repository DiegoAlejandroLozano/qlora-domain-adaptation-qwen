"""Ajuste fino del modelo base mediante QLoRA para adaptación de dominio.

Etapa central del pipeline. Carga ``Qwen2.5-3B-Instruct`` cuantizado a 4 bits
con BitsAndBytes (NF4 + doble cuantización) y entrena sobre él un adaptador
LoRA usando el ``SFTTrainer`` de TRL. Solo se actualizan los pesos del
adaptador (aproximadamente entre 0.1 % y 0.5 % de los parámetros totales), lo
que permite ejecutar el entrenamiento en GPUs de gama media con ~6 GB de VRAM.

La configuración de entrenamiento evalúa y guarda un checkpoint al final de
cada época, conservando en disco únicamente el mejor modelo según
``eval_loss``. El adaptador final y el tokenizer se persisten en
:data:`DIR_SALIDA`.

Attributes:
    RUTA_BASE (Path): Directorio raíz del proyecto, resuelto a partir de la
        ubicación de este archivo.
    ENTRADA_DATA (Path): Conjunto de entrenamiento en formato JSONL.
    VAL_DATA (Path): Conjunto de validación en formato JSONL.
    DIR_SALIDA (Path): Directorio donde se guardan el adaptador LoRA
        entrenado, el tokenizer y los checkpoints.
    MODEL_ID (str): Identificador del modelo base en Hugging Face Hub.

Example:
    Ejecución desde la raíz del proyecto::

        $ python 06_ajuste_fino_QLoRA.py
"""

import os, torch
from pathlib      import Path
from datasets     import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)   
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl  import SFTTrainer, SFTConfig

# 1. Configuración de rutas y modelos
RUTA_BASE    = Path(__file__).resolve().parent
ENTRADA_DATA = RUTA_BASE / "01_datos_procesados" / "datos_train.jsonl"
VAL_DATA     = RUTA_BASE / "01_datos_procesados" / "datos_val.jsonl"
DIR_SALIDA   = RUTA_BASE / "02_modelos_entrenadatos" / "adaptador_Domain_Adaptation"
MODEL_ID     = "Qwen/Qwen2.5-3B-Instruct"

def main():
    """Orquesta el entrenamiento QLoRA de principio a fin.

    Ejecuta secuencialmente las siguientes fases:

    1. Carga de los conjuntos de entrenamiento y validación desde JSONL.
    2. Configuración de la cuantización a 4 bits (NF4, cómputo en ``bfloat16``,
       doble cuantización).
    3. Carga del tokenizer y del modelo base cuantizado, fijando el token de
       relleno al de fin de secuencia y preparando el modelo para
       entrenamiento en precisión reducida.
    4. Inyección del adaptador LoRA (``r=8``, ``alpha=16``, ``dropout=0.10``)
       sobre las siete proyecciones de atención y MLP del transformer.
    5. Definición de los hiperparámetros de entrenamiento ajustados a ~6 GB de
       VRAM: batch size 1 con acumulación de gradiente 2, gradient
       checkpointing, optimizador ``paged_adamw_8bit`` y ``learning_rate``
       de ``1e-4`` durante 2 épocas.
    6. Entrenamiento supervisado con ``SFTTrainer``, recuperando al final el
       checkpoint con menor ``eval_loss``.
    7. Guardado del adaptador y del tokenizer en :data:`DIR_SALIDA`.

    Returns:
        None: El adaptador entrenado se persiste en disco; el progreso se
        informa por consola.

    Raises:
        FileNotFoundError: Si :data:`ENTRADA_DATA` no existe.
    """
    if not ENTRADA_DATA.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {ENTRADA_DATA}")

    print(f"Cargando dataset dese: {ENTRADA_DATA.name}...")
    dataset = load_dataset("json", data_files={
        "train" : str(ENTRADA_DATA),
        "eval"  : str(VAL_DATA)
    })

    # 2. Configuración de cuantización a 4 bits  (BitsAndByte)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True
    )

    # 3. Carga del tokenizer  modelo base
    print(f"Cargando tokenizer y modelo base cuantizado...")
    tokenizer           = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = MODEL_ID,
        quantization_config           = bnb_config,
        device_map                    = "auto",
        trust_remote_code             = True
    )

    # Preparar el modelo para entrenamieento en precisión reducida
    model = prepare_model_for_kbit_training(model=model)
    model.enable_input_require_grads()

    # 4. Configuración del Adaptador LoRA
    peft_config = LoraConfig(
        r              = 8,
        lora_alpha     = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout   = 0.10,
        bias           = "none",
        task_type      = "CAUSAL_LM"
    )
    model = get_peft_model(model=model, peft_config=peft_config)
    model.print_trainable_parameters() # Muestra cuántos parámetros entrenaremos (~0.1% a 0.5%)

    # 5. Argumentos de entrenamiento ajustados para 6GB VRAM
    sft_configu = SFTConfig(
        output_dir                  = str(DIR_SALIDA),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        gradient_checkpointing      = True,
        learning_rate               = 1e-4,
        num_train_epochs            = 2,
        fp16                        = False,
        bf16                        = True,
        logging_steps               = 2,
        
        # --- ESTRATEGIA PARA QUEDARTE SOLO CON EL MEJOR MODELO ---
        eval_strategy               = "epoch",          # Evalúa al final de cada época (o pon "steps" y eval_steps)
        save_strategy               = "epoch",          # Debe coincidir con eval_strategy
        load_best_model_at_end      = True,             # Carga el mejor checkpoint al terminar
        metric_for_best_model       = "eval_loss",      # Selecciona el de menor loss en validación
        greater_is_better           = False,            # Queremos el loss más bajo
        save_total_limit            = 1,                # Elimina checkpoints antiguos, dejando máximo 1 en disco
        # ---------------------------------------------------------
        
        optim                       = "paged_adamw_8bit",
        report_to                   = "none",
        dataset_text_field          = "text",  
        max_length                  = 512,    
    )

    # 6. SFTTrainer (Supervised Fine-Tuning Trainer)
    trainer = SFTTrainer(
        model              = model,
        train_dataset      = dataset["train"],
        eval_dataset       = dataset["eval"],
        args               = sft_configu
    )

    # 7. Ejecución del entrenamiento
    print("\nIniciando entrenamiento de Etapa 1 (Domain Adaptation)...")
    trainer.train()

    # 8. Guardado del Adaptador LoRA
    print(f"\nGuardando adaptador LoRA en: {DIR_SALIDA}...")
    trainer.model.save_pretrained(str(DIR_SALIDA))
    tokenizer.save_pretrained(str(DIR_SALIDA))
    print("¡Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()