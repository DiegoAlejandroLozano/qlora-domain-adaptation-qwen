# Domain Adaptation con QLoRA sobre Qwen2.5-3B-Instruct

Pipeline completo y reproducible de **adaptación de dominio** (*Domain Adaptation*) para
modelos de lenguaje ejecutados en local. El proyecto toma un manual técnico en Markdown,
lo transforma en un dataset de fragmentos contextualizados y entrena un adaptador **LoRA**
sobre el modelo `Qwen/Qwen2.5-3B-Instruct` cuantizado a 4 bits, de forma que el
entrenamiento completo cabe en una GPU de gama media (~6 GB de VRAM).

El objetivo no es enseñar al modelo a seguir instrucciones nuevas, sino **impregnarlo del
vocabulario, la terminología y el estilo redaccional de un dominio específico**. En este
caso, el corpus es un *Manual de Especificación Técnica de Arquitectura RAG e Ingesta de
Grafos de Conocimiento*: embeddings, estrategias de *chunking*, esquemas Pydantic,
restricciones Cypher y reglas de inferencia sobre grafos.

El repositorio incluye tanto el entrenamiento como la **verificación del resultado**:
evaluación cuantitativa por perplejidad y comparación cualitativa lado a lado entre el
modelo adaptado y el modelo base puro.

---

## Arquitectura del sistema

El pipeline es estrictamente secuencial. Cada script consume la salida del anterior, y el
prefijo numérico de los archivos indica el orden de ejecución.

```
00_documentos/01_documento.md        (corpus fuente en Markdown)
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 03_procesar_documentos.py                                       │
│ Fragmenta el Markdown en chunks de ≤400 caracteres y antepone   │
│ a cada uno su sección jerárquica: "[4.1 Embeddings] texto..."   │
└─────────────────────────────────────────────────────────────────┘
              │  dataset_completo.jsonl  (48 fragmentos)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 04_particion_datos.py                                           │
│ Baraja con semilla fija (SEED=42) y reparte 80/10/10            │
└─────────────────────────────────────────────────────────────────┘
              │  datos_train.jsonl (38) · datos_val.jsonl (4) · datos_test.jsonl (6)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 05_analizar_tokens.py               [ diagnóstico, opcional ]   │
│ Mide la distribución de tokens y sugiere el `max_length` óptimo │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 06_ajuste_fino_QLoRA.py                          [ ENTRENAR ]   │
│ Modelo base NF4 4-bit  →  adaptador LoRA  →  SFTTrainer (TRL)   │
│ Conserva únicamente el checkpoint con menor eval_loss           │
└─────────────────────────────────────────────────────────────────┘
              │  02_modelos_entrenadatos/adaptador_Domain_Adaptation/
              ▼
      ┌───────┴────────┐
      ▼                ▼
┌──────────────┐  ┌──────────────────────────────────────────────┐
│ 07_evaluar   │  │ 08_inferencia_modelo.py                      │
│ _lora.py     │  │ Autocompletado comparativo (greedy decoding) │
│ Perplejidad  │  │ CON LoRA  vs.  SIN LoRA                      │
│ base vs LoRA │  └──────────────────────────────────────────────┘
└──────────────┘
```

### Decisiones de diseño

| Decisión | Motivo |
| --- | --- |
| **Prefijo de sección en cada chunk** | Un fragmento suelto pierde su contexto jerárquico. Anteponer `[4.1 Especificación del Modelo de Embeddings]` le da al modelo una señal de en qué apartado del manual está, y permite luego usar ese mismo formato como prompt en inferencia. |
| **División por oraciones con regex** | El corte solo ocurre tras `.`, `!` o `?` seguidos de espacio y mayúscula (`(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])`), evitando romper números decimales, versiones (`Pydantic 2.x`) y abreviaturas. |
| **QLoRA (NF4 + doble cuantización)** | Permite entrenar un modelo de 3B parámetros en ~6 GB de VRAM. Solo se actualiza el adaptador: entre el 0.1 % y el 0.5 % de los pesos totales. |
| **`load_best_model_at_end` + `save_total_limit=1`** | Con un dataset pequeño el sobreajuste aparece rápido. Se conserva en disco un único checkpoint: el de menor `eval_loss`. |
| **Evaluar con `disable_adapter()`** | La comparación base vs. LoRA se hace sobre **el mismo objeto de modelo**, activando y desactivando el adaptador. Así la diferencia medida no se contamina con efectos de cuantización o hardware. |
| **Semilla fija en la partición** | Garantiza que train/val/test sean idénticos entre ejecuciones y entre máquinas. |

---

## Requisitos previos

- **Python 3.10** o superior (probado en 3.10.21).
- **GPU NVIDIA** con soporte CUDA y **mínimo 6 GB de VRAM**, con drivers y CUDA Toolkit
  instalados. `bitsandbytes` requiere GPU NVIDIA; la cuantización a 4 bits no funciona en CPU.
- Arquitectura Ampere o superior (RTX 30xx/40xx) recomendada, ya que el entrenamiento usa
  `bfloat16`.
- Conexión a internet en la primera ejecución: el modelo base (~6 GB) se descarga desde
  Hugging Face Hub y queda cacheado localmente.

### Instalación con Miniconda

El entorno se gestiona íntegramente con **Conda**. Si aún no lo tienes, instala
[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) para tu sistema operativo.

El archivo `environment.yml` está depurado de builds y paquetes específicos de plataforma,
por lo que reproduce el mismo entorno en **Windows, Linux y macOS**.

```bash
# 1. Clonar el repositorio
git clone https://github.com/DiegoAlejandroLozano/qlora-domain-adaptation-qwen.git
cd qlora-domain-adaptation-qwen

# 2. Crear el entorno de Conda a partir del archivo environment.yml
conda env create -f environment.yml

# 3. Activar el entorno
conda activate LLMs_local_env
```

Para verificar que el entorno quedó correctamente instalado:

```bash
conda env list
python -c "import torch, transformers, peft, trl; print('Entorno OK')"
```

#### Paso adicional para GPU NVIDIA (recomendado)

`environment.yml` declara `torch==2.5.1` desde PyPI para que la creación del entorno
funcione en cualquier sistema operativo. Esa build **no incluye CUDA en Windows ni macOS**.
Si dispones de una GPU NVIDIA y quieres entrenar con aceleración —imprescindible en la
práctica, ya que `bitsandbytes` exige CUDA— reinstala PyTorch con la build correspondiente
tras activar el entorno:

```bash
conda activate LLMs_local_env
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

Comprueba después que la GPU es visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> En macOS no existen builds de CUDA. El entorno se crea sin problemas y los pasos 1 a 3 del
> pipeline (preprocesamiento, partición y análisis de tokens) funcionan con normalidad, pero
> el entrenamiento y la inferencia cuantizada (pasos 4 a 6) requieren GPU NVIDIA.

Para actualizar el entorno si `environment.yml` cambia, o para eliminarlo por completo:

```bash
conda env update -f environment.yml --prune
conda remove -n LLMs_local_env --all
```

Versiones verificadas en este proyecto:

| Paquete | Versión |
| --- | --- |
| `torch` | 2.5.1 (probado con la build `+cu121`) |
| `transformers` | 5.17.0 |
| `peft` | 0.21.0 |
| `trl` | 1.13.0 |
| `datasets` | 5.0.1 |
| `accelerate` | 1.15.0 |
| `bitsandbytes` | 0.50.2 |
| `numpy` | 2.2.6 |

---

## Guía de uso rápido

Ejecuta los scripts **en orden** desde la raíz del proyecto. Todas las rutas se resuelven
de forma relativa a la ubicación de los archivos, así que no es necesario configurar nada.

```bash
# 1. Fragmentar el documento fuente  →  01_datos_procesados/dataset_completo.jsonl
python 03_procesar_documentos.py

# 2. Dividir en train / validación / test  (80 / 10 / 10, semilla 42)
python 04_particion_datos.py

# 3. (Opcional) Analizar la distribución de tokens y obtener el max_length sugerido
python 05_analizar_tokens.py

# 4. Entrenar el adaptador LoRA  →  02_modelos_entrenadatos/adaptador_Domain_Adaptation/
python 06_ajuste_fino_QLoRA.py

# 5. Evaluar: perplejidad del modelo base vs. el modelo adaptado
python 07_evaluar_lora.py

# 6. Inspección cualitativa: autocompletado con y sin LoRA
python 08_inferencia_modelo.py
```

### Salida esperada

**Paso 1 — fragmentación**

```
Procesando archivo con Regex y contexto: 01_documento.md...
¡Éxito! Se generaron 48 fragmentos robustos en 'dataset_completo.jsonl'.
```

**Paso 2 — partición**

```
Total registros: 48
 -> Entrenar (Train): 38 muestras
 -> Validación (Val): 4 muestras
 -> Prueba (Test):    6 muestras
```

**Paso 3 — análisis de tokens**

```
==================================================
        ANÁLISIS DE DISTRIBUCIÓN DE TOKENS
==================================================
Total de muestras analizadas : 38
...
RECOMENDACIÓN:
-> Para conservar el 100% de tus textos sin recortes, ajusta `max_length = 512` en `SFTConfig`.
```

**Paso 5 — evaluación comparativa**

```
=================================================================
MÉTRICA              | MODELO BASE        | MODELO LORA
=================================================================
Val Loss             | ...                | ...
Val Perplejidad      | ...                | ...
-----------------------------------------------------------------
Test Loss            | ...                | ...
Test Perplejidad     | ...                | ...
=================================================================

Mejora en Test Loss: +X.XX%
```

Un valor **positivo** en «Mejora en Test Loss» indica que el adaptador reduce la *loss*
respecto al modelo base, es decir, que la adaptación de dominio funcionó.

### Personalizar el dominio

Para adaptar el modelo a otro corpus:

1. Reemplaza `00_documentos/01_documento.md` por tu propio documento Markdown, usando
   encabezados (`#`, `##`, `###`) para delimitar las secciones.
2. Vuelve a ejecutar los pasos 1 a 4.
3. Ajusta los `prompts_prueba` de `08_inferencia_modelo.py` para que reflejen las secciones
   de tu nuevo documento.

### Hiperparámetros principales

Definidos en `06_ajuste_fino_QLoRA.py`:

| Parámetro | Valor | Nota |
| --- | --- | --- |
| Modelo base | `Qwen/Qwen2.5-3B-Instruct` | Cambiable vía `MODEL_ID` |
| Cuantización | NF4, 4 bits, doble cuantización | Cómputo en `bfloat16` |
| LoRA `r` / `alpha` / `dropout` | 8 / 16 / 0.10 | |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Atención + MLP |
| `learning_rate` | `1e-4` | |
| Épocas | 2 | |
| Batch efectivo | 2 | `batch_size=1` × `grad_accum=2` |
| `max_length` | 512 | Sugerido por el paso 3 |
| Optimizador | `paged_adamw_8bit` | Reduce el pico de VRAM |

### Resolución de problemas

| Síntoma | Solución |
| --- | --- |
| `CUDA out of memory` | Reduce `max_length` a 256 o incrementa `gradient_accumulation_steps` manteniendo `per_device_train_batch_size=1`. |
| `bfloat16 is not supported` | Tu GPU es anterior a Ampere: cambia `bf16=True` por `fp16=True` en el `SFTConfig`. |
| `FileNotFoundError` en el dataset | Ejecutaste un script fuera de orden; corre primero los pasos 1 y 2. |

---

## Estructura del proyecto

```
02_Domain_Adaptation/
│
├── 00_documentos/
│   └── 01_documento.md                  # Corpus fuente: manual técnico en Markdown
│
├── 01_datos_procesados/                 # Generado por los pasos 1 y 2
│   ├── dataset_completo.jsonl           # 48 fragmentos contextualizados
│   ├── datos_train.jsonl                # 38 muestras (80 %)
│   ├── datos_val.jsonl                  #  4 muestras (10 %)
│   └── datos_test.jsonl                 #  6 muestras (10 %)
│
├── 02_modelos_entrenadatos/             # Generado por el paso 4
│   └── adaptador_Domain_Adaptation/
│       ├── adapter_config.json          # Configuración del adaptador LoRA
│       ├── adapter_model.safetensors    # Pesos entrenados del adaptador
│       ├── tokenizer.json               # Tokenizer serializado
│       ├── tokenizer_config.json
│       ├── chat_template.jinja
│       └── checkpoint-38/               # Mejor checkpoint (menor eval_loss)
│
├── 03_procesar_documentos.py            # Markdown → chunks JSONL contextualizados
├── 04_particion_datos.py                # Partición reproducible 80/10/10
├── 05_analizar_tokens.py                # Diagnóstico de longitudes y max_length
├── 06_ajuste_fino_QLoRA.py              # Entrenamiento QLoRA con SFTTrainer
├── 07_evaluar_lora.py                   # Perplejidad: modelo base vs. LoRA
├── 08_inferencia_modelo.py              # Autocompletado comparativo
│
├── environment.yml                      # Entorno Conda multiplataforma
└── README.md                            # Este archivo
```

> **Nota sobre el control de versiones:** los directorios `01_datos_procesados/` y
> `02_modelos_entrenadatos/` contienen artefactos regenerables, y `adapter_model.safetensors`
> junto con los checkpoints pueden ocupar cientos de megabytes. Conviene excluirlos del
> repositorio mediante `.gitignore` y versionar únicamente el documento fuente y los scripts.

---

## Licencia y modelo base

El modelo base `Qwen/Qwen2.5-3B-Instruct` se distribuye bajo la licencia Apache 2.0 de Qwen.
El adaptador LoRA entrenado es un artefacto derivado y queda sujeto a los términos de esa
misma licencia.
