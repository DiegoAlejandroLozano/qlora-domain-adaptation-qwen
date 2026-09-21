# Manual de Especificación Técnica Ampliado: Arquitectura RAG e Ingesta de Grafos de Conocimiento

## 1. Contexto General del Sistema y Objetivos de Negocio

El presente documento establece los estándares arquitectónicos, técnicos y de implementación para la plataforma de *Retrieval-Augmented Generation* (RAG) dentro de la organización. El objetivo central de esta plataforma es transformar documentos corporativos no estructurados (reportes, manuales, políticas y arquitecturas de software) en una base de conocimiento estructurada, auditable y libre de alucinaciones.

Para superar las limitaciones tradicionales de los sistemas RAG basados únicamente en búsqueda vectorial —los cuales sufren al responder preguntas que requieren razonamiento multisalbo (*multi-hop reasoning*) o comprensión de relaciones jerárquicas complejas—, esta arquitectura adopta un enfoque **GraphRAG Híbrido**.

### 1.1 Objetivos Clave de la Arquitectura
* **Recuperación Multimodal Interna**: Combinar la similitud semántica en espacios vectoriales densos con la precisión estructural de un grafo de conocimiento property graph.
* **Trazabilidad de la Información**: Cada respuesta generada por el LLM debe estar vinculada explícitamente a un conjunto de nodos en el grafo y a los fragmentos de texto fuente depositados en el índice vectorial.
* **Integridad de Datos mediante Esquemas Rígidos**: Garantizar que el conocimiento extraído de manera no supervisada o semi-supervisada cumpla con estrictas validaciones de tipos antes de su persistencia.

---

## 2. Esquema de Entidades, Tipado de Datos y Validación con Pydantic

Toda información extraída desde texto no estructurado debe atravesar un proceso riguroso de estructuración y normalización. Para evitar inconsistencias sintácticas, variaciones ortográficas y duplicidad conceptual, se exige el uso de la librería `Pydantic` en su versión 2.x para la definición y validación del esquema de datos.

### 2.1 Definición de la Entidad Base (`EntidadDetalle`)
No se permite el uso de cadenas simples (*primitive strings*) para representar nodos en el grafo. Cada entidad debe ser un objeto enriquecido con metadatos.

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class EntidadDetalle(BaseModel):
    `nombre_canonico`: str = Field(
        ..., 
        description="Nombre normalizado en mayúsculas/minúsculas estándar, sin sufijos innecesarios."
    )
    categoria: str = Field(
        ..., 
        description="Categoría taxonómica del nodo (ej. ORGANIZACIÓN, TECNOLOGÍA, CONCEPTO, PROCESO)."
    )
    descripcion: Optional[str] = Field(
        None, 
        description="Resumen conciso del propósito o definición de la entidad en el contexto leído."
    )
    propiedades: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadatos clave-valor adicionales asociados a la entidad."
    )
```

### 2.2 Definición de la Relación Directed (TripletaEsquema)
Las relaciones entre entidades definen los bordes orientados del grafo.


```python
class TripletaEsquema(BaseModel):
    subject: EntidadDetalle = Field(..., description="Nodo origen de la relación.")
    predicate: str = Field(
        ..., 
        description="Verbo o conector en mayúsculas sostenidas que define la acción (ej. UTILIZA_PARA, DEPENDE_DE)."
    )
    object: EntidadDetalle = Field(..., description="Nodo destino de la relación.")
    confianza: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Puntaje de certeza asignado por el extractor."
    )
```

### 2.3 Normalización y Fusión por Embeddings 

Antes de la creación de tripletas, los nombres de las entidades deben someterse a un proceso de resolución de entidad (Entity Resolution):

1. Generación de Vector: Se calcula el embedding del campo `nombre_canonico`.
2. Búsqueda de Similitud: Se consulta en memoria o en caché si existe una entidad cuya distancia de coseno con el candidato sea mayor a 0.92.
3. Fusión de Nodos: Si la coincidencia supera el umbral, se adopta el `nombre_canonico` existente para mantener la unicidad estructural en la base de datos de grafos.

## 3. Persistencia y Operaciones en Neo4j (Base de Datos de Grafos)

La persistencia del conocimiento derivado de la extracción se realiza sobre un clúster de Neo4j Enterprise Edition. Toda interacción debe ejecutarse dentro del contexto de transacciones atómicas escritas en Cypher.

### 3.1 Reglas de Ingesta y Modelado de Datos

- Nodos: El valor del campo categoria en la estructura EntidadDetalle se mapea directamente como la etiqueta principal (Label) del nodo en Neo4j.
- Propiedades de Nodos: Cada nodo debe incluir obligatoriamente los atributos `nombre_canonico`, `fecha_creacion` y `ultima_actualizacion`.
- Relaciones: El campo predicate de la TripletaEsquema define el tipo de relación en Cypher. Todas las relaciones son dirigidas ($Subject \rightarrow Object$).

### 3.2 Estrategia Antipatrones y Prevención de Duplicados

Queda estrictamente prohibido el uso de la sentencia CREATE para la inserción de nodos y bordes. En su lugar, se requiere el patrón MERGE combinado con bloques ON CREATE SET y ON MATCH SET.

Consulta Cypher Estándar de Ingesta:

```cypher
MERGE (s:Entidad {`nombre_canonico`: $subject_nombre})
ON CREATE SET s.categoria = $subject_categoria, s.creado_el = datetime()
ON MATCH SET s.ultima_actualizacion = datetime()

MERGE (o:Entidad {`nombre_canonico`: $object_nombre})
ON CREATE SET o.categoria = $object_categoria, o.creado_el = datetime()
ON MATCH SET o.ultima_actualizacion = datetime()

MERGE (s)-[r:RELACION {tipo: $predicado}]->(o)
ON CREATE SET r.origen_documento = $doc_id, r.fecha_ingesta = datetime()
```

### 3.3 Restricciones e Índices

Para asegurar el rendimiento en lecturas de grafos extensos, se deben definir las siguientes restricciones en la instancia de Neo4j previa a la ingesta:

```cypher
CREATE CONSTRAINT constraint_entidad_nombre IF NOT EXISTS
FOR (n:Entidad) REQUIRE n.`nombre_canonico` IS UNIQUE;

CREATE INDEX index_entidad_categoria IF NOT EXISTS
FOR (n:Entidad) ON (n.categoria);
```

## 4. Almacenamiento Vectorial e Índices en Pinecone

Para complementar la estructura del grafo con la recuperación contextual del texto original, el pipeline indexa de forma paralela los fragmentos de texto en la plataforma de vectores administrada Pinecone.

### 4.1 Especificación del Modelo de Embeddings

- Proveedor y Modelo: OpenAI text-embedding-3-small.
- Dimensionalidad: 1536 dimensiones.
- Tratamiento de Dimensiones: Se prohíbe el uso de técnicas de reducción dimensional (Matryoshka Representation Learning) para mantener el nivel máximo de precisión semántica en los dominios técnicos.
- Normalización: Los vectores entregados por la API de OpenAI deben insertarse directamente sin modificaciones adicionales, ya que vienen pre-normalizados con norma unitaria.

### 4.2 Parámetros del Índice en Pinecone

- Métrica de Distancia: Coseno (cosine).
- Pod Spec / Serverless: Configuración basada en arquitectura Serverless sobre AWS (us-east-1).
- Estructura del Payload (Metadata):

```json
{
  "documento_id": "doc_hash_md5",
  "chunk_id": "doc_hash_md5_001",
  "texto_contenido": "Contenido crudo del bloque...",
  "offset_inicio": 0,
  "offset_fin": 400
}
```

### 4.3 Reglas de Consulta e Inferencia

Durante la fase de recuperación (Retrieval phase) del RAG:

1. El término de búsqueda del usuario es convertido a vector mediante text-embedding-3-small.
2. Se consulta el índice de Pinecone solicitando los $K=10$ vecinos más cercanos.
3. Filtro de Umbral de Similitud: Se aplica un filtrado estricto donde solo se conservan los vectores cuyo score de similitud con respecto a la consulta sea $\ge 0.82$.
4. Todos los fragmentos con un puntaje $< 0.82$ son descartados del contexto enviado al LLM, reduciendo significativamente la presencia de alucinaciones o respuestas irrelevantes.

## 5. Arquitectura del Pipeline de Ingesta, Rutas de Archivos y Manejo de Errores

El pipeline opera bajo un modelo de procesamiento orientado a lotes (batch pipeline) coordinado mediante scripts locales en Python.

┌──────────────────────┐
│  00_documentos/      │
│  (Markdown, PDF, TXT)│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Validación Hash MD5 │ ──[Ya procesado]──► [Ignorar Documento]
└──────────┬───────────┘
           │ [Nuevo Documento]
           ▼
┌──────────────────────┐
│ Chunking (400 chars) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Validaciones Pydantic│ ──[Fallo de Validación]──► [Abortar Lote & Rollback]
└──────────┬───────────┘
           │ [Lote Válido]
           ├───────────────────────────────┐
           ▼                               ▼
┌──────────────────────┐       ┌──────────────────────┐
│ Persistencia Neo4j   │       │ Indexación Pinecone  │
│ (Cypher MERGE)       │       │ (Vectors 1536d)      │
└──────────────────────┘       └──────────────────────┘

### 5.1 Especificación del Fragmentado de Texto (Chunking)

- Ruta de Ingesta Local: `00_documentos/`
- Estrategia de Ventana: Tamaño de bloque fijo de 400 caracteres con un solapamiento (overlap) de 50 caracteres.
- Delimitadores Primitivos: La separación debe priorizar los saltos de línea doble (\n\n), saltos de línea simples (\n) y puntos seguidos (. ) para evitar partir palabras a la mitad.

### 5.2 Control de Estado e Idempotencia

Para evitar reprocesar archivos inalterados, el sistema calcula un hash MD5 del contenido binario del archivo fuente antes de iniciar la lectura. El hash se verifica contra una base de datos local SQLite o archivo manifiesto (.ingest_registry.json). Si el hash ya existe, el proceso omite dicho documento.

### 5.3 Tolerancia a Fallos y Manejo de Excepciones

1. Transaccionalidad en Memoria: Todas las tripletas y fragmentos de un documento deben ser parseados y validados completamente en objetos Pydantic antes de realizar llamadas a las APIs o motores de bases de datos.

2. Política de Aborto: Si una sola muestra del lote falla en la validación contra EntidadDetalle o TripletaEsquema, la operación completa del archivo se detiene de inmediato.

3. Rollback e Aislamiento: No se debe enviar ningún nodo a Neo4j ni ningún vector a Pinecone si el paso de validación en memoria no obtiene una tasa de éxito del 100%.

## 6. Monitoreo, Auditoría de Calidad y Registro de Logs

Para mantener el sistema operativo en un entorno de producción, la infraestructura registra métricas clave durante el ciclo de vida de la ingesta y consultas.

### 6.1 Métricas de Ingesta

- Tiempo Total por Documento: Registro en milisegundos desde la lectura del archivo en 00_documentos/ hasta la confirmación de escritura en Neo4j y Pinecone.

- Tasa de Errores de Validación: Porcentaje de tripletas rechazadas por el modelo Pydantic por archivo procesado.

- Densidad del Grafo: Proporción entre el número de bordes y nodos generados por bloque de 400 caracteres.

### 6.2 Estrategia de Logging

Los eventos del sistema se deben estructurar en formato JSON para facilitar su ingesta por herramientas de monitoreo (como Datadog, ELK o CloudWatch):

```json
{
  "timestamp": "2026-09-21T10:00:00Z",
  "level": "ERROR",
  "component": "PydanticValidator",
  "document": "00_documentos/manual_arquitectura.md",
  "error_message": "1 validation error for TripletaEsquema -> predicate: field required",
  "action": "TRANSACTION_ABORTED"
}
```


## 7. Estrategia de Mantenimiento y Recuperación ante Desastres

### 7.1 Respaldos de la Base de Datos de Grafos

Se deben realizar respaldos diarios automatizados de Neo4j utilizando las herramientas nativas de neo4j-admin backup. Estos respaldos se deben sincronizar con almacenamiento en la nube (AWS S3) con políticas de retención de 30 días.

### 7.2 Reconstrucción del Índice Vectorial

Dado que Pinecone actúa como un almacén secundario derivado de los textos originales, el sistema debe incluir una rutina de recuperación (reindex_pinecone.py) capaz de leer los archivos fuente procesados y regenerar el índice de Pinecone por completo en caso de corrupción o pérdida de conectividad con la plataforma Serverless.

## 8. Consideraciones de Seguridad y Privacidad de Datos

- Manejo de Secretos: Todas las credenciales de conexión (claves de API de OpenAI, credenciales de Pinecone y contraseñas de Neo4j) deben inyectarse exclusivamente a través de variables de entorno (.env) usando pydantic-settings.

- Sanitización de Datos: El proceso de ingesta debe filtrar y enmascarar automáticamente cualquier dato de identificación personal (PII) detectado dentro de los textos fuente utilizando expresiones regulares predefinidas antes de generar los embeddings vectoriales o enviar datos a la base de datos de grafos.