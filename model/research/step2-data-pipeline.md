# Step 2 — Data Pipeline

## Goal

A reproducible script that descarga, limpia, tokeniza y divide el corpus en
train/val/test. Output: archivos listos para entrenar el modelo en Step 3 y 4.

## Data sources

| Source | Type | Size (sampled) | Code-switching | Access |
|---|---|---|---|---|
| Wikipedia es | Formal, clean | ~50k sentences | No | HuggingFace |
| Wikipedia en | Formal, clean | ~50k sentences | No | HuggingFace |
| OPUS-100 es-en | Conversational | ~30k sentences | Poco | HuggingFace |
| Miami Corpus | Spanglish real | ~1k utterances | Sí — nativo | Local (manual) |

**Por qué Wikipedia:** limpio, sin ruido, buen español/inglés base para el modelo.  
**Por qué OPUS-100:** más informal y conversacional — se acerca más a mensajes de texto.  
**Por qué Miami Corpus:** oro puro — conversaciones reales en Spanglish. Pequeño pero
se pondera más alto en el mix porque es exactamente el tipo de texto que OhMyWord necesita.

## Tokenizador

`Qwen/Qwen2.5-1.5B` — mismo que el teacher. El student hereda el vocabulario
directamente, lo que hace la destilación más directa (probabilidades comparables token a token).

## Formato de salida

Cada ejemplo es una ventana deslizante de 128 tokens:
- `input_ids`: tokens 0–126 (contexto)
- `labels`: tokens 1–127 (targets, shifted by 1)

Esta es la forma estándar de entrenar un causal LM — el modelo aprende a predecir
cada token dado los anteriores.

```
texto:        "me gusta el ceviche con limón"
input_ids:    [423, 8921, 201, 15043, 892, ...]   ← primeros 127 tokens
labels:       [8921, 201, 15043, 892, 3421, ...]  ← mismos pero shifteados 1 posición
```

## Splits

| Split | % | Uso |
|---|---|---|
| train | 90% | Entrenar el modelo |
| val | 5% | Evaluar durante entrenamiento (early stopping) |
| test | 5% | Evaluación final — NO tocar hasta el Step 4 |

El test set no se toca hasta tener el modelo entrenado. Usarlo antes contamina la evaluación.

## Cómo agregar el Miami Corpus

1. Descargar desde https://bangortalk.org.uk/speakers.php?c=miami
2. Guardar los archivos `.cha` en `model/data/raw/miami/`
3. El pipeline los detecta automáticamente y los incluye

## Output

```
model/data/processed/
├── train.jsonl    # ~90% de los ejemplos
├── val.jsonl      # ~5%
└── test.jsonl     # ~5%
```
