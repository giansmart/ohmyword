# Step 1 — Task Definition & Evaluation Metrics

## The task

**Next-word prediction**: given the last N tokens of context, predict the top-3 most likely next words.

This maps directly to the keyboard suggestion bar — 3 tappable word suggestions that appear
as the user types. After tapping one, the model runs again to predict the next 3.

This is NOT:
- Mid-word completion (autocorrect) — different problem, out of scope for now
- Full-phrase prediction — harder, higher latency, not needed for MVP

## Input / Output spec

| | |
|---|---|
| Input | Last 3–5 sentences of user context + current typed text |
| Output | Top-3 next word candidates + their probabilities |
| Vocabulary | ~15,000 tokens (bilingual es/en, following Apple's keyboard benchmark) |
| Context window | 128 tokens — covers ~5 sentences of conversational context |

Why 128 tokens: sentence-level context (20 tokens) is ambiguous for many predictions.
Conversational context changes the picture entirely:

```
Oye, mañana después de la playa        ← sentence -3
vamos por unas chelas y comemos algo   ← sentence -2
se me antoja comida marina             ← sentence -1
me gustaría comer un rico ___          ← current
→ cevichito >> arroz chaufa > helado   (context makes it obvious)
```

Without those 3 prior sentences, all 3 suggestions are equally valid.
With them, the model has a strong signal toward seafood.

**iOS note:** the keyboard only has native access to the current text field. To get
previous sentences, we maintain a local rolling buffer of the user's recent typed text
(last 5 sentences, stored on-device). This buffer feeds the model and also serves
the personalization pipeline in Step 7.

## Evaluation metrics

| Metric | Type | Target | Why |
|---|---|---|---|
| **Top-3 accuracy** | Quality (primary) | > 30% on test set | Is the correct next word in the 3 suggestions? |
| **Top-1 accuracy** | Quality (secondary) | > 15% on test set | Is the #1 suggestion correct? |
| **Perplexity** | Quality (training signal) | As low as possible | Standard LM metric — lower = better |
| **Latency** | Hard constraint | < 50 ms on-device | Suggestion must appear before user notices |
| **Model size** | Hard constraint | < 20 MB | App Store download psychology |

Note on accuracy targets: these are minimum thresholds to beat the n-gram baseline (Step 3),
not final product targets. A production keyboard with 30% top-3 accuracy is already useful —
the user only needs to tap instead of type.

## Hand-curated test suite

These sentences cover the core use cases: pure Spanish, pure English, code-switching, informal,
formal, and different topics. Each has 3 acceptable next words (ground truth).

### Pure Spanish
| Context | Acceptable completions |
|---|---|
| "mañana voy a ir al" | trabajo, gym, mercado |
| "me gusta el ceviche con" | limón, ají, mariscos |
| "nos vemos en el" | parque, trabajo, aeropuerto |
| "no puedo creer que" | esto, eso, hayas |
| "qué rico está el" | ceviche, pollo, arroz |
| "te llamo más" | tarde, temprano, luego |
| "ya casi llego al" | trabajo, hotel, aeropuerto |
| "se me olvidó el" | teléfono, cargador, pasaporte |

### Pure English
| Context | Acceptable completions |
|---|---|
| "I need to send that" | email, message, report |
| "let me know if you" | can, want, need |
| "are you coming to the" | meeting, party, office |
| "I'll be there in" | minutes, seconds, a |
| "don't forget to" | bring, send, check |

### Code-switching (es/en mix)
| Context | Acceptable completions |
|---|---|
| "oye I need to send that email para" | mañana, hoy, el |
| "el meeting de mañana lo" | cancelaron, movieron, confirmo |
| "el jefe me dijo que el deadline es" | hoy, mañana, viernes |
| "voy a hacer el checkout del" | hotel, repo, proyecto |
| "let me know si puedes" | venir, ir, mañana |
| "no mames ese partido estuvo" | increíble, brutal, buenísimo |
| "ya mandé el PR para" | revisión, review, aprobación |
| "te mando el link del" | repo, doc, meet |

### Informal / slang
| Context | Acceptable completions |
|---|---|
| "no manches qué" | rollo, onda, locura |
| "está muy" | bien, bueno, chido |
| "qué onda con el" | trabajo, rollo, plan |
| "ya wey se" | acabó, fue, cancelo |

### Conversational context (multi-sentence)
These cases are only solvable correctly with prior context. They validate that the model
actually uses the rolling buffer, not just the current sentence.

| Prior context (last 3 sentences) | Current | Expected top-1 |
|---|---|---|
| "mañana playa / chelas y comemos / se me antoja comida marina" | "me gustaría comer un rico ___" | cevichito |
| "terrible el tráfico / tardé 2 horas / llegué agotado" | "lo único que quiero es ___" | dormir |
| "llevamos 3 sprints sin deploy / el repo está lleno de bugs / el team está quemado" | "hay que hablar con ___" | product / el jefe |
| "voy al gym / necesito energía / me tomé un café" | "a ver si puedo con el ___" | entreno / workout |
| "cumple de mi mamá / le compré flores / tenemos reserva en" | "el restaurante queda en ___" | [location word] |

---

## What "beats the baseline" means

At Step 3 we train an n-gram model and record its top-3 accuracy on this test suite.
Every neural model in Step 4+ must exceed that number. If it doesn't, it's not worth
the added complexity and size.
