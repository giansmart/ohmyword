# Step 0 — Model Candidates Research

**Goal:** Identify viable model architectures for OhMyWord's core next-word prediction task,
filtered against three hard constraints:

| Constraint | Target |
|---|---|
| Size | < 20 MB on-device |
| Latency | < 50 ms per inference (mobile CPU) |
| Quality | Handles bilingual es/en code-switching + personal vocabulary |

---

## What we're actually building

Before comparing models, the task must be clear: **causal next-word prediction**.

Given the last N tokens typed by the user → predict the top-3 most likely next words.

This rules out encoder-only architectures (BERT family) immediately — masked LM is the wrong
task. We need **autoregressive / causal LM** models or sequence models trained specifically
for next-word prediction.

---

## Reference benchmarks (production keyboards)

These are real shipped products, not research. They define what's acceptable.

### Gboard (Google, 2018–present)
- Architecture: CIFG-LSTM (Coupled Input-Forget Gate, a lightweight LSTM variant)
- Parameters: 1.4M
- Size: **1.4 MB** after weight quantization
- Vocabulary: 10,000 words
- Training: Federated Learning with FedAvg — the original keyboard FL paper
- Deployment: TensorFlow Lite

> This is our floor reference. Extremely small, proven in production, FL-native.

### Apple Predictive Text (iOS 17+)
- Architecture: custom GPT-2 style (6 decoder blocks, hidden_size=512)
- Parameters: 34M
- Vocabulary: 15,000 tokens
- Deployment: Core ML (Espresso runtime)
- Size: not publicly disclosed, estimated ~20–35 MB quantized

> This is our ceiling reference. Apple decided 34M params + quantization is the right
> balance for a keyboard shipped to billions of devices.

---

## Candidate models

### 1. Custom CIFG-LSTM (Gboard-style)
| | |
|---|---|
| Params | 1–5M (configurable) |
| Raw size | 4–20 MB |
| Quantized | **1–5 MB** |
| Latency | < 10 ms |
| Core ML | Native (LSTM layers supported) |
| Bilingual | Must train from scratch on es/en corpus |
| FL support | Yes — original Gboard FL paper uses this architecture |

Pros: proven in production, trivially small, fastest inference, FL-ready.  
Cons: lower quality ceiling than transformers, limited context window, must train from scratch.

---

### 2. Custom GPT-2 Tiny (Apple-style)
| | |
|---|---|
| Params | 20–40M (configurable) |
| Raw size | 80–160 MB |
| Quantized (INT8) | **20–40 MB** |
| Latency | 30–50 ms (estimated, mobile CPU) |
| Core ML | Yes — via [swift-transformers](https://github.com/huggingface/swift-transformers) or manual export |
| Bilingual | Depends on training data |
| FL support | Yes |

Pros: this is exactly what Apple ships; better context understanding than LSTM.  
Cons: right at the size/latency edge of our constraints — quantization is non-negotiable.

---

### 3. SmolLM-135M (HuggingFace, 2024)
| | |
|---|---|
| Params | 135M |
| Raw size (bfloat16) | 270 MB |
| Quantized (INT4) | **~110 MB** |
| Latency | TBD — not benchmarked on iOS CPU |
| Core ML | Not native; needs ONNX → Core ML conversion |
| Bilingual | Reasonable es/en out of the box |
| FL support | Yes |

Pros: modern architecture, good quality, available on HuggingFace, active community.  
Cons: 110 MB is over our 20 MB target even at INT4 — would require aggressive pruning
or distillation. Note: 4-bit quantization degrades quality noticeably at this model size.

---

### Ruled out

| Model | Params | Size | Reason |
|---|---|---|---|
| DistilBERT | 66M | 207 MB | Encoder-only (wrong task) + too big |
| TinyBERT-4 | 14.5M | 55 MB | Encoder-only (wrong task) |
| MobileBERT-TINY | 15.1M | ~60 MB | Encoder-only (wrong task) |
| Phi-3 Mini | 3.8B | 1.8 GB (INT4) | Way too big for a keyboard |
| Gemma 3 1B | 1B | 720 MB (INT4) | Too big |
| SmolLM 360M / 1.7B | 360M–1.7B | > 200 MB | Too big |

The BERT family (DistilBERT, TinyBERT, MobileBERT, ALBERT) is specifically excluded: these
are masked language models designed for classification, not next-word prediction. Adapting
them to causal LM is possible but defeats the purpose of their size advantage.

---

## Shortlist

Three candidates advance to Step 3 (baseline) and Step 4 (neural model):

| # | Model | Strategy | When to use |
|---|---|---|---|
| A | **CIFG-LSTM custom** | Train from scratch on bilingual corpus | Step 3 baseline — the performance floor |
| B | **GPT-2 Tiny custom** | Train from scratch or distill from larger model | Step 4 primary candidate |
| C | **SmolLM-135M** | Fine-tune + INT4 + aggressive pruning | Step 4 wildcard — only viable if we can get under 20 MB |

The realistic production path is **A → B**: start with the proven LSTM to validate the
data pipeline and FL setup, then upgrade to a small GPT-2 style model for better quality.

SmolLM-135M is worth a size/latency benchmark but is unlikely to fit our constraints
without custom distillation work.

---

## Open questions for Step 1

Before training anything, these need answers:

1. What vocabulary size is right for a bilingual personal keyboard? (Gboard uses 10k, Apple 15k)
2. How many tokens of context do we feed? (affects model architecture and latency)
3. Do we predict next word only, or next 2–3 words? (affects output head design)
4. What's the minimum acceptable suggestion quality? (defines our evaluation threshold)

---

## Empirical evaluation (2026-05-29)

Tested via Ollama on M2 Max. Task: bilingual es/en sentence completion in chat/instruct mode.

| Model | Spanish quality | Code-switching | Completion behavior | Verdict |
|---|---|---|---|---|
| smollm2:135m | Broken — pure hallucination | No | N/A | Discarded |
| smollm2:1.7b | Basic greetings only, collapses on any real task | Partial | Hallucinates wildly | Discarded |
| qwen2.5:0.5b | Good, consistent | Yes | Needs emphasis (`___`, ALL CAPS) to follow one-word instruction | Candidate |
| qwen2.5:1.5b | Good, natural | Yes | Inconsistent — sometimes perfect (lima, hoy, parque), sometimes ignores instruction | **Selected as teacher** |

**Key insight:** The instruction-following inconsistency in Qwen models is a *chat/instruct mode* artifact,
not a language model problem. In production we use the **base** (non-instruct) variant, which purely
predicts the next token — the "ignores one-word rule" behavior disappears by design.

When qwen2.5:1.5b follows the task, its completions are the most natural of all tested models:
single words, contextually appropriate, grammatically correct.

**Decision:** `qwen2.5:1.5b` (base variant) as teacher for distillation. SmolLM2 family fully discarded.

---

## Sources

- [Federated Learning for Mobile Keyboard Prediction (Google/Gboard, 2018)](https://arxiv.org/abs/1811.03604)
- [A look at Apple's Transformer-powered predictive text model](https://jackcook.com/2023/09/08/predictive-text.html)
- [Inside Apple's 2023 Transformer Models](https://stephenpanaro.com/blog/inside-apples-2023-transformers)
- [SmolLM-135M on HuggingFace](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
- [SmolLM-360M GGUF quantized](https://huggingface.co/QuantFactory/SmolLM-360M-GGUF)
- [MobileBERT summary — DAIR.AI](https://dair.ai/posts/Summary-MobileBERT/)
- [Comparative Analysis of DistilBERT, TinyBERT, MobileBERT](https://zenodo.org/records/15907007)
- [Demystifying Small Language Models for Edge Deployment (ACL 2025)](https://aclanthology.org/2025.acl-long.718.pdf)
- [Best Mobile LLM Models 2026: Phi-4 Mini vs Gemma 3 vs SmolLM](https://www.promptquorum.com/power-local-llm/mobile-llm-models-phi4-gemma-smollm)
