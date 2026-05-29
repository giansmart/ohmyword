# Model Roadmap

The core of OhMyWord is the model. Everything else (the keyboard UI, federated learning, the app) depends on getting this right first.

Three hard constraints drive every decision here:
- **Size** — must feel like a keyboard, not a game. Target: <20MB
- **Latency** — suggestion must appear before the user notices. Target: <50ms on-device
- **Quality** — must handle bilingual text (es/en code-switching) and learn personal vocabulary

---

## Step 0 — Model Research

Before committing to any architecture, evaluate candidates against the three hard constraints.

- Survey small language models suitable for next-word/next-phrase prediction: DistilBERT, TinyBERT, MobileBERT, ALBERT, small LSTMs, n-gram hybrids, and any newer options (2024–2025)
- Filter by: model size (raw + quantized), Core ML export support, bilingual capability (es/en), inference speed benchmarks on mobile CPUs
- Check App Store precedents: any existing keyboard or on-device NLP app that ships a model — what size do they use?
- Produce a comparison table: model × (size, latency estimate, bilingual quality, Core ML support, ease of fine-tuning)
- Pick 2–3 finalists to carry into Step 4

*Output: a short decision doc + ranked shortlist of model candidates*

---

## Step 1 — Define the task precisely

Before any code: nail down exactly what we're predicting.

- Next word? Next 2–3 words? Full-phrase completion?
- Input: last N characters? last N tokens? full sentence context?
- Pick evaluation metrics: **perplexity** (model quality) + **latency** (ms per inference) + **model size** (MB)
- Write a small test suite of bilingual sentences to evaluate by hand throughout the project

*Output: a short spec doc + hand-curated test cases*

---

## Step 2 — Curate a baseline dataset

A small, representative bilingual corpus to train and evaluate on.

- Sources: public es/en data (Wikipedia dumps, OpenSubtitles, Common Crawl subsets)
- Mix ratio: ~50/50 es-en, plus code-switching examples
- Size: start small — 10–50MB of text is enough to validate architecture choices
- Clean and tokenize: handle accents, punctuation, lowercase normalization

*Output: a reproducible data pipeline (Python script) that produces train/val/test splits*

---

## Step 3 — Baseline: n-gram model

Start with the simplest thing that could work before touching neural networks.

- Train a trigram or 4-gram model (KenLM or pure Python)
- Evaluate on the bilingual test set
- Benchmark size and inference time
- This gives a performance floor — every neural model must beat this

*Output: a working baseline with numbers to beat*

---

## Step 4 — Small neural model (student)

Train the student model via knowledge distillation from the teacher (qwen2.5:1.5b).

- Architecture: custom GPT-2 Tiny (6 decoder blocks, hidden_size=512, ~30M params)
- **LoRA support is mandatory from the start** — the model must accept adapter weights
  so personal vocabulary can be layered on top without touching the base model
- Distillation loss = cross-entropy on hard labels + KL divergence from teacher's soft labels
- Train on the bilingual corpus from Step 2
- Evaluate: does it beat the n-gram baseline on the bilingual test suite?

*Output: a trained student checkpoint + eval numbers vs. baseline*

---

## Step 5 — Compression

Squeeze the model to fit the size and latency targets.

- Quantize to INT8 (Post-Training Quantization or QAT)
- Prune low-magnitude weights if needed
- Evaluate quality degradation vs. size/speed gain
- Goal: <20MB, <50ms per inference (benchmark on CPU, not GPU — mobile reality)

*Output: compressed model that hits both targets without falling below n-gram quality*

---

## Step 6 — On-device simulation

Validate the model actually works in a mobile-like environment.

- Export to Core ML (`.mlpackage`) and/or ONNX
- Run inference on an iPhone (or M-chip Mac as proxy) — measure real latency
- Profile memory usage during inference
- Identify bottlenecks before touching the keyboard UI

*Output: Core ML model that runs in <50ms on real hardware*

---

## Step 7 — Personalization via LoRA adapters

Each user gets their own personal layer on top of the shared base model.

**Two-layer architecture:**
```
┌──────────────────────────────────┐
│     Global base model (~20MB)    │  ← shared across all users, updated via FL
└─────────────────┬────────────────┘
                  │
┌─────────────────▼────────────────┐
│   Personal LoRA adapter (~1-2MB) │  ← lives only on the user's device, never leaves
└──────────────────────────────────┘
```

- LoRA trains only a small set of additional weights that steer the base model
  toward each user's vocabulary and style — without modifying the base model itself
- Simulate 3 user profiles: heavy Spanglish (MX slang), tech worker (EN+jargon), formal ES
- Train a LoRA adapter per profile on synthetic personal data
- Verify: personalized model beats generic model on each user's own test sentences
- Verify: user A's adapter does not bleed into user B's predictions

*Output: LoRA training pipeline + per-user accuracy improvement evidence*

---

## Step 8 — Federated Learning simulation

Prove that the global model improves across users while personal vocabulary stays private.

**What travels to the server vs. what stays on device:**
```
Device                          Server
──────                          ──────
base model weights (delta)  →   FedAvg aggregation
                                    ↓
                            updated global model  →  all devices
LoRA adapter weights        ✗   never leaves the device
raw typing data             ✗   never leaves the device
```

- Use Flower (`flwr`) to simulate N clients (the user profiles from Step 7)
- Each client fine-tunes only the **base model** locally; sends weight deltas to server
- LoRA adapters are excluded from aggregation — they are purely personal
- Aggregate base model deltas with FedAvg; distribute updated global base model
- Verify: global model improves on general bilingual text across all simulated users
- Verify: no client's personal vocabulary (slang, jargon) leaks into the global model

*Output: Flower simulation script + metrics showing FL convergence without personal data leakage*

---

## Step 9 — Wire into iOS Keyboard Extension

The model meets the product.

- Build a minimal `UIInputViewController` in Swift
- Load the Core ML model and run inference on every keystroke
- Show top-3 suggestions in the suggestion bar
- No styling — just proof that the model works in a real keyboard context

*Output: a functional (ugly) keyboard that suggests words using OhMyWord's model*
