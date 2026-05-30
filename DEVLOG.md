# Dev Log

---

## 2026-05-28 — Ideation & architecture

**Decisions:**
- Task: next-word prediction, top-3 suggestions, 128-token context window (3-5 conversational sentences)
- Two-layer architecture: global base model (GPT-2 Tiny, ~20MB) + personal LoRA adapter (~1-2MB, never leaves device)
- Distillation teacher: `qwen2.5:1.5b` — best bilingual (es/en) quality among tested candidates (smollm2 discarded)
- FL: only base model weight deltas reach the server. LoRA adapter stays 100% local → privacy and legal argument

**Stack:**
- Python + PyTorch, uv as package manager
- Tokenizer: Qwen2.5 (inherited by student for direct distillation — same vocabulary, comparable token distributions)
- Data: Wikipedia es/en + OPUS-100 + Miami Corpus (real Spanglish, oversampled 5x)

---

## 2026-05-29 — Setup & Steps 2-3

**Setup:**
- `uv sync` to initialize environment
- HuggingFace auth: `hf auth login` (`huggingface-cli login` is deprecated)

**Issues:**
- `trust_remote_code=True` no longer supported in newer `datasets` versions → removed
- `[Errno 9] Bad file descriptor` on OPUS-100 download → rate limiting due to missing HF auth. Fixed with `hf auth login`
- "Done." printed before OPUS error → HuggingFace streaming downloads in a background thread, error surfaces after main thread finishes. Not a bug.

**Progress:**
- Data pipeline (`model/data/build_dataset.py`) — downloads, cleans, tokenizes, splits corpus
- N-gram baseline (`model/train/ngram_baseline.py`) — 4-gram with stupid backoff
- Evaluator (`model/eval/evaluate.py`) — top-1/top-3 accuracy, reusable across all models
- Hand-curated test suite (`model/eval/test_suite.json`) — 24 cases: Spanish, English, code-switching, slang, conversational
