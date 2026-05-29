"""
OhMyWord — data pipeline (Step 2)

Downloads, cleans, tokenizes, and splits the training corpus.
Output: model/data/processed/{train,val,test}.jsonl

Usage:
    python model/data/build_dataset.py
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────

TOKENIZER_NAME  = "Qwen/Qwen2.5-1.5B"
CONTEXT_WINDOW  = 128
STRIDE          = 64
SAMPLE_SIZES    = {
    "wiki_es":      50_000,
    "wiki_en":      50_000,
    "opus":         30_000,
}
SPLIT_RATIOS    = {"train": 0.90, "val": 0.05, "test": 0.05}
OUTPUT_DIR      = Path("model/data/processed")
MIAMI_DIR       = Path("model/data/raw/miami")
SEED            = 42

# ── Download ───────────────────────────────────────────────────────────────────

def load_wikipedia(lang: str, n: int) -> list[str]:
    print(f"  Downloading Wikipedia ({lang})...")
    ds = load_dataset(
        "wikimedia/wikipedia", f"20231101.{lang}",
        split="train", streaming=True, trust_remote_code=True,
    )
    sentences = []
    for article in ds:
        for line in article["text"].split("\n"):
            line = line.strip()
            if 15 < len(line) < 400:
                sentences.append(line)
                if len(sentences) >= n:
                    return sentences
    return sentences


def load_opus(n: int) -> list[str]:
    print("  Downloading OPUS-100 (es-en)...")
    ds = load_dataset(
        "Helsinki-NLP/opus-100", "en-es",
        split="train", streaming=True,
    )
    sentences = []
    for item in ds:
        sentences.append(item["translation"]["es"])
        sentences.append(item["translation"]["en"])
        if len(sentences) >= n:
            return sentences[:n]
    return sentences


def load_miami_corpus() -> list[str]:
    """Loads .cha transcription files from the Miami corpus if present."""
    if not MIAMI_DIR.exists():
        return []
    print(f"  Loading Miami Corpus from {MIAMI_DIR}...")
    sentences = []
    for cha_file in MIAMI_DIR.glob("*.cha"):
        for line in cha_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            # .cha format: utterance lines start with *SPE: or similar speaker codes
            if line.startswith("*"):
                text = re.sub(r"^\*\w+:\s*", "", line)
                text = re.sub(r"\[.*?\]|\d+_\d+|[<>]", "", text).strip()
                if 5 < len(text) < 300:
                    sentences.append(text)
    print(f"    → {len(sentences)} utterances loaded")
    return sentences

# ── Clean ──────────────────────────────────────────────────────────────────────

def clean(texts: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for t in texts:
        t = t.strip()
        # length filter
        if len(t) < 10 or len(t) > 500:
            continue
        # remove lines that are mostly non-alpha (URLs, tables, code)
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in t) / len(t)
        if alpha_ratio < 0.70:
            continue
        # dedup
        if t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    return cleaned

# ── Tokenize & window ──────────────────────────────────────────────────────────

def make_windows(
    texts: list[str],
    tokenizer: AutoTokenizer,
    window: int,
    stride: int,
) -> list[dict]:
    examples = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for start in range(0, len(ids) - window, stride):
            chunk = ids[start : start + window]
            if len(chunk) == window:
                examples.append({
                    "input_ids": chunk[:-1],  # context: tokens 0..126
                    "labels":    chunk[1:],   # targets: tokens 1..127 (shifted)
                })
    return examples

# ── Save ───────────────────────────────────────────────────────────────────────

def save_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)

    print("=== OhMyWord — build_dataset.py ===\n")

    print("[1/4] Downloading sources...")
    texts: list[str] = []
    texts += load_wikipedia("es", SAMPLE_SIZES["wiki_es"])
    texts += load_wikipedia("en", SAMPLE_SIZES["wiki_en"])
    texts += load_opus(SAMPLE_SIZES["opus"])

    miami = load_miami_corpus()
    # Miami corpus is small but high-value (real Spanglish) — repeat 5x in the mix
    texts += miami * 5

    print(f"\n[2/4] Cleaning... ({len(texts)} raw sentences)")
    texts = clean(texts)
    random.shuffle(texts)
    print(f"  → {len(texts)} clean sentences")

    print(f"\n[3/4] Tokenizing with {TOKENIZER_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    examples = make_windows(texts, tokenizer, CONTEXT_WINDOW, STRIDE)
    print(f"  → {len(examples)} training windows ({CONTEXT_WINDOW} tokens each)")

    print("\n[4/4] Splitting and saving...")
    n       = len(examples)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])
    splits  = {
        "train": examples[:n_train],
        "val":   examples[n_train : n_train + n_val],
        "test":  examples[n_train + n_val :],
    }
    for name, data in splits.items():
        path = OUTPUT_DIR / f"{name}.jsonl"
        save_jsonl(data, path)
        print(f"  {name:5s}: {len(data):>7,} examples → {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
