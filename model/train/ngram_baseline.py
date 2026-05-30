"""
OhMyWord — N-gram baseline (Step 3)

Trains a 4-gram language model with stupid backoff on the bilingual corpus.
This is the performance floor: every neural model must beat these numbers.

A 4-gram model predicts the next word given the last 3 words as context.
If the exact 3-word context was never seen in training, it backs off to
2-word context, then 1-word, then the most common words overall.

Usage:
    uv run python model/train/ngram_baseline.py
    uv run python model/train/ngram_baseline.py --sample 20000
"""

import argparse
import json
import string
from collections import Counter, defaultdict
from pathlib import Path


# ── Config ─────────────────────────────────────────────────────────────────────

N           = 4          # 4-gram: uses last 3 words as context
SAMPLE_SIZE = 50_000     # sentences per source (reduce with --sample for quick tests)
OUTPUT_DIR  = Path("model/train/checkpoints")
MODEL_PATH  = OUTPUT_DIR / "ngram_baseline.json"

# ── Text normalization ─────────────────────────────────────────────────────────

def normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()

# ── Data loading ───────────────────────────────────────────────────────────────

RAW_TEXT_PATH = Path("model/data/processed/raw_text.txt")

def load_texts(sample_size: int) -> list[str]:
    if RAW_TEXT_PATH.exists():
        print(f"  Loading from {RAW_TEXT_PATH}...")
        texts = RAW_TEXT_PATH.read_text().splitlines()
        if sample_size < len(texts):
            texts = texts[:sample_size]
        return texts

    raise FileNotFoundError(
        f"Raw text not found at {RAW_TEXT_PATH}. "
        "Run `make dataset` first to build the corpus."
    )

# ── N-gram model ───────────────────────────────────────────────────────────────

class NgramModel:
    """
    4-gram language model with stupid backoff.

    Stupid backoff: if the exact context is not found, shorten the context
    by one word and try again. Simple and effective for our use case.
    """

    def __init__(self, n: int = N):
        self.n = n
        # counts[(w1, w2, w3)] = Counter({next_word: frequency})
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.unigrams: Counter = Counter()

    def train(self, texts: list[str]) -> None:
        for text in texts:
            words = normalize(text)
            if len(words) < self.n:
                continue
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i : i + self.n - 1])
                next_word = words[i + self.n - 1]
                self.counts[context][next_word] += 1
                self.unigrams[next_word] += 1

    def predict(self, context: str, top_k: int = 3) -> list[str]:
        """Return top_k predicted next words given a text context."""
        words = normalize(context)

        # try progressively shorter contexts (stupid backoff)
        for length in range(self.n - 1, 0, -1):
            key = tuple(words[-length:])
            if key in self.counts:
                return [w for w, _ in self.counts[key].most_common(top_k)]

        # last resort: globally most common words
        return [w for w, _ in self.unigrams.most_common(top_k)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "n": self.n,
            "counts":   {str(k): dict(v) for k, v in self.counts.items()},
            "unigrams": dict(self.unigrams),
        }
        json_path = path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"  Model saved → {json_path}")

    @staticmethod
    def load(path: Path) -> "NgramModel":
        import ast
        json_path = path.with_suffix(".json")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        model = NgramModel(n=data["n"])
        model.counts = defaultdict(Counter, {
            ast.literal_eval(k): Counter(v)
            for k, v in data["counts"].items()
        })
        model.unigrams = Counter(data["unigrams"])
        return model

# ── Main ───────────────────────────────────────────────────────────────────────

def main(sample_size: int) -> None:
    print(f"=== OhMyWord — N-gram baseline (N={N}) ===\n")

    print("[1/2] Loading data...")
    texts = load_texts(sample_size)
    print(f"  → {len(texts):,} sentences\n")

    print("[2/2] Training model...")
    model = NgramModel(n=N)
    model.train(texts)

    unique_contexts = len(model.counts)
    total_ngrams = sum(sum(c.values()) for c in model.counts.values())
    print(f"  → {unique_contexts:,} unique contexts")
    print(f"  → {total_ngrams:,} total n-grams")
    print(f"  → {len(model.unigrams):,} unique words\n")

    model.save(MODEL_PATH)

    # quick sanity check
    print("Quick predictions:")
    samples = [
        "me gusta el ceviche con",
        "nos vemos en el",
        "I need to send that",
        "oye I need to send that email para",
    ]
    for ctx in samples:
        preds = model.predict(ctx)
        print(f"  '{ctx}' → {preds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE,
                        help="Sentences per source (default: 50000)")
    args = parser.parse_args()
    main(args.sample)
