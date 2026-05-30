"""
OhMyWord — evaluation script (Steps 3-8)

Evaluates any model against the hand-curated test suite.
Reports top-1 and top-3 accuracy, broken down by category.

Usage:
    # evaluate n-gram baseline
    uv run python model/eval/evaluate.py --model ngram

    # evaluate a specific checkpoint (neural models, Step 4+)
    uv run python model/eval/evaluate.py --model checkpoint --path model/train/checkpoints/student_v1.pt
"""

import argparse
import json
from pathlib import Path

# ── Load test suite ────────────────────────────────────────────────────────────

SUITE_PATH = Path("model/eval/test_suite.json")

def load_suite() -> list[dict]:
    with open(SUITE_PATH) as f:
        return json.load(f)["cases"]

# ── Model loaders ──────────────────────────────────────────────────────────────

def load_ngram():
    from model.train.ngram_baseline import NgramModel
    path = Path("model/train/checkpoints/ngram_baseline.json")
    if not path.exists():
        raise FileNotFoundError(f"Train the n-gram first: uv run python model/train/ngram_baseline.py")
    return NgramModel.load(path)

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, cases: list[dict]) -> dict:
    results = []

    for case in cases:
        predictions = model.predict(case["context"], top_k=3)
        acceptable  = [w.lower() for w in case["acceptable"]]
        predictions = [p.lower() for p in predictions]

        top1_hit = len(predictions) > 0 and predictions[0] in acceptable
        top3_hit = any(p in acceptable for p in predictions)

        results.append({
            "id":          case["id"],
            "type":        case["type"],
            "context":     case["context"],
            "predictions": predictions,
            "acceptable":  acceptable,
            "top1":        top1_hit,
            "top3":        top3_hit,
        })

    return results

def print_report(results: list[dict], model_name: str) -> None:
    types = sorted(set(r["type"] for r in results))

    print(f"\n{'─' * 60}")
    print(f"  Evaluation: {model_name}")
    print(f"{'─' * 60}")

    # per-category breakdown
    print(f"\n{'Category':<16} {'Top-1':>6} {'Top-3':>6} {'N':>4}")
    print(f"{'─'*16} {'─'*6} {'─'*6} {'─'*4}")
    for t in types:
        group = [r for r in results if r["type"] == t]
        top1 = sum(r["top1"] for r in group) / len(group)
        top3 = sum(r["top3"] for r in group) / len(group)
        print(f"{t:<16} {top1:>5.0%} {top3:>5.0%} {len(group):>4}")

    # overall
    top1_all = sum(r["top1"] for r in results) / len(results)
    top3_all = sum(r["top3"] for r in results) / len(results)
    print(f"{'─'*16} {'─'*6} {'─'*6} {'─'*4}")
    print(f"{'OVERALL':<16} {top1_all:>5.0%} {top3_all:>5.0%} {len(results):>4}")

    # per-case detail
    print(f"\n{'─' * 60}")
    print("  Per-case detail")
    print(f"{'─' * 60}")
    for r in results:
        status = "✓" if r["top3"] else "✗"
        preds  = " | ".join(r["predictions"][:3]) if r["predictions"] else "(none)"
        print(f"  {status} [{r['id']}] '{r['context'][-40:]}' → {preds}")
        if not r["top3"]:
            print(f"      expected one of: {r['acceptable']}")

    print(f"\n  Top-1: {top1_all:.0%}  |  Top-3: {top3_all:.0%}  ({len(results)} cases)\n")

# ── Main ───────────────────────────────────────────────────────────────────────

def main(model_name: str, checkpoint_path: str | None) -> None:
    cases = load_suite()

    if model_name == "ngram":
        model = load_ngram()
    else:
        raise NotImplementedError(f"Model '{model_name}' not yet implemented. Coming in Step 4.")

    results = evaluate(model, cases)
    print_report(results, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["ngram"],
                        help="Model to evaluate")
    parser.add_argument("--path", default=None,
                        help="Path to checkpoint (for neural models)")
    args = parser.parse_args()
    main(args.model, args.path)
