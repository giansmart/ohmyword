.PHONY: install dataset baseline evaluate

install:
	uv sync

dataset:
	uv run python model/data/build_dataset.py

baseline:
	uv run python -m model.train.ngram_baseline

baseline-quick:
	uv run python -m model.train.ngram_baseline --sample 5000

evaluate-baseline:
	uv run python model/eval/evaluate.py --model ngram
