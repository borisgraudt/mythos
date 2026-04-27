.PHONY: install test lint data-debug data-full train-debug train-150m train-500m infer export-hf export-gguf clean clean-data help

# ── Dev setup ──────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev,eval]"

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-model:
	pytest tests/test_model.py -v

test-training:
	pytest tests/test_training.py -v

# ── Linting ────────────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/

fmt:
	ruff format src/ scripts/ tests/

# ── Data pipeline ──────────────────────────────────────────────────────────
data-debug:
	python scripts/prepare_data.py --debug

data-full:
	python scripts/prepare_data.py

data-stage:
	python scripts/prepare_data.py --stages $(STAGE)

# ── Training ───────────────────────────────────────────────────────────────
train-debug:
	python scripts/train.py --mode debug

train-150m:
	python scripts/train.py \
		--mode data \
		--data data/debug

train-500m:
	python scripts/train.py \
		--mode full \
		--model configs/model/base_500m.yaml \
		--training configs/training/base.yaml \
		--data data

train-resume:
	python scripts/train.py \
		--mode full \
		--model configs/model/base_500m.yaml \
		--training configs/training/base.yaml \
		--resume $(CHECKPOINT)

# ── Inference ──────────────────────────────────────────────────────────────
infer-debug:
	python scripts/infer.py \
		--checkpoint checkpoints/debug/step_000100.pt \
		--prompt "Once upon a time"

infer:
	python scripts/infer.py \
		--checkpoint $(CHECKPOINT) \
		--prompt "$(PROMPT)"

# ── Export ─────────────────────────────────────────────────────────────────
REPO ?= bgraudt/mythos

export-hf:
	python scripts/export_hf.py \
		--checkpoint $(CHECKPOINT) \
		--tokenizer data/tokenizer/tokenizer.json \
		--repo $(REPO)

export-gguf:
	python scripts/export_gguf.py \
		--checkpoint $(CHECKPOINT) \
		--model_config configs/model/base_500m.yaml \
		--tokenizer data/tokenizer/tokenizer.json \
		--quantize q4_k_m

ollama-create:
	ollama create mythos -f Modelfile

ollama-run:
	ollama run mythos

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/ *.pyc .ruff_cache/

clean-data:
	@echo "WARNING: this deletes all downloaded/processed data."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ]
	rm -rf data/raw/ data/encoded/ data/splits/ data/tokenizer/ data/interim/ data/processed/ data/debug/

clean-checkpoints:
	@echo "WARNING: this deletes all checkpoints."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ]
	rm -rf checkpoints/

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo "Mythos — available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install                   Install all dependencies"
	@echo ""
	@echo "  Tests:"
	@echo "    make test                      Run all tests (34 tests)"
	@echo "    make test-model                Model component tests only"
	@echo "    make test-training             Training loop tests only"
	@echo ""
	@echo "  Data:"
	@echo "    make data-debug                Download small dataset (5K docs, Wikipedia)"
	@echo "    make data-full                 Full pipeline (FineWeb + Code + Books)"
	@echo ""
	@echo "  Training:"
	@echo "    make train-debug               1M model, 100 steps, dummy data (sanity check)"
	@echo "    make train-150m                150M model on real data"
	@echo "    make train-500m                Full 500M training"
	@echo "    make train-resume CHECKPOINT=checkpoints/... Resume training"
	@echo ""
	@echo "  Inference:"
	@echo "    make infer-debug               Test generation with debug model"
	@echo "    make infer CHECKPOINT=... PROMPT='...'  Generate text"
	@echo ""
	@echo "  Export:"
	@echo "    make export-hf CHECKPOINT=...  Upload to HuggingFace"
	@echo "    make export-gguf CHECKPOINT=... Convert to GGUF for Ollama"
	@echo "    make ollama-create             Register with Ollama"
	@echo "    make ollama-run                Run in Ollama"
