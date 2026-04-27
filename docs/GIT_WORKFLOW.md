# Git Workflow — Mythos

План: залить всё за неделю по веткам, обучить модель, закрыть PR в main.

---

## Структура веток

```
main          ← только стабильные, проверенные версии (v0.1, v1.0)
└── dev       ← интеграционная ветка, всегда рабочая
    ├── feat/upload-all       ← всё что уже готово (код + доки)
    └── experiment/150m       ← обучение + реальные результаты
```

---

## День 1 — Залить всё что уже готово

### Создать ветку

```bash
git checkout main
git checkout -b feat/upload-all
```

### Коммит 1: архитектура модели

```bash
git add src/core/ src/inference/ src/utils/ src/__init__.py configs/model/
git commit -m "feat: transformer architecture from scratch

- Decoder-only transformer (LLaMA-style)
- Grouped Query Attention: 16Q / 8KV heads, 2x smaller KV cache
- SwiGLU FFN: +10% quality over GeLU at same FLOPs
- RoPE positional embeddings (no learned params)
- RMSNorm pre-norm: 10-15% faster than LayerNorm
- Weight tying: embedding and output matrix shared
- KV cache for O(1) autoregressive generation
- Streaming generation with top-p / top-k sampling"
```

### Коммит 2: тренировочный пайплайн

```bash
git add src/training/ configs/training/
git commit -m "feat: training pipeline

- AdamW optimizer with weight decay
- Cosine decay scheduler with linear warmup
- bfloat16 mixed precision (2x memory reduction)
- Gradient accumulation and gradient clipping
- Checkpoint save/load with embedded model config
- Auto-resumes from any checkpoint
- Debug mode (1M model, dummy data, 100 steps)"
```

### Коммит 3: скрипты

```bash
git add scripts/train.py scripts/prepare_data.py scripts/infer.py
git add scripts/eval.py scripts/export_hf.py scripts/export_gguf.py
git add Modelfile
git commit -m "feat: training, inference, and export scripts

- train.py: debug mode + full data mode
- prepare_data.py: download → tokenize → encode → shard
- infer.py: streaming generation, auto-loads config from checkpoint
- eval.py: WikiText-2 / LAMBADA evaluation
- export_hf.py: safetensors + HuggingFace Hub upload
- export_gguf.py: GGUF for Ollama / llama.cpp (Q4_K_M)"
```

### Коммит 4: тесты

```bash
git add tests/
git commit -m "test: unit tests for model and training loop

Model tests:
- RMSNorm, RoPE, GQA attention, SwiGLU FFN shapes and dtypes
- Autoregressive causality (token at pos i must not affect logits at pos < i)
- Weight tying verification
- All model configs load without error

Training tests:
- Optimizer parameter groups and weight decay
- Scheduler LR warmup and cosine decay
- Trainer forward/backward pass
- Loss decreases over 20 steps (fixed seed)
- Checkpoint save/load round-trip"
```

### Коммит 5: документация и конфиги проекта

```bash
git add docs/ README.md CONTRIBUTING.md CITATION.cff LICENSE
git add Makefile pyproject.toml requirements.txt .github/ .gitignore
git commit -m "docs: README, architecture docs, CI, project config

- README: architecture table, quickstart, model variants
- ARCHITECTURE.md: detailed component breakdown
- RESEARCH.md: design rationale with citations
- results.md: benchmark targets (PPL, LAMBADA, MMLU)
- CONTRIBUTING.md: development workflow
- CITATION.cff: academic citation info
- GitHub Actions CI: tests on Python 3.11 and 3.12
- Makefile: complete command reference"
```

### Push + PR → dev

```bash
git push -u origin feat/upload-all

gh pr create \
  --base dev \
  --head feat/upload-all \
  --title "feat: complete codebase upload" \
  --body "All code, docs, tests, and CI. No training results yet — coming in experiment/150m."
```

**После merge: dev содержит весь код. main пока чистый.**

---

## День 2-5 — Обучить модель (ветка experiment/150m)

```bash
git checkout dev
git checkout -b experiment/150m
```

### Что нужно сделать на этой ветке

1. Запустить debug прогон (убедиться что всё работает):
   ```bash
   python scripts/train.py --mode debug
   ```

2. Подготовить данные:
   ```bash
   python scripts/prepare_data.py --debug
   ```

3. Обучить 150M:
   ```bash
   python scripts/train.py --mode data --data data/debug --config configs/model/150m.yaml
   ```

4. Оценить:
   ```bash
   python scripts/eval.py --checkpoint checkpoints/150m_*/final.pt
   ```

5. Экспортировать на HuggingFace Hub:
   ```bash
   python scripts/export_hf.py --checkpoint checkpoints/150m_*/final.pt \
     --repo borisgraudt/mythos-150m
   ```

### Какие файлы создать/обновить на ветке

| Файл | Что добавить |
|------|-------------|
| `docs/results.md` | Реальные числа: PPL, loss curve, hardware stats |
| `notebooks/demo.ipynb` | Forward pass + generation примеры + attention viz |
| `docs/results.md` | Ссылку на HuggingFace Hub с весами |

### Коммит с результатами

```bash
git add docs/results.md notebooks/
git commit -m "experiment: Mythos-150M training results

- Dataset: FineWeb-Edu debug (5K docs), vocab=3252
- Steps: 5000, bfloat16, Apple M2
- WikiText-2 PPL: XX.X
- Throughput: ~X,XXX tokens/sec
- Weights: https://huggingface.co/borisgraudt/mythos-150m"
```

```bash
git push -u origin experiment/150m

gh pr create \
  --base dev \
  --head experiment/150m \
  --title "experiment: Mythos-150M training complete" \
  --body "Real training results, demo notebook, HuggingFace weights."
```

---

## День 6-7 — Merge в main, тег v0.1.0

После того как experiment/150m влит в dev и всё проверено:

```bash
# PR dev → main
gh pr create \
  --base main \
  --head dev \
  --title "release: Mythos v0.1.0 — 150M model trained and released" \
  --body "Complete transformer from scratch: GQA + SwiGLU + RoPE.
150M model trained, evaluated, exported to HuggingFace Hub.
All tests passing."

# После merge:
git checkout main && git pull
git tag v0.1.0 -m "Mythos-150M: first trained release"
git push origin v0.1.0
```

---

## Чего не хватает для MIT portfolio (файлы)

Весь код уже есть. Не хватает только:

| Файл | Где создать | Когда |
|------|-------------|-------|
| `notebooks/demo.ipynb` | корень проекта | на ветке experiment/150m |
| `docs/results.md` с реальными числами | уже существует, заполнить | после обучения |
| HuggingFace Hub с весами | внешний сервис | после обучения |

Это всё делается на одной ветке `experiment/150m`.

---

## Что НЕ попадает в git

```
checkpoints/    → HuggingFace Hub
data/raw/       → скачивается через prepare_data.py
data/debug/     → генерируется локально
mythos/         → виртуальное окружение
.DS_Store       → macOS мусор
export/         → артефакты экспорта
```

---

## Правила коммитов

| Prefix | Когда |
|--------|-------|
| `feat:` | новый компонент / функционал |
| `fix:` | исправление бага |
| `docs:` | только документация |
| `test:` | тесты |
| `experiment:` | результаты обучения |
| `chore:` | зависимости, конфиги CI |
