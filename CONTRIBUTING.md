# Contributing to Mythos

Thank you for your interest in contributing.

## Getting Started

```bash
git clone https://github.com/borisgraudt/mythos
cd mythos
pip install -e ".[dev]"
```

## Development Workflow

1. Fork the repository and create a branch from `main`
2. Make your changes
3. Ensure all tests pass: `make test`
4. Ensure the linter is happy: `make lint`
5. Open a pull request with a clear description

## Code Style

- **Formatter**: `ruff format` (line length 100)
- **Linter**: `ruff check`
- Keep functions short and focused
- Prefer clarity over cleverness — this is an educational codebase

## Testing

All new functionality must be covered by tests in `tests/`.

```bash
make test          # run all 34 tests
make test-model    # architecture tests only
make test-training # training loop tests only
```

## Areas to Contribute

- **Data pipeline**: better deduplication, more data sources
- **Evaluation**: LAMBADA, MMLU, perplexity benchmarks
- **Training stability**: learning rate schedules, mixed precision edge cases
- **Documentation**: architecture diagrams, training tutorials
- **Export**: GGUF conversion improvements, HuggingFace integration

## Reporting Issues

Open a GitHub issue with:
- A minimal reproduction case
- Python and PyTorch version
- Hardware (CPU / CUDA / MPS)
