.PHONY: test-device test-model clean help

test-device:
	python scripts/test_device.py

test-model:
	python scripts/test_model.py

clean:
	rm -rf __pycache__ runs/ checkpoints/ .pytest_cache/ *.pyc

help:
	@echo "Mythos — available commands:"
	@echo "  make test-device     Test MPS availability"
	@echo "  make test-model      Test model creation"
	@echo "  make clean           Clean temporary files"