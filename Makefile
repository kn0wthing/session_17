.PHONY: clean lint format install dev run train

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 src
	mypy src
	black --check src
	isort --check-only src

format:
	black src
	isort src

install:
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -e ".[dev]"

run:
	python src/app.py

train:
	@echo "Usage: make train STYLE=<style_name> [STEPS=<steps>] [LR=<learning_rate>]"
	@echo "Example: make train STYLE=dhoni STEPS=3000 LR=1e-4"
	@if [ -n "$(STYLE)" ]; then \
		STEPS=$${STEPS:-3000}; \
		LR=$${LR:-1e-4}; \
		python src/train_style.py $(STYLE) --steps $$STEPS --lr $$LR; \
	fi 