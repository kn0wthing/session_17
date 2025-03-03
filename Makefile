.PHONY: clean lint format test install dev

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 src tests
	mypy src tests
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

test:
	pytest

install:
	pip install -r requirements.txt
	pip install -e .

dev: install
	pip install -e ".[dev]" 