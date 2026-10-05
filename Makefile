.PHONY: all help test tests integration_test integration_tests lint format check_imports lint_imports

all: help

# Unit tests run offline: tests/unit_tests/conftest.py refuses network sockets.
test tests:
	uv run pytest tests/unit_tests

# Live and billed to STRING_API_KEY; every test is skipped when it is unset.
integration_test integration_tests:
	uv run pytest tests/integration_tests

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy langchain_string tests

format:
	uv run ruff format .
	uv run ruff check --fix .

check_imports:
	uv run python scripts/check_imports.py $(shell find langchain_string -name '*.py')

lint_imports:
	./scripts/lint_imports.sh

help:
	@echo 'test                - run unit tests'
	@echo 'integration_test    - run the live integration tests'
	@echo 'lint                - run ruff and mypy'
	@echo 'format              - format and autofix'
	@echo 'check_imports       - import every package module'
	@echo 'lint_imports        - check the package imports nothing beyond langchain-core'
