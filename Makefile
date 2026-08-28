.PHONY: install doctor format lint type test check

install:
	uv sync --all-extras --dev

doctor:
	uv run short-engine doctor

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests

type:
	uv run ty check

test:
	uv run pytest --cov=short_engine

check: format lint type test
