.PHONY: install dev-install test lint format typecheck clean run-server run-example help

help:
	@echo "ML-AI Framework - Make Commands"
	@echo "================================"
	@echo "install        Install production dependencies"
	@echo "dev-install    Install development dependencies"
	@echo "test           Run tests with coverage"
	@echo "lint           Run linting checks"
	@echo "format         Format code with black"
	@echo "typecheck      Run type checking with mypy"
	@echo "clean          Clean build artifacts"
	@echo "run-server     Start AG-UI server"
	@echo "run-example    Run CrewAI example"
	@echo "setup          Complete setup (install + env)"

install:
	pip install -r requirements.txt

dev-install: install
	pip install pytest pytest-asyncio pytest-cov black ruff mypy pre-commit

test:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html -v

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/ examples/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ *.egg-info/

run-server:
	python -m src.ag_ui_server

run-example:
	python examples/simple_workflow.py

run-langgraph:
	python examples/langgraph_workflow.py

setup: dev-install
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file - please add your API keys"; fi
	@echo "Setup complete! Edit .env with your API keys."
