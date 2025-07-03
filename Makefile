# Document Chat Makefile
# Provides convenient commands for common operations

.PHONY: help install install-dev install-enhanced ui cli ingest chat history provider observability clean test lint format typecheck setup-ollama

# Default target - show help
help:
	@echo "Document Chat - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install base dependencies"
	@echo "  make install-dev      - Install development dependencies"
	@echo "  make install-enhanced - Install enhanced processing dependencies"
	@echo "  make setup-ollama     - Pull default Ollama model (llama3.2:latest)"
	@echo ""
	@echo "Running the Application:"
	@echo "  make ui              - Launch Streamlit web interface"
	@echo "  make cli             - Run CLI in interactive mode"
	@echo "  make chat            - Start a chat session (CLI)"
	@echo ""
	@echo "Document Management:"
	@echo "  make ingest path=/path/to/docs - Ingest documents from a directory"
	@echo "  make history         - View conversation history"
	@echo "  make provider        - Show current provider configuration"
	@echo "  make observability   - Check observability status"
	@echo ""
	@echo "Development:"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run code linting (flake8)"
	@echo "  make format          - Format code with black"
	@echo "  make typecheck       - Run type checking with mypy"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean           - Clean up temporary files and caches"
	@echo "  make clean-index     - Remove vector store index"
	@echo "  make clean-state     - Remove conversation state"
	@echo "  make clean-all       - Remove all generated data"

# Install base dependencies
install:
	poetry install

# Install development dependencies
install-dev:
	poetry install --with dev

# Install enhanced processing dependencies
install-enhanced:
	poetry install --with enhanced

# Launch Streamlit UI
ui:
	poetry run ui

# Alternative UI command
streamlit:
	poetry run streamlit run src/ui.py

# Run CLI in interactive mode
cli:
	poetry run python -m src.cli

# Start a chat session
chat:
	poetry run python -m src.cli chat

# Start chat with Ollama
chat-ollama:
	poetry run python -m src.cli chat --llm-provider ollama --llm-model llama3.2:latest

# Start chat with OpenAI
chat-openai:
	poetry run python -m src.cli chat --llm-provider openai --llm-model gpt-4o

# Ingest documents from a directory
ingest:
ifndef path
	@echo "Error: Please specify a path to ingest documents from"
	@echo "Usage: make ingest path=/path/to/documents"
	@exit 1
endif
	poetry run python -m src.cli ingest "$(path)"

# View conversation history
history:
	poetry run python -m src.cli history

# Show provider configuration
provider:
	poetry run python -m src.cli provider

# Check observability status
observability:
	poetry run python -m src.cli observability

# Run tests
test:
	poetry run pytest

# Run tests with coverage
test-coverage:
	poetry run pytest --cov=src --cov-report=html --cov-report=term

# Run linting
lint:
	poetry run flake8 src/

# Format code with black
format:
	poetry run black src/

# Check code formatting without modifying
format-check:
	poetry run black --check src/

# Run type checking
typecheck:
	poetry run mypy src/

# Run all code quality checks
quality: format-check lint typecheck

# Setup Ollama with default model
setup-ollama:
	@echo "Pulling default Ollama model (llama3.2:latest)..."
	@command -v ollama >/dev/null 2>&1 || { echo "Error: Ollama is not installed. Please install from https://ollama.com"; exit 1; }
	ollama pull llama3.2:latest
	@echo "Ollama setup complete!"

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -rf .mypy_cache 2>/dev/null || true
	rm -rf htmlcov 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true

# Clean vector store index
clean-index:
	rm -rf data/vector_store/* 2>/dev/null || true
	@echo "Vector store index cleaned"

# Clean conversation state
clean-state:
	rm -rf data/graph_state/* 2>/dev/null || true
	@echo "Conversation state cleaned"

# Clean all generated data
clean-all: clean clean-index clean-state
	@echo "All generated data cleaned"

# Create necessary directories
setup-dirs:
	mkdir -p data/vector_store
	mkdir -p data/graph_state
	mkdir -p data/sample_docs
	@echo "Directories created"

# Quick start for new users
quickstart: install setup-dirs
	@echo ""
	@echo "🚀 Setup complete! Next steps:"
	@echo ""
	@echo "1. Copy .env.example to .env and add your API keys:"
	@echo "   cp .env.example .env"
	@echo ""
	@echo "2. For OpenAI (cloud):"
	@echo "   - Add your OPENAI_API_KEY to .env"
	@echo ""
	@echo "3. For Ollama (local):"
	@echo "   - Install Ollama from https://ollama.com"
	@echo "   - Run: make setup-ollama"
	@echo ""
	@echo "4. Launch the application:"
	@echo "   - Web UI: make ui"
	@echo "   - CLI: make chat"
	@echo ""

# Development server with auto-reload
dev:
	poetry run streamlit run src/ui.py --server.runOnSave true

# Check environment setup
check-env:
	@echo "Checking environment setup..."
	@echo ""
	@echo "Python version:"
	@poetry run python --version
	@echo ""
	@echo "Poetry version:"
	@poetry --version
	@echo ""
	@echo "Installed packages:"
	@poetry show --tree
	@echo ""
	@echo "Environment variables:"
	@poetry run python -c "from src.config import settings; print(f'LLM Provider: {settings.llm_provider}'); print(f'LLM Model: {settings.llm_model}'); print(f'Embedding Provider: {settings.embedding_provider}'); print(f'Embedding Model: {settings.embedding_model}')"

# Generate requirements.txt from poetry
requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

# Generate requirements-dev.txt from poetry
requirements-dev:
	poetry export -f requirements.txt --output requirements-dev.txt --without-hashes --with dev

# Docker-related targets (if you add Docker support later)
docker-build:
	@echo "Docker support not yet implemented"

docker-run:
	@echo "Docker support not yet implemented"