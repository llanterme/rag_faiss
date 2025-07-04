# Document Chat Makefile
# Provides convenient commands for common operations

.PHONY: help install install-dev install-enhanced ui cli ingest chat history provider observability clean test lint format typecheck setup-ollama student-create student-list student-select student-current student-delete

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
	@echo "  make ui              - Launch Educational Insight Platform (teacher-centric UI)"
	@echo "  make cli             - Run CLI in interactive mode"
	@echo "  make chat            - Start a chat session (CLI)"
	@echo ""
	@echo "Student Management:"
	@echo "  make student-create name=\"Student Name\" - Create a new student"
	@echo "  make student-list           - List all students"
	@echo "  make student-select id=ID   - Select a student for operations"
	@echo "  make student-current        - Show currently selected student"
	@echo "  make student-delete id=ID   - Delete a student"
	@echo ""
	@echo "Document Management:"
	@echo "  make ingest path=/path/to/docs - Ingest documents for selected student"
	@echo "  make history         - View conversation history for selected student"
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
	@echo "  make clean                  - Clean up temporary files and caches"
	@echo "  make clean-index            - Remove vector store index"
	@echo "  make clean-state            - Remove conversation state"
	@echo "  make clean-students         - Remove all student data"
	@echo "  make clean-student-selection - Clear current student selection"
	@echo "  make clean-all              - Remove all generated data"

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

# Student management commands
student-create:
ifndef name
	@echo "Error: Please specify a student name"
	@echo "Usage: make student-create name=\"Student Name\""
	@exit 1
endif
	poetry run python -m src.cli student create "$(name)"

student-list:
	poetry run python -m src.cli student list

student-select:
ifndef id
	@echo "Error: Please specify a student ID"
	@echo "Usage: make student-select id=student_abc123_john_doe"
	@exit 1
endif
	poetry run python -m src.cli student select "$(id)"

student-current:
	poetry run python -m src.cli student current

student-delete:
ifndef id
	@echo "Error: Please specify a student ID"
	@echo "Usage: make student-delete id=student_abc123_john_doe"
	@exit 1
endif
	poetry run python -m src.cli student delete "$(id)"

student-delete-force:
ifndef id
	@echo "Error: Please specify a student ID"
	@echo "Usage: make student-delete-force id=student_abc123_john_doe"
	@exit 1
endif
	poetry run python -m src.cli student delete "$(id)" --force

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

# Clean student data
clean-students:
	rm -rf data/students/* 2>/dev/null || true
	@echo "All student data cleaned"

# Clean current student selection
clean-student-selection:
	rm -f data/students/.current_student 2>/dev/null || true
	@echo "Student selection cleared"

# Clean all generated data
clean-all: clean clean-index clean-state clean-students
	@echo "All generated data cleaned"

# Create necessary directories
setup-dirs:
	mkdir -p data/vector_store
	mkdir -p data/graph_state
	mkdir -p data/students
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
	@echo "4. Create and select a student:"
	@echo "   - Create: make student-create name=\"Student Name\""
	@echo "   - Select: make student-select id=STUDENT_ID"
	@echo ""
	@echo "5. Launch the Educational Insight Platform:"
	@echo "   - Teacher Dashboard: make ui"
	@echo "   - CLI Interface: make chat"
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