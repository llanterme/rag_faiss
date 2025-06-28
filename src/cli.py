"""Command-line interface for the document chat application."""

import os
import sys
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Suppress specific LangChain deprecation warnings related to Ollama
# These are no longer needed as we've migrated to langchain-ollama
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

import typer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage

from src.config import LLMProvider, EmbeddingProvider, settings
from src.embeddings import create_faiss_index, load_index, save_index
from src.ingestion import load_docx, load_pdf, load_txt
from src.enhanced_ingestion import EnhancedDocumentProcessor
from src.langgraph_chain import build_retrieval_chain

# Create enums for CLI provider options that match our internal enums
class ProviderOption(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


app = typer.Typer(help="Document chat application")

# Store the chain and config globally
qa_chain: Any = None
chain_config: Optional[Dict] = None


@app.command()
def ingest(
    directory: str = typer.Argument(
        ..., help="Directory containing documents to ingest"
    )
):
    """Ingest documents from a directory into the vector store."""
    # Display the embedding configuration being used
    typer.echo(f"Using {settings.embedding_provider.value.title()} embeddings with model {settings.embedding_model}")
    
    # Check if OpenAI API key is set when using OpenAI embeddings
    if settings.embedding_provider == EmbeddingProvider.OPENAI and not settings.openai_api_key:
        typer.echo("Error: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Check if Ollama is running when using Ollama embeddings
    if settings.embedding_provider == EmbeddingProvider.OLLAMA:
        import requests
        try:
            requests.get(f"{settings.ollama_base_url}/api/version", timeout=2)
        except requests.exceptions.ConnectionError:
            typer.echo(f"\nError: Cannot connect to Ollama at {settings.ollama_base_url}")
            typer.echo("Make sure Ollama is installed and running.")
            typer.echo("Installation instructions: https://ollama.com/download")
            typer.echo("\nAfter installing, start Ollama and try again.")
            return
    
    # Convert to Path object
    dir_path = Path(directory)
    
    # Check if directory exists
    if not dir_path.exists() or not dir_path.is_dir():
        typer.echo(f"Error: {directory} is not a valid directory")
        return

    # Use enhanced processing if enabled
    if settings.use_enhanced_processing:
        typer.echo("Using enhanced document processing...")
        processor = EnhancedDocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            extract_tables=settings.extract_tables,
            extract_images=settings.extract_images,
            preserve_formatting=settings.preserve_formatting,
            use_ocr=settings.use_ocr,
        )
        
        # Collect all files to process
        files_to_process = []
        if dir_path.is_file():
            files_to_process.append(dir_path)
        else:
            # Process all supported files in directory
            typer.echo(f"Scanning directory: {dir_path}")
            for file_path in dir_path.glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in [".txt", ".md", ".pdf", ".docx", ".doc"]:
                    files_to_process.append(file_path)
        
        if not files_to_process:
            typer.echo("No supported documents found to ingest.")
            return
        
        # Process documents
        all_docs = []
        for file_path in files_to_process:
            typer.echo(f"Processing {file_path.name}...")
            try:
                docs = processor.process_document(file_path)
                typer.echo(f"  - Generated {len(docs)} chunks")
                all_docs.extend(docs)
            except Exception as e:
                typer.echo(f"  - Error processing {file_path.name}: {str(e)}")
                continue
        
        if not all_docs:
            typer.echo("No documents were successfully processed.")
            return
        
        # Extract chunks and metadata
        chunks = [doc.page_content for doc in all_docs]
        metadatas = [doc.metadata for doc in all_docs]
    
    else:
        # Fallback to original processing
        documents: List[Tuple[str, str]] = []  # List of (doc_path, content)

        if dir_path.is_file():
            # Process single file
            if dir_path.suffix.lower() == ".txt":
                typer.echo(f"Ingesting TXT file: {dir_path}")
                documents.append((str(dir_path), load_txt(dir_path)))
            elif dir_path.suffix.lower() == ".pdf":
                typer.echo(f"Ingesting PDF file: {dir_path}")
                documents.append((str(dir_path), load_pdf(dir_path)))
            elif dir_path.suffix.lower() in [".docx", ".doc"]:
                typer.echo(f"Ingesting DOCX file: {dir_path}")
                documents.append((str(dir_path), load_docx(dir_path)))
            else:
                typer.echo(f"Unsupported file type: {dir_path.suffix}")
                return
        elif dir_path.is_dir():
            # Process all supported files in directory
            typer.echo(f"Ingesting files from directory: {dir_path}")
            for file_path in dir_path.glob("**/*"):
                if file_path.is_file():
                    if file_path.suffix.lower() == ".txt":
                        typer.echo(f"Ingesting TXT file: {file_path}")
                        documents.append((str(file_path), load_txt(file_path)))
                    elif file_path.suffix.lower() == ".pdf":
                        typer.echo(f"Ingesting PDF file: {file_path}")
                        documents.append((str(file_path), load_pdf(file_path)))
                    elif file_path.suffix.lower() in [".docx", ".doc"]:
                        typer.echo(f"Ingesting DOCX file: {file_path}")
                        documents.append((str(file_path), load_docx(file_path)))

        if not documents:
            typer.echo("No supported documents found to ingest.")
            return

        # Create text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )

        # Split documents into chunks
        typer.echo(f"Splitting {len(documents)} document(s) into chunks...")
        chunks = []
        metadatas = []
        for doc_path, content in documents:
            doc_chunks = text_splitter.split_text(content)
            doc_name = Path(doc_path).name
            typer.echo(f"  - {doc_name}: {len(doc_chunks)} chunks")
            
            # Add document source to each chunk's metadata
            for chunk in doc_chunks:
                chunks.append(chunk)
                metadatas.append({"source": doc_name, "source_path": doc_path})

    typer.echo(f"Creating vector embeddings for {len(chunks)} text chunks...")
    index = create_faiss_index(chunks, metadatas=metadatas)

    # Save index to disk
    save_index(index)
    typer.echo(
        f"Index saved to {settings.vector_store_path / settings.faiss_index_filename}"
    )


@app.command()
def chat(
    llm_provider: Optional[str] = typer.Option(
        None, 
        help="LLM provider to use for chat (openai or ollama)"
    ),
    llm_model: Optional[str] = typer.Option(
        None, 
        help="LLM model name to use for chat"
    )
):
    """Start an interactive chat session with the AI assistant."""
    global qa_chain, chain_config
    
    # Update LLM provider if specified (this doesn't affect embeddings)
    if llm_provider:
        try:
            settings.llm_provider = LLMProvider(llm_provider.lower())
        except ValueError:
            typer.echo(f"Invalid provider: {llm_provider}. Must be 'openai' or 'ollama'.")
            return
    
    # Update LLM model if specified (this doesn't affect embeddings)
    if llm_model:
        settings.llm_model = llm_model
    
    # Display current settings
    print(f"Active LLM provider: {settings.llm_provider.value}")
    print(f"Active model: {settings.llm_model}")
    
    # Check if OpenAI API key is set when using OpenAI
    if settings.llm_provider == LLMProvider.OPENAI and not settings.openai_api_key:
        print("Error: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        return
    
    # Check if Ollama is running when using Ollama
    if settings.llm_provider == LLMProvider.OLLAMA:
        import requests
        try:
            requests.get(f"{settings.ollama_base_url}/api/version", timeout=2)
        except requests.exceptions.ConnectionError:
            print(f"\nError: Cannot connect to Ollama at {settings.ollama_base_url}")
            print("Make sure Ollama is installed and running.")
            print("Installation instructions: https://ollama.com/download")
            print("\nAfter installing, start Ollama and try again.")
            return
    
    try:
        # Try to load existing index
        index = load_index()
        print("Loaded existing vector index.")
    except FileNotFoundError:
        print("No vector index found. Please ingest documents first.")
        return
    except ValueError as e:
        # This will be raised by our modified load_index function
        # when there's a dimension mismatch
        print(f"Error loading index: {str(e)}")
        return

    # Build retrieval chain with LangGraph
    qa_chain = build_retrieval_chain(index)

    typer.echo("Starting chat session. Type 'exit' to quit.")

    while True:
        # Get user query
        query = typer.prompt("You")

        if query.lower() in ["exit", "quit"]:
            break
            
        # Process the user query through the graph
        try:
            # Use the LangGraph chain with the question
            result = qa_chain.invoke({"question": query})
            
            # LangGraph 0.4.x returns the final state with answer field
            if isinstance(result, dict):
                # First check for direct answer field
                if "answer" in result and result["answer"]:
                    typer.echo(f"AI: {result['answer']}")
                # Then look for messages (especially the last one)
                elif "messages" in result and result["messages"]:
                    for message in result["messages"]:
                        if isinstance(message, AIMessage) and message == result["messages"][-1]:
                            typer.echo(f"AI: {message.content}")
                            break
                else:
                    # Fallback if we can't find the answer
                    typer.echo("AI: I couldn't generate a response. Please try again.")
            else:
                typer.echo(f"AI: {result}")
                
        except Exception as e:
            typer.echo(f"Error processing query: {str(e)}")
            typer.echo("Please try again or restart the chat session.")
            continue


@app.command()
def history():
    """Print conversation history."""
    global qa_chain, chain_config

    if qa_chain is None:
        typer.echo("No active chat session or conversation history.")
        return

    # Get the current state from the LangGraph chain
    try:
        # Different ways to access state depending on LangGraph version
        try:
            # Try direct state access first
            current_state = qa_chain.get_current_state()
        except AttributeError:
            try:
                # Try getting state through config
                if chain_config:
                    current_state = qa_chain.get_state_value(chain_config)
                else:
                    current_state = qa_chain.get_state()
            except Exception:
                # Fall back to direct state if other methods fail
                current_state = qa_chain.get_state() if hasattr(qa_chain, "get_state") else None
        
        # Extract messages from the state
        if current_state and isinstance(current_state, dict) and "messages" in current_state and current_state["messages"]:
            # Print conversation history
            typer.echo("\n--- Conversation History ---")
            for message in current_state["messages"]:
                # Handle different message types
                if isinstance(message, HumanMessage):
                    typer.echo(f"Human: {message.content}")
                elif isinstance(message, AIMessage):
                    typer.echo(f"AI: {message.content}")
                else:
                    typer.echo(f"Message: {str(message)}")
            typer.echo("---------------------------\n")
        else:
            typer.echo("No messages in conversation history yet. Start a chat to see history.")
    except Exception as e:
        typer.echo(f"Error retrieving conversation history: {str(e)}")
        typer.echo("Try starting a new chat session first.")
        return


@app.command()
def provider():
    """Display the current LLM and embedding provider configuration."""
    typer.echo("=== LLM Configuration ===")
    typer.echo(f"LLM provider: {settings.llm_provider.value}")
    typer.echo(f"LLM model: {settings.llm_model}")
    
    typer.echo("\n=== Embedding Configuration ===")
    typer.echo(f"Embedding provider: {settings.embedding_provider.value}")
    typer.echo(f"Embedding model: {settings.embedding_model}")
    
    # Show additional provider-specific information
    typer.echo("\n=== Provider Details ===")
    if settings.embedding_provider == EmbeddingProvider.OPENAI or settings.llm_provider == LLMProvider.OPENAI:
        api_key_status = "configured" if settings.openai_api_key else "missing"
        typer.echo(f"OpenAI API key: {api_key_status}")
    
    if settings.embedding_provider == EmbeddingProvider.OLLAMA or settings.llm_provider == LLMProvider.OLLAMA:
        typer.echo(f"Ollama base URL: {settings.ollama_base_url}")
    
    # Provide instructions for changing the configuration
    typer.echo("\nTo change the LLM configuration (for chat), use:")
    typer.echo("  chat --llm-provider openai|ollama [--llm-model MODEL_NAME]")
    
    typer.echo("\nTo change the embedding configuration, edit your .env file or settings:")
    typer.echo("  EMBEDDING_PROVIDER=openai|ollama")
    typer.echo("  EMBEDDING_MODEL=model_name")
    typer.echo("\nNote: Changing the embedding configuration requires re-ingesting documents.")


@app.command()
def test_prompts(
    style: str = typer.Option(
        "default", 
        help="Prompt style to test: default, detailed, concise, academic, technical"
    )
):
    """Test different prompt styles with a sample query."""
    from src.prompts import create_rag_prompt
    
    # Sample context and question for demonstration
    sample_context = """
Company Policy Manual - Section 3.2
Employees are entitled to 15 days of paid vacation per year. Vacation requests must be submitted at least 2 weeks in advance through the HR portal. Emergency leave may be granted with manager approval.

Benefits Guide - Chapter 4  
Health insurance covers employee and immediate family members. The company matches 401k contributions up to 5% of salary. Professional development budget is $2000 per year per employee.
"""
    
    sample_question = "What are the vacation and benefits policies?"
    
    # Test the prompt style
    valid_styles = ["default", "detailed", "concise", "academic", "technical"]
    if style not in valid_styles:
        typer.echo(f"❌ Invalid style: {style}")
        typer.echo(f"Valid options: {', '.join(valid_styles)}")
        return
    
    typer.echo(f"🎯 Testing '{style}' prompt style")
    typer.echo("=" * 60)
    
    enhanced_prompt = create_rag_prompt(
        context=sample_context,
        question=sample_question,
        style=style
    )
    
    typer.echo("📝 Generated Prompt:")
    typer.echo("-" * 30)
    typer.echo(enhanced_prompt)
    
    typer.echo("\n" + "=" * 60)
    typer.echo(f"💡 To use this style permanently:")
    typer.echo(f"   Add 'PROMPT_STYLE={style}' to your .env file")
    
    typer.echo(f"\n🧪 To test with actual documents:")
    typer.echo(f"   1. Ingest some documents")
    typer.echo(f"   2. Set PROMPT_STYLE={style} in .env")
    typer.echo(f"   3. Run: poetry run python -m src.cli chat")


@app.command()
def observability():
    """Show observability status and configuration."""
    from src.observability import get_logfire_status
    
    status = get_logfire_status()
    
    typer.echo("🔍 Observability Status")
    typer.echo("=" * 30)
    typer.echo(f"Logfire Available: {'✅' if status['available'] else '❌'}")
    typer.echo(f"Logfire Initialized: {'✅' if status['initialized'] else '❌'}")
    typer.echo(f"Logfire Enabled: {'✅' if status['enabled'] else '❌'}")
    
    typer.echo("\n📊 Configuration")
    typer.echo("=" * 30)
    typer.echo(f"Enable Logfire: {settings.enable_logfire}")
    typer.echo(f"Logfire Token: {'Set' if settings.logfire_token else 'Not set (using local mode)'}")
    typer.echo(f"Project Name: {settings.logfire_project_name}")
    typer.echo(f"Prompt Logging: {settings.logfire_log_prompts}")
    typer.echo(f"Prompt Style: {settings.prompt_style}")
    
    if status['enabled']:
        typer.echo("\n📈 Features Active")
        typer.echo("=" * 30)
        typer.echo("• LLM interaction logging")
        typer.echo("• Document processing metrics")
        typer.echo("• Embedding creation tracking")
        typer.echo("• Query performance monitoring")
        typer.echo("• Error tracking and debugging")
        
        if not settings.logfire_token:
            typer.echo("\n💡 Tip: Set LOGFIRE_TOKEN in .env to send logs to Logfire cloud dashboard")
        else:
            typer.echo("\n🌐 Logs are being sent to Logfire cloud dashboard")
    else:
        typer.echo("\n⚠️ Observability is disabled. Set ENABLE_LOGFIRE=true in .env to enable.")


@app.command()
def exit():
    """Exit the application."""
    typer.echo("Exiting document-chat application.")
    sys.exit(0)


if __name__ == "__main__":
    # Ensure the vector store directory exists
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)

    app()
