"""Logfire observability setup for monitoring LLM interactions."""

import os
import logging
from typing import Optional
from functools import wraps

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

logger = logging.getLogger(__name__)


class LogfireManager:
    """Manages Logfire initialization and instrumentation."""
    
    def __init__(self):
        self.initialized = False
        self.enabled = False
    
    def initialize(
        self, 
        enable: bool = True, 
        token: Optional[str] = None,
        project_name: str = "document-chat"
    ) -> bool:
        """Initialize Logfire with the given configuration.
        
        Args:
            enable: Whether to enable Logfire
            token: Logfire API token (optional, can use local mode)
            project_name: Project name for Logfire
            
        Returns:
            True if successfully initialized, False otherwise
        """
        if not LOGFIRE_AVAILABLE:
            logger.warning("Logfire not available. Install with: pip install logfire")
            return False
        
        if not enable:
            logger.info("Logfire disabled by configuration")
            return False
        
        try:
            # Configure Logfire
            if token:
                # Remote logging with token
                logfire.configure(token=token)
            else:
                # Local logging mode - console output only
                logfire.configure(send_to_logfire=False)
            
            # Instrument OpenAI if available
            try:
                logfire.instrument_openai()
                logger.info("Instrumented OpenAI")
            except Exception as e:
                logger.debug(f"Could not instrument OpenAI: {e}")
            
            # Instrument HTTP requests for LLM calls
            try:
                logfire.instrument_httpx()
                logfire.instrument_requests()
                logger.info("Instrumented HTTP libraries")
            except Exception as e:
                logger.debug(f"Could not instrument HTTP libraries: {e}")
            
            self.initialized = True
            self.enabled = True
            
            logger.info(f"Logfire initialized for project: {project_name}")
            if token:
                logger.info("Using remote Logfire logging")
            else:
                logger.info("Using local Logfire logging (console only)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Logfire: {e}")
            return False
    
    def log_document_processing(self, file_path: str, chunk_count: int, processing_time: float):
        """Log document processing metrics."""
        if not self.enabled:
            return
            
        logfire.info(
            "Document processed",
            file_path=file_path,
            chunk_count=chunk_count,
            processing_time_seconds=processing_time,
        )
    
    def log_embedding_creation(self, chunk_count: int, processing_time: float, provider: str):
        """Log embedding creation metrics."""
        if not self.enabled:
            return
            
        logfire.info(
            "Embeddings created",
            chunk_count=chunk_count,
            processing_time_seconds=processing_time,
            provider=provider,
        )
    
    def log_query(self, query: str, response: str, sources: list, response_time: float):
        """Log query and response with sources."""
        if not self.enabled:
            return
            
        logfire.info(
            "RAG query processed",
            query=query,
            response_length=len(response),
            source_count=len(sources),
            response_time_seconds=response_time,
            sources=[s.get('source', 'unknown') for s in sources] if sources else [],
        )
    
    def log_conversation_turn(self, turn_number: int, query: str, response: str):
        """Log a conversation turn."""
        if not self.enabled:
            return
            
        logfire.info(
            "Conversation turn",
            turn_number=turn_number,
            query_length=len(query),
            response_length=len(response),
        )
    
    def span(self, operation_name: str, **kwargs):
        """Create a Logfire span for operation tracking."""
        if not self.enabled or not LOGFIRE_AVAILABLE:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
        
        return logfire.span(operation_name, **kwargs)
    
    def log_error(self, error: Exception, context: str):
        """Log an error with context."""
        if not self.enabled:
            return
            
        logfire.error(
            f"Error in {context}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
        )


# Global instance
logfire_manager = LogfireManager()


def log_operation(operation_name: str):
    """Decorator to automatically log function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with logfire_manager.span(operation_name):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    logfire_manager.log_error(e, operation_name)
                    raise
        return wrapper
    return decorator


def setup_logfire_from_config():
    """Initialize Logfire using the application configuration."""
    from src.config import settings
    
    return logfire_manager.initialize(
        enable=settings.enable_logfire,
        token=settings.logfire_token,
        project_name=settings.logfire_project_name,
    )


def get_logfire_status() -> dict:
    """Get the current status of Logfire."""
    return {
        "available": LOGFIRE_AVAILABLE,
        "initialized": logfire_manager.initialized,
        "enabled": logfire_manager.enabled,
    }