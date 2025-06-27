# Observability with Logfire

This document explains the observability features available in the RAG FAISS system using Pydantic Logfire.

## Overview

Logfire provides comprehensive observability for your LLM interactions, giving you insights into:

- **LLM Performance**: Track response times, token usage, and model performance
- **Document Processing**: Monitor ingestion times, chunk counts, and processing efficiency  
- **Embedding Creation**: Track embedding generation performance and costs
- **Query Analytics**: Understand user queries, response quality, and retrieval accuracy
- **Error Tracking**: Catch and debug issues across the entire pipeline

## Configuration

### Basic Setup (Local Logging)

By default, Logfire is enabled and logs to the console:

```env
# .env file
ENABLE_LOGFIRE=true  # Default: true
LOGFIRE_PROJECT_NAME=document-chat  # Default: document-chat
```

### Cloud Dashboard (Optional)

For the full Logfire experience with web dashboard:

1. Sign up at [logfire.pydantic.dev](https://logfire.pydantic.dev)
2. Create a project and get your token
3. Add to your `.env` file:

```env
LOGFIRE_TOKEN=your_token_here
```

## Features

### 1. LLM Interaction Logging

**What's tracked:**
- Query text and response content
- Response times and token counts
- Model and provider information
- Error handling and retries

**Example log output:**
```
09:53:28.743 Embedding Creation with 'text-embedding-ada-002' [LLM]
09:53:32.451 RAG query processed
  ├── query: "What is the main topic of the document?"
  ├── response_time_seconds: 1.23
  ├── source_count: 2
  └── response_length: 156
```

### 2. Document Processing Metrics

**What's tracked:**
- File processing times by type (PDF, DOCX, TXT)
- Chunk generation statistics
- Extraction method success/failure
- File sizes and processing efficiency

**Example log output:**
```
09:53:27.563 document_processing
  ├── file_path: "document.pdf"
  ├── file_type: ".pdf"
  ├── file_size_bytes: 1024000
  ├── chunks_created: 15
  └── processing_time_seconds: 2.34
```

### 3. Embedding Creation Tracking

**What's tracked:**
- Embedding provider and model
- Batch sizes and processing times
- Total text length processed
- API call performance

**Example log output:**
```
09:53:28.582 embedding_creation
  ├── chunk_count: 15
  ├── total_text_length: 12450
  ├── embedding_provider: "openai"
  ├── embedding_model: "text-embedding-ada-002"
  └── processing_time_seconds: 3.87
```

### 4. Query Performance Monitoring

**What's tracked:**
- Document retrieval performance
- Context length and relevance
- LLM generation times
- End-to-end query latency

**Example log output:**
```
09:53:27.414 document_retrieval
  ├── query: "What are the key findings?"
  ├── query_length: 26
  ├── documents_retrieved: 4
  └── sources: ["doc1.pdf", "doc2.pdf"]

09:53:27.416 llm_generation
  ├── context_length: 2340
  ├── llm_provider: "openai"
  ├── llm_model: "gpt-4o"
  ├── message_count: 3
  └── response_time_seconds: 1.45
```

### 5. Error Tracking and Debugging

**What's tracked:**
- Exception types and messages
- Stack traces and context
- Operation names where errors occur
- Recovery and retry attempts

**Example log output:**
```
Error in llm_generation
  ├── error_type: "RateLimitError"
  ├── error_message: "Rate limit exceeded"
  └── context: "llm_generation"
```

## Usage

### CLI Commands

Check observability status:
```bash
poetry run python -m src.cli observability
```

View real-time logs during operations:
```bash
# Ingest documents with logging
poetry run python -m src.cli ingest documents/

# Chat with logging
poetry run python -m src.cli chat
```

### Programmatic Usage

```python
from src.observability import logfire_manager
import logfire

# Custom operation logging
with logfire.span("custom_operation") as span:
    span.set_attribute("user_id", "123")
    span.set_attribute("operation_type", "bulk_processing")
    
    # Your code here
    result = process_documents()
    
    span.set_attribute("documents_processed", len(result))

# Custom metrics
logfire_manager.log_query(
    query="Custom query",
    response="Custom response", 
    sources=[{"source": "doc.pdf"}],
    response_time=1.23
)
```

### Web UI Integration

The Streamlit UI automatically logs all operations when observability is enabled. No additional configuration needed.

## Advanced Configuration

### Environment Variables

```env
# Core settings
ENABLE_LOGFIRE=true
LOGFIRE_TOKEN=your_token_here
LOGFIRE_PROJECT_NAME=my-rag-project

# Processing settings
USE_ENHANCED_PROCESSING=true
EXTRACT_TABLES=true
PRESERVE_FORMATTING=true

# Performance tuning
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

### Custom Instrumentation

```python
from src.observability import log_operation

@log_operation("custom_function")
def my_custom_function():
    # Function automatically logged
    pass

# Manual span creation
with logfire_manager.span("database_query") as span:
    span.set_attribute("table", "documents")
    span.set_attribute("query_type", "SELECT")
    # Database operation here
```

## Monitoring and Alerts

### Key Metrics to Watch

1. **Response Time Percentiles**
   - P50, P95, P99 for query response times
   - Embedding creation latency
   - Document processing speed

2. **Error Rates**
   - LLM API failures
   - Document processing errors
   - Embedding generation failures

3. **Usage Patterns**
   - Queries per hour/day
   - Document ingestion frequency
   - Peak usage times

4. **Resource Utilization**
   - Token usage and costs
   - API rate limit utilization
   - Processing queue lengths

### Setting Up Alerts (Cloud Dashboard)

1. Navigate to your Logfire project dashboard
2. Go to Alerts section
3. Create alerts for:
   - High error rates (>5%)
   - Slow response times (>10 seconds)
   - API quota exhaustion
   - Processing failures

## Troubleshooting

### Common Issues

1. **Logfire Not Initializing**
   ```bash
   # Check status
   poetry run python -m src.cli observability
   
   # Verify installation
   poetry show logfire
   ```

2. **Missing Logs**
   - Ensure `ENABLE_LOGFIRE=true` in .env
   - Check console output for initialization errors
   - Verify network connectivity (for cloud mode)

3. **Performance Impact**
   - Logging adds minimal overhead (<5ms per operation)
   - Disable in production if needed: `ENABLE_LOGFIRE=false`
   - Use sampling for high-volume applications

4. **Token Issues**
   - Verify token is correct and active
   - Check project permissions
   - Ensure network allows HTTPS to logfire.pydantic.dev

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run operations to see detailed logs
```

## Best Practices

### Development
- Keep observability enabled during development
- Use local mode to avoid token usage
- Monitor console output for immediate feedback

### Production
- Use cloud dashboard for centralized monitoring
- Set up alerts for critical errors
- Monitor costs and token usage
- Consider log sampling for high-volume systems

### Security
- Never commit tokens to version control
- Use environment variables for all configuration
- Regularly rotate Logfire tokens
- Monitor for sensitive data in logs

## Integration Examples

### Custom RAG Pipeline
```python
import logfire
from src.observability import logfire_manager

@logfire.instrument
class CustomRAGPipeline:
    def process_query(self, query: str):
        with logfire.span("custom_rag_pipeline") as span:
            span.set_attribute("query", query)
            
            # Retrieval
            docs = self.retrieve_documents(query)
            span.set_attribute("docs_retrieved", len(docs))
            
            # Generation
            response = self.generate_response(docs, query)
            span.set_attribute("response_length", len(response))
            
            return response
```

### Batch Processing
```python
def process_document_batch(documents):
    with logfire.span("batch_processing") as span:
        span.set_attribute("batch_size", len(documents))
        
        results = []
        for i, doc in enumerate(documents):
            with logfire.span(f"process_document_{i}"):
                result = process_single_document(doc)
                results.append(result)
        
        span.set_attribute("successful_docs", len(results))
        return results
```

## Resources

- [Logfire Documentation](https://logfire.pydantic.dev/docs/)
- [OpenTelemetry Python](https://opentelemetry-python.readthedocs.io/)
- [Pydantic Logfire GitHub](https://github.com/pydantic/logfire)
- [LLM Observability Best Practices](https://logfire.pydantic.dev/docs/guides/web_ui/)