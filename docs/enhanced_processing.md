# Enhanced Document Processing

This document explains the enhanced document processing capabilities available in the RAG FAISS system.

## Overview

The enhanced document processing module provides advanced extraction capabilities for PDF and DOCX files, including:

- **Table extraction**: Preserves tabular data structure
- **Metadata extraction**: Captures document properties, author, creation date, etc.
- **Layout preservation**: Maintains document structure and formatting
- **OCR support**: Processes scanned documents (requires PyMuPDF)
- **Smart chunking**: Context-aware splitting that preserves semantic boundaries

## Installation

### Basic Installation (Default)
The basic installation includes PyPDF and python-docx for simple document processing:
```bash
poetry install
```

### Enhanced Installation
To enable advanced features, install the optional enhanced dependencies:
```bash
poetry install --with enhanced
```

This installs:
- **pdfplumber**: Advanced PDF processing with table extraction
- **PyMuPDF (fitz)**: Fast PDF processing with OCR capabilities
- **unstructured**: Advanced document parsing with layout analysis

## Configuration

Enhanced processing is controlled via environment variables or `.env` file:

```env
# Enable/disable enhanced processing (default: true)
USE_ENHANCED_PROCESSING=true

# Extract tables from documents (default: true)
EXTRACT_TABLES=true

# Extract images from documents (default: false)
EXTRACT_IMAGES=false

# Preserve document formatting and structure (default: true)
PRESERVE_FORMATTING=true

# Use OCR for scanned documents (default: false)
USE_OCR=false

# Chunking parameters
CHUNK_SIZE=1500  # Larger chunks for better context
CHUNK_OVERLAP=300  # More overlap for continuity
```

## Usage

### CLI
When enhanced processing is enabled, the `ingest` command automatically uses the advanced features:

```bash
# Process a single PDF with table extraction
poetry run python -m src.cli ingest document.pdf

# Process a folder of mixed documents
poetry run python -m src.cli ingest ./documents
```

### Programmatic Usage
```python
from src.enhanced_ingestion import EnhancedDocumentProcessor

# Initialize processor with custom settings
processor = EnhancedDocumentProcessor(
    chunk_size=1500,
    chunk_overlap=300,
    extract_tables=True,
    preserve_formatting=True,
    use_ocr=True  # For scanned documents
)

# Process a document
documents = processor.process_document(Path("document.pdf"))

# Each document contains:
# - page_content: The extracted text
# - metadata: Rich metadata including page numbers, sections, tables, etc.
```

## Processing Strategies

### PDF Processing

The system uses a fallback strategy for PDF processing:

1. **Unstructured (if available)**: Best for complex layouts, tables, and formatting
   - Uses "hi_res" strategy for maximum accuracy
   - Extracts document structure and elements
   - Identifies headers, paragraphs, tables, etc.

2. **PDFPlumber (if available)**: Excellent for table extraction
   - Preserves table structure in markdown format
   - Maintains cell relationships
   - Good for data-heavy documents

3. **PyMuPDF (if available)**: Fast processing with OCR
   - Handles scanned documents
   - Preserves layout information
   - Extracts document metadata

4. **PyPDF (fallback)**: Basic text extraction
   - Always available
   - Simple and reliable
   - Limited formatting preservation

### DOCX Processing

1. **Unstructured (if available)**: Advanced structure extraction
   - Identifies document elements
   - Preserves formatting information
   - Handles complex layouts

2. **python-docx (fallback)**: Standard DOCX processing
   - Extracts paragraphs and tables
   - Preserves basic formatting
   - Handles document properties

## Metadata Extraction

Enhanced processing extracts rich metadata:

```python
{
    "source": "document.pdf",
    "file_type": "pdf",
    "page_number": 1,
    "total_pages": 10,
    "author": "John Doe",
    "title": "Technical Report",
    "creation_date": "2024-01-15",
    "section": "Introduction",
    "element_type": "paragraph|table|header",
    "chunk_index": 0,
    "total_chunks": 5
}
```

## Chunking Strategies

### Semantic Chunking
- Respects document structure (headers, paragraphs)
- Preserves context across chunk boundaries
- Avoids splitting tables or important sections

### Hierarchical Splitting
The system uses a hierarchy of separators:
1. Double newlines (paragraph boundaries)
2. Single newlines
3. Sentence endings (. ! ?)
4. Clause boundaries (; :)
5. Spaces
6. Characters (last resort)

### Special Handling
- **Tables**: Kept intact as single chunks when possible
- **Code blocks**: Preserved without splitting
- **Headers**: Used as section markers for context

## Performance Considerations

1. **Processing Speed**:
   - PyMuPDF: Fastest
   - PyPDF: Fast, basic extraction
   - PDFPlumber: Moderate, better for tables
   - Unstructured: Slowest, highest quality

2. **Memory Usage**:
   - Process large documents in batches
   - Use streaming for very large files
   - Consider disabling image extraction for memory-constrained environments

3. **Quality vs Speed Trade-offs**:
   - Enable `preserve_formatting` for better context
   - Disable `extract_images` unless needed
   - Use `use_ocr` only for scanned documents

## Troubleshooting

### Common Issues

1. **Import errors for optional libraries**:
   - Solution: Install with `poetry install --with enhanced`
   - The system will fall back to basic processing if libraries are missing

2. **OCR not working**:
   - Ensure PyMuPDF is installed
   - Check that the PDF contains actual scanned images

3. **Table extraction failing**:
   - Try different PDF processors (PDFPlumber vs Unstructured)
   - Some complex tables may require manual processing

4. **Memory errors with large documents**:
   - Reduce chunk_size
   - Process documents individually
   - Disable image extraction

### Debug Mode

Enable logging to see which processor is being used:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Best Practices

1. **Document Preparation**:
   - Use searchable PDFs when possible
   - Avoid scanned documents unless OCR is enabled
   - Ensure consistent formatting

2. **Configuration Tuning**:
   - Adjust chunk_size based on your use case
   - Increase chunk_overlap for better context preservation
   - Enable only the features you need

3. **Quality Assurance**:
   - Review extracted chunks for completeness
   - Test retrieval with sample queries
   - Monitor metadata quality

## Future Enhancements

Planned improvements include:
- Support for more file formats (EPUB, HTML, RTF)
- Advanced table extraction with structure preservation
- Multi-modal embeddings for images and diagrams
- Language-specific processing optimizations
- Automatic quality assessment and validation