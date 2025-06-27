#!/usr/bin/env python3
"""Test script to demonstrate enhanced document processing capabilities."""

import sys
from pathlib import Path
from pprint import pprint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_ingestion import EnhancedDocumentProcessor
from src.ingestion import load_pdf, load_docx


def compare_processing(file_path: Path):
    """Compare basic vs enhanced processing for a document."""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")
    print(f"{'='*60}\n")
    
    # Basic processing
    print("BASIC PROCESSING:")
    print("-" * 30)
    try:
        if file_path.suffix.lower() == ".pdf":
            content = load_pdf(file_path)
        elif file_path.suffix.lower() in [".docx", ".doc"]:
            content = load_docx(file_path)
        else:
            print(f"Unsupported file type: {file_path.suffix}")
            return
            
        print(f"Content length: {len(content)} characters")
        print(f"First 200 chars: {content[:200]}...")
        print(f"Number of lines: {len(content.splitlines())}")
    except Exception as e:
        print(f"Error in basic processing: {e}")
    
    # Enhanced processing
    print("\n\nENHANCED PROCESSING:")
    print("-" * 30)
    try:
        processor = EnhancedDocumentProcessor(
            chunk_size=1500,
            chunk_overlap=300,
            extract_tables=True,
            preserve_formatting=True,
            use_ocr=False  # Set to True if you have scanned documents
        )
        
        documents = processor.process_document(file_path)
        
        print(f"Number of chunks: {len(documents)}")
        print(f"\nFirst chunk metadata:")
        pprint(documents[0].metadata)
        print(f"\nFirst chunk content (first 200 chars):")
        print(documents[0].page_content[:200] + "...")
        
        # Check for tables
        table_chunks = [doc for doc in documents if doc.metadata.get("element_type") == "table"]
        if table_chunks:
            print(f"\nFound {len(table_chunks)} table(s)")
            print("First table content:")
            print(table_chunks[0].page_content[:300] + "...")
        
        # Show unique metadata keys across all chunks
        all_keys = set()
        for doc in documents:
            all_keys.update(doc.metadata.keys())
        print(f"\nMetadata fields extracted: {sorted(all_keys)}")
        
    except Exception as e:
        print(f"Error in enhanced processing: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("Enhanced Document Processing Test")
    print("=================================\n")
    
    # Check for test files
    test_files = []
    
    # Look for test documents in common locations
    possible_paths = [
        Path("./data/sample_docs"),
        Path("./test_docs"),
        Path("./documents"),
        Path("."),
    ]
    
    for path in possible_paths:
        if path.exists():
            pdf_files = list(path.glob("*.pdf"))
            docx_files = list(path.glob("*.docx"))
            test_files.extend(pdf_files[:2])  # Take up to 2 PDFs
            test_files.extend(docx_files[:2])  # Take up to 2 DOCXs
            
    if not test_files:
        print("No test files found. Please provide PDF or DOCX files.")
        print("\nUsage: python test_enhanced_processing.py [file1.pdf] [file2.docx] ...")
        
        # Check command line arguments
        if len(sys.argv) > 1:
            test_files = [Path(arg) for arg in sys.argv[1:] if Path(arg).exists()]
    
    if not test_files:
        print("\nCreating a sample test document...")
        # Create a simple test document
        test_txt = Path("test_document.txt")
        test_txt.write_text("""
# Test Document

This is a test document for enhanced processing.

## Section 1: Introduction

This document contains multiple sections and paragraphs to test the chunking capabilities.

## Section 2: Data

Here's some sample data:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data A   | Data B   | Data C   |
| Data D   | Data E   | Data F   |

## Section 3: Conclusion

This concludes our test document. The enhanced processor should:
- Preserve formatting
- Extract metadata
- Handle tables properly
- Create semantic chunks
""")
        test_files = [test_txt]
    
    # Process each test file
    for file_path in test_files:
        if file_path.exists():
            compare_processing(file_path)
        else:
            print(f"File not found: {file_path}")
    
    print("\n\nTest completed!")
    print("\nTo test with your own documents:")
    print("python test_enhanced_processing.py path/to/document.pdf path/to/document.docx")


if __name__ == "__main__":
    main()