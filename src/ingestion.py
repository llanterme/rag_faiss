"""Document ingestion module for loading various file formats."""

import re
from pathlib import Path
from typing import List

import pypdf
from docx import Document as DocxDocument


def load_txt(path: Path) -> str:
    """Load text from a plain text file.

    Args:
        path: Path to the text file.

    Returns:
        Extracted and normalized text content.
    """
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove simple header/footer placeholders
    text = re.sub(r"^\s*[Pp]age\s+\d+\s+of\s+\d+\s*", "", text)
    text = re.sub(r"^\s*[Hh]eader:.*?\n", "", text)
    text = re.sub(r"\n\s*[Ff]ooter:.*?$", "", text)

    return text


def load_pdf(path: Path) -> str:
    """Extract text from a PDF document.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted and normalized text content.
    """
    reader = pypdf.PdfReader(path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove simple header/footer placeholders
    text = re.sub(r"^\s*[Pp]age\s+\d+\s+of\s+\d+\s*", "", text)
    text = re.sub(r"^\s*[Hh]eader:.*?\n", "", text)
    text = re.sub(r"\n\s*[Ff]ooter:.*?$", "", text)

    return text


def load_docx(path: Path) -> str:
    """Extract text from a DOCX document.

    Args:
        path: Path to the DOCX file.

    Returns:
        Extracted and normalized text content.
    """
    doc = DocxDocument(path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove simple header/footer placeholders
    text = re.sub(r"^\s*[Pp]age\s+\d+\s+of\s+\d+\s*", "", text)
    text = re.sub(r"^\s*[Hh]eader:.*?\n", "", text)
    text = re.sub(r"\n\s*[Ff]ooter:.*?$", "", text)

    return text
