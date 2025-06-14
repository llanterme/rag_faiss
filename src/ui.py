"""Streamlit UI for document chat application.

This module provides a web interface for document ingestion, chat interaction,
and source attribution using Streamlit. Supports both individual file uploads
and folder-based document ingestion.
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage

from src.config import settings
from src.embeddings import create_faiss_index, load_index, save_index
from src.ingestion import load_txt, load_pdf, load_docx
from src.langgraph_chain import build_retrieval_chain


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "sources" not in st.session_state:
        st.session_state.sources = []
    
    # Initialize index_exists flag if not present
    if "index_exists" not in st.session_state:
        st.session_state.index_exists = False
        
    if "retrieval_chain" not in st.session_state:
        # Check if the index file exists before attempting to load it
        index_path = settings.vector_store_path / settings.faiss_index_filename
        if not index_path.exists() or not (index_path / "index.faiss").exists():
            st.info(f"No existing index found at {index_path}")
            st.info("Please upload documents using the sidebar to create a new index.")
            st.session_state.index_exists = False
            return
            
        try:
            # Try to load existing index
            st.info("Attempting to load vector index...")
            index = load_index()
            st.success(f"Successfully loaded FAISS index with {len(index.index_to_docstore_id)} documents")
            
            # Initialize the retrieval chain
            st.info("Building retrieval chain...")
            st.session_state.retrieval_chain = build_retrieval_chain(index)
            st.success("Retrieval chain built successfully")
            st.session_state.index_exists = True
        except Exception as e:
            import traceback
            st.error(f"Error loading index: {str(e)}")
            st.code(traceback.format_exc())
            st.session_state.index_exists = False


def display_chat_history():
    """Display chat history from session state."""
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
                
                # Display sources if available for this message
                if hasattr(message, "sources") and message.sources:
                    with st.expander("Sources"):
                        for source in message.sources:
                            st.write(f"📄 {source['source']}")


def process_file(file_path: Path) -> Tuple[str, str]:
    """Process a single file and return its content.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Tuple containing the file name and extracted text content
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
        
    file_name = file_path.name
    
    # Process based on file type
    if file_path.suffix.lower() == ".txt":
        content = load_txt(file_path)
    elif file_path.suffix.lower() == ".pdf":
        content = load_pdf(file_path)
    elif file_path.suffix.lower() in [".docx", ".doc"]:
        content = load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
    return file_name, content


def process_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """Process an uploaded file and return its content.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple containing the file path and extracted text content
    """
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    
    # Process based on file type
    file_path = Path(temp_path)
    try:
        file_name = uploaded_file.name
        if file_name.lower().endswith(".txt"):
            content = load_txt(file_path)
        elif file_name.lower().endswith(".pdf"):
            content = load_pdf(file_path)
        elif file_name.lower().endswith((".docx", ".doc")):
            content = load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
            
        return file_name, content
    except Exception as e:
        os.unlink(temp_path)  # Clean up the temp file
        raise e





def count_supported_files(directory_path: str) -> Tuple[int, List[str]]:
    """Count supported document files in a directory and its subdirectories.
    
    Args:
        directory_path: The path to the directory to scan
        
    Returns:
        A tuple with (count, sample_files) where count is the total number of files
        and sample_files is a list of example file paths (limited to 5)
    """
    count = 0
    sample_files = []
    
    try:
        # Recursively scan the folder for supported files
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith((".txt", ".pdf", ".docx", ".doc")):
                    count += 1
                    if len(sample_files) < 5:
                        # Add a shortened relative path for display
                        rel_path = os.path.relpath(os.path.join(root, file), directory_path)
                        sample_files.append(rel_path)
    except Exception as e:
        print(f"Error scanning directory {directory_path}: {e}")
    
    return count, sample_files


def process_folder(folder_path: str) -> List[Tuple[str, str]]:
    """Process all supported files in a folder.
    
    Args:
        folder_path: Path to the folder containing documents
        
    Returns:
        List of tuples containing file names and their content
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    if not folder.is_dir():
        raise ValueError(f"{folder_path} is not a directory")
    
    documents = []  # List of (doc_name, content)
    supported_extensions = [".txt", ".pdf", ".docx", ".doc"]
    
    # Find all supported files in the directory (including subdirectories)
    for file_path in folder.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                file_name, content = process_file(file_path)
                documents.append((file_name, content))
            except Exception as e:
                # Log the error but continue processing other files
                print(f"Error processing {file_path}: {str(e)}")
    
    return documents


def ingest_documents(files=None, folder_path=None):
    """Process and ingest documents into the vector store.
    
    Args:
        files: List of uploaded Streamlit file objects (optional)
        folder_path: Path to a folder containing documents (optional)
    """
    if not files and not folder_path:
        st.error("No files or folder specified")
        return
    
    documents = []  # List of (doc_name, content)
    progress_text = st.empty()
    
    # Process uploaded files if provided
    if files:
        for file in files:
            progress_text.write(f"Processing {file.name}...")
            try:
                doc_name, content = process_uploaded_file(file)
                documents.append((doc_name, content))
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
    
    # Process folder if provided
    if folder_path:
        progress_text.write(f"Processing files from folder: {folder_path}...")
        try:
            folder_documents = process_folder(folder_path)
            if folder_documents:
                progress_text.write(f"Found {len(folder_documents)} documents in folder")
                documents.extend(folder_documents)
            else:
                st.warning(f"No supported documents found in folder: {folder_path}")
        except Exception as e:
            st.error(f"Error processing folder {folder_path}: {str(e)}")
    
    if not documents:
        st.error("No valid documents to process")
        return
    
    # Create text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    
    # Split documents into chunks with metadata
    progress_text.write(f"Splitting {len(documents)} document(s) into chunks...")
    chunks = []
    metadatas = []
    
    for doc_name, content in documents:
        doc_chunks = text_splitter.split_text(content)
        
        # Add document source to each chunk's metadata
        for chunk in doc_chunks:
            chunks.append(chunk)
            metadatas.append({"source": doc_name, "source_path": doc_name})
    
    # Create vector embeddings
    progress_text.write(f"Creating vector embeddings for {len(chunks)} text chunks...")
    with st.spinner("Creating embeddings... This might take a while."):
        index = create_faiss_index(chunks, metadatas=metadatas)
    
    # Save index to disk
    save_index(index)
    progress_text.write("✅ Documents processed and indexed successfully!")
    
    # Update the retrieval chain with the new index
    st.session_state.retrieval_chain = build_retrieval_chain(index)
    st.session_state.index_exists = True


def main():
    """Main Streamlit UI application."""
    st.set_page_config(
        page_title="Document Chat",
        page_icon="📚",
        layout="wide",
    )
    
    st.title("📚 Document Chat")
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar for document upload and ingestion
    with st.sidebar:
        st.header("Document Upload")
        st.write("Upload documents to chat with")
        
        upload_tab, folder_tab = st.tabs(["Upload Files", "Select Folder"])
        
        with upload_tab:
            uploaded_files = st.file_uploader(
                "Upload PDF, DOCX, or TXT files",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"]
            )
            
            if st.button("Process Files", disabled=len(uploaded_files) == 0):
                ingest_documents(files=uploaded_files)
        
        with folder_tab:
            st.write("Select a folder containing documents:")
            
            # Upload method 1: Direct file upload (multiple files)
            uploaded_files = st.file_uploader(
                "Choose documents", 
                type=["pdf", "txt", "docx", "doc"], 
                accept_multiple_files=True,
                help="Upload multiple document files (.txt, .pdf, .docx)"
            )
            
            if uploaded_files:
                st.success(f"Selected {len(uploaded_files)} file(s)")
                with st.expander("Files ready for processing"):
                    for file in uploaded_files:
                        st.write(f"📄 {file.name}")
                        
                if st.button(f"Process {len(uploaded_files)} Files", type="primary"):
                    ingest_documents(files=uploaded_files)
            
            # Leave some space between UI elements
            st.write("")
                
            # Leave a clean UI
            st.write("")
    
    # Check if vector store exists
    if not st.session_state.index_exists:
        st.info("No documents have been processed yet. Please upload documents using the sidebar.")
        return
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message to chat history
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from retrieval chain
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                try:
                    # Process the query through the graph
                    with st.spinner("Searching documents for relevant information..."):
                        result = st.session_state.retrieval_chain.invoke({"question": prompt})
                    
                    # Debug output to see what's being returned
                    with st.expander("Debug Response Structure", expanded=False):
                        st.write("Response type:", type(result))
                        if isinstance(result, dict):
                            st.write("Response keys:", list(result.keys()))
                            for key in result.keys():
                                st.write(f"{key} type:", type(result[key]))
                    
                    # Multiple ways to extract the answer, similar to CLI implementation
                    answer = None
                    sources = []
                    
                    if not isinstance(result, dict):
                        response_placeholder.error(f"Unexpected response format. Expected dictionary but got {type(result)}")
                        return
                        
                    # First try the direct answer field
                    if "answer" in result and result["answer"]:
                        answer = result["answer"]
                        sources = result.get("sources", [])
                    # Then look for messages (especially the last AIMessage)
                    elif "messages" in result and result["messages"]:
                        for message in result["messages"]:
                            if isinstance(message, AIMessage) and message == result["messages"][-1]:
                                answer = message.content
                                # Try to get sources from the message if available
                                if hasattr(message, "sources"):
                                    sources = message.sources
                                break
                    
                    if answer is None:
                        response_placeholder.error("The system couldn't generate a response. Please try asking a different question or upload more relevant documents.")
                        return
                    
                    # Handle different types of answers including error messages
                    if answer and ("error:" in answer.lower() or "couldn't find" in answer.lower()):
                        # This is likely an error message from our improved error handling
                        response_placeholder.warning(answer)
                        
                        # Give helpful tips based on the error
                        answer_lower = answer.lower() if answer else ""
                        if "context" in answer_lower or "relevant" in answer_lower:
                            st.info("📚 Tip: Try uploading more relevant documents or rephrasing your question.")
                        elif "token" in answer_lower or "too large" in answer_lower:
                            st.info("📏 Tip: Your documents may be too large. Try uploading smaller documents or asking about specific sections.")
                    else:
                        # Regular answer - display it nicely
                        response_placeholder.write(answer)
                        
                        # Display sources if available
                        if sources:
                            with sources_placeholder.expander("📄 Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"{i}. **{source['source']}**")
                        else:
                            sources_placeholder.info("No specific sources were used for this answer.")
                        
                        # Store the result in chat history
                        # Ensure the answer is a valid string before creating the AIMessage
                        if answer is not None:
                            try:
                                ai_message = AIMessage(content=str(answer))
                                if hasattr(ai_message, "__setattr__"):
                                    ai_message.sources = sources  # Attach sources to the message
                                st.session_state.messages.append(ai_message)
                            except Exception as msg_err:
                                st.error(f"Error storing message in history: {str(msg_err)}")
                
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    st.error("⚠️ Something went wrong while processing your question.")
                    
                    with st.expander("Technical details for troubleshooting"):
                        st.code(f"Error: {str(e)}\n\n{tb}")


if __name__ == "__main__":
    # Ensure the vector store directory exists
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    
    # Run the Streamlit app
    main()
