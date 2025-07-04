"""Teacher-centric Streamlit UI for educational document analysis.

This module provides a web interface designed specifically for teachers to manage
student documents, track progress, and gain educational insights through AI-powered
conversations about individual students.
"""

import os
import tempfile
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage

from src.config import settings
from src.embeddings import (
    create_faiss_index, 
    student_index_exists, 
    load_student_index, 
    save_student_index,
    get_student_document_count,
    list_student_indexes
)
from src.ingestion import load_txt, load_pdf, load_docx
from src.enhanced_ingestion import EnhancedDocumentProcessor
from src.langgraph_chain import build_retrieval_chain
from src.student_manager import student_manager, Student


# Document type categories for educational context
DOCUMENT_TYPES = {
    "📋 Report Cards": "report_card",
    "📝 Assessment Results": "assessment",
    "🎨 Work Samples": "work_sample",
    "👀 Observations": "observation",
    "📚 Reading Logs": "reading_log",
    "🏆 Achievements": "achievement",
    "⚠️ Intervention Notes": "intervention",
    "📞 Parent Communications": "parent_comm"
}

# Pre-built educational query templates
EDUCATIONAL_QUERIES = {
    "📊 Progress Tracking": [
        "How is {student_name} progressing in mathematics compared to last month?",
        "What improvements has {student_name} shown in reading comprehension?",
        "Show me {student_name}'s growth in writing skills over time.",
        "How has {student_name}'s behavior improved this semester?"
    ],
    "🎯 Learning Gaps": [
        "What are {student_name}'s main learning gaps in mathematics?",
        "Which reading skills does {student_name} need to develop further?",
        "What concepts is {student_name} struggling with in science?",
        "Where should I focus intervention efforts for {student_name}?"
    ],
    "💪 Strengths & Areas": [
        "What are {student_name}'s strongest academic skills?",
        "In which subjects does {student_name} excel?",
        "What learning strategies work best for {student_name}?",
        "What are {student_name}'s preferred learning styles?"
    ],
    "📈 Performance Analysis": [
        "Compare {student_name}'s performance across different subjects.",
        "What patterns do you see in {student_name}'s assessment results?",
        "How consistent is {student_name}'s academic performance?",
        "What factors might be affecting {student_name}'s learning?"
    ]
}


def initialize_session_state():
    """Initialize Streamlit session state variables for teacher interface."""
    # Student management
    if "current_student_id" not in st.session_state:
        st.session_state.current_student_id = None
    
    if "students_cache" not in st.session_state:
        st.session_state.students_cache = {}
        
    # UI state
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
        
    # Chat state per student
    if "student_messages" not in st.session_state:
        st.session_state.student_messages = {}
        
    if "student_retrieval_chains" not in st.session_state:
        st.session_state.student_retrieval_chains = {}
        
    # Document upload state
    if "upload_progress" not in st.session_state:
        st.session_state.upload_progress = {}


def get_student_metrics(student_id: str) -> Dict:
    """Get comprehensive metrics for a student."""
    doc_count = 0
    last_activity = "No activity"
    has_index = False
    
    if student_index_exists(student_id):
        has_index = True
        doc_count = get_student_document_count(student_id)
    
    # Get last chat activity from session state
    if student_id in st.session_state.student_messages:
        messages = st.session_state.student_messages[student_id]
        if messages:
            last_activity = "Recent chat"
    
    return {
        "document_count": doc_count,
        "has_index": has_index,
        "last_activity": last_activity,
        "chat_count": len(st.session_state.student_messages.get(student_id, [])) // 2  # Pairs of user/ai
    }


def render_student_card(student: Student, metrics: Dict):
    """Render a student card with key information and metrics."""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### 👤 {student.name}")
            st.write(f"**ID:** {student.id[:20]}...")
            st.write(f"**Created:** {student.created_at.split('T')[0]}")
            
            # Metrics
            col_docs, col_chats = st.columns(2)
            with col_docs:
                st.metric("📄 Documents", metrics["document_count"])
            with col_chats:
                st.metric("💬 Conversations", metrics["chat_count"])
        
        with col2:
            # Progress indicator
            if metrics["document_count"] == 0:
                st.warning("⚠️ No documents")
            elif metrics["document_count"] < 5:
                st.info("📈 Getting started")
            else:
                st.success("📈 Good data")
            
            # Action buttons
            if st.button(f"Chat", key=f"chat_{student.id}", use_container_width=True):
                st.session_state.current_student_id = student.id
                st.session_state.page = "student_view"
                st.rerun()
                
            if st.button(f"📄 Docs", key=f"docs_{student.id}", use_container_width=True):
                st.session_state.current_student_id = student.id
                st.session_state.page = "document_management"
                st.rerun()


def render_dashboard():
    """Render the main teacher dashboard."""
    st.title("🏫 My Classroom Dashboard")
    
    # Load all students
    students = student_manager.list_students()
    
    # Class overview statistics
    st.markdown("### 📊 Class Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_docs = sum(get_student_metrics(s.id)["document_count"] for s in students)
    total_chats = sum(get_student_metrics(s.id)["chat_count"] for s in students)
    
    with col1:
        st.metric("👥 Students", len(students))
    with col2:
        st.metric("📄 Total Documents", total_docs)
    with col3:
        st.metric("💬 Total Conversations", total_chats)
    with col4:
        students_with_docs = sum(1 for s in students if get_student_metrics(s.id)["document_count"] > 0)
        st.metric("📈 Students with Data", f"{students_with_docs}/{len(students)}")
    
    st.divider()
    
    # Student search and filtering
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("🔍 Search students", placeholder="Enter student name...")
    with col2:
        if st.button("➕ Add New Student", type="primary", key="add_student_dashboard"):
            st.session_state.page = "add_student"
            st.rerun()
    
    # Filter students based on search
    filtered_students = students
    if search_term:
        filtered_students = [s for s in students if search_term.lower() in s.name.lower()]
    
    # Student cards grid
    st.markdown("### 📚 Students")
    
    if not filtered_students:
        if search_term:
            st.info(f"No students found matching '{search_term}'")
        else:
            st.info("No students yet. Add your first student to get started!")
        return
    
    # Display students in a grid (3 per row)
    for i in range(0, len(filtered_students), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(filtered_students):
                student = filtered_students[i + j]
                metrics = get_student_metrics(student.id)
                with col:
                    with st.container():
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0;">
                            <h4>👤 {student.name}</h4>
                            <p><strong>Documents:</strong> {metrics['document_count']}</p>
                            <p><strong>Conversations:</strong> {metrics['chat_count']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_chat, col_docs = st.columns(2)
                        with col_chat:
                            if st.button("💬 Chat", key=f"chat_card_{student.id}"):
                                st.session_state.current_student_id = student.id
                                st.session_state.page = "student_view"
                                st.rerun()
                        with col_docs:
                            if st.button("📄 Docs", key=f"docs_card_{student.id}"):
                                st.session_state.current_student_id = student.id
                                st.session_state.page = "document_management"
                                st.rerun()


def render_add_student():
    """Render the add student interface."""
    st.title("➕ Add New Student")
    
    with st.form("add_student_form"):
        student_name = st.text_input("Student Name", placeholder="Enter student's full name")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Create Student", type="primary"):
                if student_name.strip():
                    try:
                        student = student_manager.create_student(student_name.strip())
                        st.success(f"✅ Created student: {student.name}")
                        st.success(f"Student ID: {student.id}")
                        st.session_state.current_student_id = student.id
                        # Refresh students cache
                        st.session_state.students_cache = {}
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error creating student: {str(e)}")
                else:
                    st.error("Please enter a student name")
        
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state.page = "dashboard"
                st.rerun()


def render_educational_queries(student: Student):
    """Render pre-built educational query interface."""
    st.markdown("### 🎯 Educational Queries")
    st.info("💡 Click any question below to get instant insights about " + student.name + "!")
    
    # Check if student has documents
    if not student_index_exists(student.id):
        st.warning(f"Upload documents for {student.name} first to use educational queries.")
        return
    
    # Create tabs for different query categories
    tabs = st.tabs(list(EDUCATIONAL_QUERIES.keys()))
    
    for i, (category, queries) in enumerate(EDUCATIONAL_QUERIES.items()):
        with tabs[i]:
            st.markdown(f"**{category}**")
            
            for query_template in queries:
                query = query_template.format(student_name=student.name)
                if st.button(query, key=f"query_{category}_{hash(query)}", use_container_width=True):
                    # Process the query immediately in this tab
                    st.markdown("---")
                    
                    # Add to chat history
                    if student.id not in st.session_state.student_messages:
                        st.session_state.student_messages[student.id] = []
                    
                    st.session_state.student_messages[student.id].append(HumanMessage(content=query))
                    
                    # Show the query being processed
                    with st.chat_message("user"):
                        st.write(query)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing documents..."):
                            try:
                                # Initialize or load retrieval chain for this student
                                if student.id not in st.session_state.student_retrieval_chains:
                                    try:
                                        index = load_student_index(student.id)
                                        chain = build_retrieval_chain(index, student_id=student.id)
                                        st.session_state.student_retrieval_chains[student.id] = chain
                                    except Exception as e:
                                        st.error(f"Error loading documents for {student.name}: {str(e)}")
                                        return
                                
                                chain = st.session_state.student_retrieval_chains[student.id]
                                result = chain.invoke({"question": query})
                                
                                # Extract answer and sources
                                answer = None
                                sources = []
                                
                                if isinstance(result, dict):
                                    if "answer" in result and result["answer"]:
                                        answer = result["answer"]
                                        sources = result.get("sources", [])
                                    elif "messages" in result and result["messages"]:
                                        for message in result["messages"]:
                                            if isinstance(message, AIMessage) and message == result["messages"][-1]:
                                                answer = message.content
                                                if hasattr(message, "sources"):
                                                    sources = message.sources
                                                break
                                
                                if answer:
                                    st.write(answer)
                                    
                                    # Display sources
                                    if sources:
                                        with st.expander("📄 Sources"):
                                            for i, source in enumerate(sources, 1):
                                                st.write(f"{i}. **{source.get('source', 'Unknown')}**")
                                    
                                    # Add to chat history
                                    ai_message = AIMessage(content=answer)
                                    if sources:
                                        ai_message.sources = sources
                                    st.session_state.student_messages[student.id].append(ai_message)
                                    
                                    # Show helpful message
                                    st.success("💬 This conversation has been saved to the Chat tab for future reference!")
                                else:
                                    st.error("Could not generate a response. Please try rephrasing your question.")
                            
                            except Exception as e:
                                st.error(f"Error processing query: {str(e)}")
                    
                    st.rerun()


def process_uploaded_file(uploaded_file) -> Tuple[str, str]:
    """Process an uploaded file and return its content."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)
    
    try:
        if file_extension == "txt":
            content = load_txt(tmp_path)
        elif file_extension == "pdf":
            content = load_pdf(tmp_path)
        elif file_extension in ["docx", "doc"]:
            content = load_docx(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return uploaded_file.name, content
    finally:
        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)


def render_document_management(student: Student):
    """Render document management interface for a student."""
    st.markdown(f"### 📄 Document Management - {student.name}")
    
    # Document upload section
    st.markdown("#### Upload Documents")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_files = st.file_uploader(
            f"Upload documents for {student.name}",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
            help="Upload report cards, assessments, work samples, or other educational documents"
        )
    
    with col2:
        document_type = st.selectbox(
            "Document Type",
            options=list(DOCUMENT_TYPES.keys()),
            help="Categorize the document for better organization"
        )
    
    if uploaded_files:
        st.success(f"Selected {len(uploaded_files)} file(s) for {student.name}")
        
        if st.button("📤 Upload Documents", type="primary", key=f"upload_docs_{student.id}"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            documents = []
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i) / len(uploaded_files))
                
                try:
                    doc_name, content = process_uploaded_file(file)
                    documents.append((doc_name, content))
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            if documents:
                status_text.text("Creating embeddings...")
                progress_bar.progress(0.8)
                
                # Process documents with enhanced processing if enabled
                if settings.use_enhanced_processing:
                    # Use enhanced processing for better educational document handling
                    processor = EnhancedDocumentProcessor(
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        extract_tables=settings.extract_tables,
                        extract_images=settings.extract_images,
                        preserve_formatting=settings.preserve_formatting,
                        use_ocr=settings.use_ocr,
                    )
                    
                    all_docs = []
                    for doc_name, content in documents:
                        # Create a temporary file for the processor
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
                            tmp.write(content)
                            tmp_path = Path(tmp.name)
                        
                        try:
                            docs = processor.process_document(tmp_path)
                            # Add document type metadata
                            for doc in docs:
                                doc.metadata.update({
                                    "student_id": student.id,
                                    "student_name": student.name,
                                    "document_type": DOCUMENT_TYPES[document_type],
                                    "upload_date": datetime.now().isoformat(),
                                    "original_filename": doc_name
                                })
                            all_docs.extend(docs)
                        finally:
                            tmp_path.unlink(missing_ok=True)
                    
                    chunks = [doc.page_content for doc in all_docs]
                    metadatas = [doc.metadata for doc in all_docs]
                else:
                    # Fallback to basic processing
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        length_function=len,
                    )
                    
                    chunks = []
                    metadatas = []
                    for doc_name, content in documents:
                        doc_chunks = text_splitter.split_text(content)
                        for chunk in doc_chunks:
                            chunks.append(chunk)
                            metadatas.append({
                                "source": doc_name,
                                "student_id": student.id,
                                "student_name": student.name,
                                "document_type": DOCUMENT_TYPES[document_type],
                                "upload_date": datetime.now().isoformat()
                            })
                
                # Create or update the student's index
                index = create_faiss_index(chunks, metadatas=metadatas)
                save_student_index(index, student.id)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Upload complete!")
                
                # Clear retrieval chain cache for this student
                if student.id in st.session_state.student_retrieval_chains:
                    del st.session_state.student_retrieval_chains[student.id]
                
                st.success(f"Successfully uploaded {len(documents)} documents for {student.name}")
                st.rerun()
    
    # Document statistics
    if student_index_exists(student.id):
        doc_count = get_student_document_count(student.id)
        st.info(f"📊 {student.name} currently has {doc_count} documents in their knowledge base")
    else:
        st.warning(f"No documents uploaded for {student.name} yet")


def render_student_chat(student: Student):
    """Render chat interface for a specific student."""
    st.markdown(f"### 💬 Chat with {student.name}'s Documents")
    
    # Check if student has documents
    if not student_index_exists(student.id):
        st.warning(f"No documents available for {student.name}. Please upload documents first.")
        if st.button("📤 Upload Documents", key=f"upload_prompt_{student.id}"):
            st.session_state.page = "document_management"
            st.rerun()
        return
    
    # Initialize or load retrieval chain for this student
    if student.id not in st.session_state.student_retrieval_chains:
        try:
            index = load_student_index(student.id)
            chain = build_retrieval_chain(index, student_id=student.id)
            st.session_state.student_retrieval_chains[student.id] = chain
        except Exception as e:
            st.error(f"Error loading documents for {student.name}: {str(e)}")
            return
    
    # Initialize messages for this student
    if student.id not in st.session_state.student_messages:
        st.session_state.student_messages[student.id] = []
    
    # Display chat history
    messages = st.session_state.student_messages[student.id]
    
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
                # Display sources if available
                if hasattr(message, 'sources') and message.sources:
                    with st.expander("📄 Sources"):
                        for i, source in enumerate(message.sources, 1):
                            st.write(f"{i}. **{source.get('source', 'Unknown')}**")
    
    # Note: Educational queries are now processed directly in the Quick Queries tab
    # Chat input
    if prompt := st.chat_input(f"Ask about {student.name}..."):
        # Add user message
        user_message = HumanMessage(content=prompt)
        st.session_state.student_messages[student.id].append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chain = st.session_state.student_retrieval_chains[student.id]
                    result = chain.invoke({"question": prompt})
                    
                    # Extract answer and sources
                    answer = None
                    sources = []
                    
                    if isinstance(result, dict):
                        if "answer" in result and result["answer"]:
                            answer = result["answer"]
                            sources = result.get("sources", [])
                        elif "messages" in result and result["messages"]:
                            for message in result["messages"]:
                                if isinstance(message, AIMessage) and message == result["messages"][-1]:
                                    answer = message.content
                                    if hasattr(message, "sources"):
                                        sources = message.sources
                                    break
                    
                    if answer:
                        st.write(answer)
                        
                        # Display sources
                        if sources:
                            with st.expander("📄 Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"{i}. **{source.get('source', 'Unknown')}**")
                        
                        # Add to chat history
                        ai_message = AIMessage(content=answer)
                        if sources:
                            ai_message.sources = sources
                        st.session_state.student_messages[student.id].append(ai_message)
                    else:
                        st.error("Could not generate a response. Please try rephrasing your question.")
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                
        st.rerun()


def render_student_view(student: Student):
    """Render the complete student view with all tabs."""
    # Student header with context
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"👤 {student.name}")
        metrics = get_student_metrics(student.id)
        st.write(f"📄 {metrics['document_count']} documents • 💬 {metrics['chat_count']} conversations")
    
    with col2:
        if st.button("🔄 Switch Student", key="switch_student_main"):
            st.session_state.page = "dashboard"
            st.session_state.current_student_id = None
            st.rerun()
    
    # Student navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Documents", "🎯 Quick Queries", "📊 Insights"])
    
    with tab1:
        render_student_chat(student)
    
    with tab2:
        render_document_management(student)
    
    with tab3:
        render_educational_queries(student)
    
    with tab4:
        render_student_insights(student)


def render_student_insights(student: Student):
    """Render insights dashboard for a student."""
    st.markdown(f"### 📊 Insights for {student.name}")
    
    if not student_index_exists(student.id):
        st.warning("Upload documents to see insights")
        return
    
    metrics = get_student_metrics(student.id)
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📄 Documents", metrics["document_count"])
    with col2:
        st.metric("💬 Conversations", metrics["chat_count"])
    with col3:
        # Simple progress indicator based on document count
        if metrics["document_count"] >= 10:
            progress_level = "Excellent"
            progress_color = "🟢"
        elif metrics["document_count"] >= 5:
            progress_level = "Good"
            progress_color = "🟡"
        else:
            progress_level = "Getting Started"
            progress_color = "🔵"
        
        st.metric("📈 Data Status", f"{progress_color} {progress_level}")
    
    # Document timeline (if we had dates, this would be more meaningful)
    st.markdown("#### 📅 Document Upload Timeline")
    st.info("Timeline visualization will show document upload patterns over time")
    
    # Conversation summary
    if student.id in st.session_state.student_messages:
        messages = st.session_state.student_messages[student.id]
        if messages:
            st.markdown("#### 💬 Recent Conversations")
            st.write(f"Total messages: {len(messages)}")
            
            # Show recent topics (last few user questions)
            user_messages = [msg.content for msg in messages if isinstance(msg, HumanMessage)]
            if user_messages:
                st.write("Recent topics:")
                for msg in user_messages[-3:]:  # Last 3 questions
                    st.write(f"• {msg[:100]}...")
    
    # Export options
    st.markdown("#### 📤 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Export Conversation Summary", key=f"export_conv_{student.id}"):
            if student.id in st.session_state.student_messages:
                messages = st.session_state.student_messages[student.id]
                summary = f"Conversation Summary for {student.name}\n"
                summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        summary += f"Question: {msg.content}\n"
                    elif isinstance(msg, AIMessage):
                        summary += f"Answer: {msg.content}\n\n"
                
                st.download_button(
                    "Download Summary",
                    data=summary,
                    file_name=f"{student.name}_conversation_summary.txt",
                    mime="text/plain"
                )
    
    with col2:
        if st.button("📊 Export Student Report", key=f"export_report_{student.id}"):
            report = f"Student Report: {student.name}\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            report += f"Documents: {metrics['document_count']}\n"
            report += f"Conversations: {metrics['chat_count']}\n"
            report += f"Created: {student.created_at}\n"
            
            st.download_button(
                "Download Report",
                data=report,
                file_name=f"{student.name}_student_report.txt",
                mime="text/plain"
            )


def render_navigation():
    """Render the navigation sidebar."""
    with st.sidebar:
        st.markdown("## 🏫 Navigation")
        
        # Current student indicator
        if st.session_state.current_student_id:
            student = student_manager.get_student(st.session_state.current_student_id)
            if student:
                st.success(f"Current: {student.name}")
                if st.button("🔄 Switch Student", key="switch_student_sidebar"):
                    st.session_state.page = "dashboard"
                    st.session_state.current_student_id = None
                    st.rerun()
        
        st.divider()
        
        # Main navigation
        if st.button("🏠 Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.page = "dashboard"
            st.rerun()
        
        if st.button("➕ Add Student", use_container_width=True, key="nav_add_student"):
            st.session_state.page = "add_student"
            st.rerun()
        
        # Quick stats
        students = student_manager.list_students()
        if students:
            st.divider()
            st.markdown("### 📊 Quick Stats")
            st.write(f"👥 Total Students: {len(students)}")
            
            total_docs = sum(get_student_metrics(s.id)["document_count"] for s in students)
            st.write(f"📄 Total Documents: {total_docs}")
            
            students_with_docs = sum(1 for s in students if get_student_metrics(s.id)["document_count"] > 0)
            st.write(f"📈 Students with Data: {students_with_docs}/{len(students)}")


def main():
    """Main Streamlit application for teachers."""
    st.set_page_config(
        page_title="Educational Insight Platform",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Render navigation
    render_navigation()
    
    # Main content area
    if st.session_state.page == "dashboard":
        render_dashboard()
    
    elif st.session_state.page == "add_student":
        render_add_student()
    
    elif st.session_state.page == "student_view":
        if st.session_state.current_student_id:
            student = student_manager.get_student(st.session_state.current_student_id)
            if student:
                render_student_view(student)
            else:
                st.error("Student not found")
                st.session_state.page = "dashboard"
                st.rerun()
        else:
            st.session_state.page = "dashboard"
            st.rerun()
    
    elif st.session_state.page == "document_management":
        if st.session_state.current_student_id:
            student = student_manager.get_student(st.session_state.current_student_id)
            if student:
                render_document_management(student)
                if st.button("← Back to Student View", key=f"back_to_student_{student.id}"):
                    st.session_state.page = "student_view"
                    st.rerun()
            else:
                st.error("Student not found")
                st.session_state.page = "dashboard"
                st.rerun()
        else:
            st.session_state.page = "dashboard"
            st.rerun()


if __name__ == "__main__":
    # Ensure necessary directories exist
    settings.vector_store_path.mkdir(parents=True, exist_ok=True)
    (settings.vector_store_path.parent / "students").mkdir(parents=True, exist_ok=True)
    
    # Run the Streamlit app
    main()