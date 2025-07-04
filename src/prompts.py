"""Enhanced prompt templates for the RAG system."""

from typing import Dict, Optional, List
from enum import Enum


class PromptStyle(Enum):
    """Different prompt styles for various use cases."""

    DEFAULT = "default"
    DETAILED = "detailed"
    CONCISE = "concise"
    ACADEMIC = "academic"
    TECHNICAL = "technical"


# Base system prompts for different styles
SYSTEM_PROMPTS = {
    PromptStyle.DEFAULT: """You are a helpful AI assistant that answers questions based on the provided context. Follow these guidelines:

1. Base your answer ONLY on the information provided in the context
2. If the context doesn't contain enough information, say so clearly
3. Be accurate and factual
4. Cite which document your information comes from when possible
5. If you're uncertain, express that uncertainty
6. Keep answers clear and well-structured""",
    PromptStyle.DETAILED: """You are a comprehensive research assistant. When answering questions:

1. Provide thorough, detailed responses based on the context
2. Structure your answer with clear sections if appropriate
3. Include all relevant information from the provided documents
4. Cite specific sources and quote directly when helpful
5. Explain technical terms and provide background when needed
6. If information is incomplete, explicitly state what's missing
7. Use bullet points or numbered lists for clarity""",
    PromptStyle.CONCISE: """You are a concise AI assistant. Follow these rules:

1. Give brief, direct answers based on the context
2. Get straight to the point without unnecessary elaboration
3. Use bullet points for multiple items
4. Only include the most essential information
5. If context lacks info, say "Not found in provided documents"
6. Aim for answers under 100 words when possible""",
    PromptStyle.ACADEMIC: """You are an academic research assistant with expertise in scholarly analysis:

1. Provide well-researched answers with academic rigor
2. Cite sources in a format like: "According to [Document Name]..."
3. Distinguish between facts, interpretations, and inferences
4. Use formal academic language
5. Acknowledge limitations and gaps in the available information
6. Consider multiple perspectives when relevant
7. Structure responses with clear thesis and supporting evidence""",
    PromptStyle.TECHNICAL: """You are a technical documentation assistant specialized in precise technical communication:

1. Provide technically accurate answers based on the documentation
2. Use proper technical terminology from the context
3. Include code examples, configurations, or commands when present
4. Structure answers with clear steps for procedures
5. Highlight warnings, prerequisites, or important notes
6. If technical details are missing, specify what information is needed
7. Distinguish between required and optional elements""",
}


def format_context_with_sources(
    chunks: List[Dict], include_metadata: bool = True
) -> str:
    """Format retrieved chunks with source information.

    Args:
        chunks: List of document chunks with content and metadata
        include_metadata: Whether to include source metadata

    Returns:
        Formatted context string with clear source attribution
    """
    if not chunks:
        return "No relevant documents found."

    formatted_chunks = []

    for i, chunk in enumerate(chunks, 1):
        if include_metadata and isinstance(chunk, dict):
            # Extract metadata
            source = chunk.get("metadata", {}).get("source", "Unknown")
            page = chunk.get("metadata", {}).get("page_number", "")
            content = chunk.get("page_content", str(chunk))

            # Format with clear source marking
            chunk_header = f"[Document {i}: {source}"
            if page:
                chunk_header += f", Page {page}"
            chunk_header += "]"

            formatted_chunk = f"{chunk_header}\n{content}\n"
        else:
            # Simple format
            content = str(chunk)
            formatted_chunk = f"[Document {i}]\n{content}\n"

        formatted_chunks.append(formatted_chunk)

    return "\n---\n".join(formatted_chunks)


def create_enhanced_prompt(
    context: str,
    question: str,
    style: PromptStyle = PromptStyle.DEFAULT,
    custom_instructions: Optional[str] = None,
    conversation_history: Optional[List[Dict]] = None,
) -> str:
    """Create an enhanced prompt with better structure and instructions.

    Args:
        context: The retrieved document context
        question: The user's question
        style: The prompt style to use
        custom_instructions: Additional custom instructions
        conversation_history: Previous Q&A pairs

    Returns:
        Complete formatted prompt
    """
    # Start with the base system prompt for the style
    prompt = SYSTEM_PROMPTS[style]

    # Add custom instructions if provided
    if custom_instructions:
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"

    # Add conversation history context if available
    if conversation_history:
        prompt += "\n\nPrevious Conversation:"
        for entry in conversation_history[-3:]:  # Last 3 exchanges
            if entry.get("question"):
                prompt += f"\nQ: {entry['question']}"
            if entry.get("answer"):
                prompt += f"\nA: {entry['answer'][:200]}..."

    # Add the context section
    prompt += f"\n\nProvided Context:\n{context}"

    # Add response format hints based on question type
    question_lower = question.lower()

    if any(word in question_lower for word in ["how many", "what number", "count"]):
        prompt += "\n\nNote: The user is asking for a specific number or quantity."
    elif any(word in question_lower for word in ["list", "enumerate", "what are"]):
        prompt += (
            "\n\nNote: The user is asking for a list. Use bullet points or numbers."
        )
    elif any(word in question_lower for word in ["explain", "describe", "how does"]):
        prompt += "\n\nNote: The user wants an explanation. Be thorough but clear."
    elif any(word in question_lower for word in ["compare", "difference", "versus"]):
        prompt += "\n\nNote: The user wants a comparison. Structure your answer to highlight differences."
    elif "?" not in question and any(
        word in question_lower for word in ["tell me", "show me"]
    ):
        prompt += "\n\nNote: Provide a comprehensive overview of the requested topic."

    # Add the actual question
    prompt += f"\n\nUser Question: {question}"

    # Add response format reminder
    prompt += "\n\nRemember: Base your answer only on the provided context. If the answer isn't in the context, say so."

    return prompt


def create_rag_prompt(
    context: str,
    question: str,
    sources: Optional[List[Dict]] = None,
    style: Optional[str] = None,
) -> str:
    """Create a RAG prompt with proper formatting and source attribution.

    This is a convenience function that handles the most common RAG use case.

    Args:
        context: Retrieved document chunks as string
        question: User's question
        sources: Optional list of source metadata
        style: Optional style name (default, detailed, concise, academic, technical)

    Returns:
        Formatted system prompt
    """
    # Convert style string to enum
    prompt_style = PromptStyle.DEFAULT
    if style:
        try:
            prompt_style = PromptStyle(style.lower())
        except ValueError:
            # Invalid style, use default
            pass

    # Format context with sources if available
    if sources and context:
        # Format the context with source information
        formatted_context = ""
        for i, source in enumerate(sources, 1):
            source_name = source.get("source", "Unknown")
            formatted_context += f"[Document {i}: {source_name}]\n"
        formatted_context += f"{context}"
        context = formatted_context

    # Create the enhanced prompt
    return create_enhanced_prompt(
        context=context, question=question, style=prompt_style
    )


# Example few-shot prompts for specific tasks
FEW_SHOT_EXAMPLES = {
    "factual_qa": [
        {
            "context": "The Python programming language was created by Guido van Rossum and first released in 1991.",
            "question": "Who created Python?",
            "answer": "According to the provided context, Python was created by Guido van Rossum.",
        },
        {
            "context": "The Python programming language was created by Guido van Rossum and first released in 1991.",
            "question": "When was Java created?",
            "answer": "The provided context doesn't contain information about when Java was created. It only mentions Python, which was first released in 1991.",
        },
    ],
    "summarization": [
        {
            "context": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
            "question": "Summarize this in one sentence",
            "answer": "Machine learning is an AI subset that enables systems to learn from data through supervised, unsupervised, and reinforcement learning approaches.",
        }
    ],
}


def add_few_shot_examples(prompt: str, task_type: str = "factual_qa") -> str:
    """Add few-shot examples to a prompt for better performance.

    Args:
        prompt: The base prompt
        task_type: Type of task (factual_qa, summarization, etc.)

    Returns:
        Prompt with few-shot examples
    """
    if task_type in FEW_SHOT_EXAMPLES:
        examples = FEW_SHOT_EXAMPLES[task_type]
        example_text = "\n\nHere are some examples of good responses:\n"

        for i, example in enumerate(examples, 1):
            example_text += f"\nExample {i}:"
            example_text += f"\nContext: {example['context']}"
            example_text += f"\nQuestion: {example['question']}"
            example_text += f"\nGood Answer: {example['answer']}\n"

        prompt = prompt.replace(
            "Provided Context:", example_text + "\nProvided Context:"
        )

    return prompt
