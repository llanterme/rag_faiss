# Enhanced Prompt System

The RAG system now includes an advanced prompt engineering system that significantly improves response quality, accuracy, and appropriateness for different use cases.

## Overview

The enhanced prompt system replaces the basic prompt with sophisticated, context-aware prompts that:

- Provide clear instructions for staying within the document context
- Include source citation requirements
- Handle uncertainty and missing information properly
- Adapt to different question types automatically
- Offer specialized styles for different domains

## Prompt Styles

### 1. Default Style
**Best for**: General purpose questions, balanced responses

```
You are a helpful AI assistant that answers questions based on the provided context. Follow these guidelines:

1. Base your answer ONLY on the information provided in the context
2. If the context doesn't contain enough information, say so clearly
3. Be accurate and factual
4. Cite which document your information comes from when possible
5. If you're uncertain, express that uncertainty
6. Keep answers clear and well-structured
```

### 2. Detailed Style
**Best for**: Research queries, comprehensive analysis

```
You are a comprehensive research assistant. When answering questions:

1. Provide thorough, detailed responses based on the context
2. Structure your answer with clear sections if appropriate
3. Include all relevant information from the provided documents
4. Cite specific sources and quote directly when helpful
5. Explain technical terms and provide background when needed
6. If information is incomplete, explicitly state what's missing
7. Use bullet points or numbered lists for clarity
```

### 3. Concise Style
**Best for**: Quick facts, brief answers, mobile usage

```
You are a concise AI assistant. Follow these rules:

1. Give brief, direct answers based on the context
2. Get straight to the point without unnecessary elaboration
3. Use bullet points for multiple items
4. Only include the most essential information
5. If context lacks info, say "Not found in provided documents"
6. Aim for answers under 100 words when possible
```

### 4. Academic Style
**Best for**: Research papers, scholarly analysis, citations

```
You are an academic research assistant with expertise in scholarly analysis:

1. Provide well-researched answers with academic rigor
2. Cite sources in a format like: "According to [Document Name]..."
3. Distinguish between facts, interpretations, and inferences
4. Use formal academic language
5. Acknowledge limitations and gaps in the available information
6. Consider multiple perspectives when relevant
7. Structure responses with clear thesis and supporting evidence
```

### 5. Technical Style
**Best for**: API documentation, configuration guides, technical manuals

```
You are a technical documentation assistant specialized in precise technical communication:

1. Provide technically accurate answers based on the documentation
2. Use proper technical terminology from the context
3. Include code examples, configurations, or commands when present
4. Structure answers with clear steps for procedures
5. Highlight warnings, prerequisites, or important notes
6. If technical details are missing, specify what information is needed
7. Distinguish between required and optional elements
```

## Smart Question Detection

The system automatically detects question types and adds appropriate hints:

| Question Pattern | Auto-Added Hint |
|-----------------|-----------------|
| "How many...", "What number..." | "The user is asking for a specific number or quantity." |
| "List...", "What are..." | "The user is asking for a list. Use bullet points or numbers." |
| "Explain...", "How does..." | "The user wants an explanation. Be thorough but clear." |
| "Compare...", "Difference..." | "The user wants a comparison. Structure to highlight differences." |
| "Tell me...", "Show me..." | "Provide a comprehensive overview of the requested topic." |

## Configuration

### Environment Variables

```env
# Prompt configuration
PROMPT_STYLE=default  # default, detailed, concise, academic, technical
LOGFIRE_LOG_PROMPTS=true  # Show prompts in logs
```

### CLI Commands

Test different prompt styles:
```bash
# Test a specific style
poetry run python -m src.cli test-prompts --style academic

# View current prompt configuration
poetry run python -m src.cli observability
```

### Runtime Usage

The enhanced prompts are used automatically in:
- CLI chat sessions
- Streamlit web interface
- Programmatic API calls

## Key Improvements vs Basic Prompts

### Before (Basic Prompt)
```
You are a helpful AI assistant answering questions based on the provided context.

Context: [documents]

Question: [user question]
```

### After (Enhanced Prompt)
```
You are a helpful AI assistant that answers questions based on the provided context. Follow these guidelines:

1. Base your answer ONLY on the information provided in the context
2. If the context doesn't contain enough information, say so clearly
3. Be accurate and factual
4. Cite which document your information comes from when possible
5. If you're uncertain, express that uncertainty
6. Keep answers clear and well-structured

Provided Context:
[Formatted documents with clear source attribution]

Note: The user is asking for a list. Use bullet points or numbers.

User Question: [question]

Remember: Base your answer only on the provided context. If the answer isn't in the context, say so.
```

## Expected Improvements

### Response Quality
- ✅ **Better accuracy**: Clear context boundaries reduce hallucination
- ✅ **Source attribution**: Users know where information comes from
- ✅ **Uncertainty handling**: AI admits when it doesn't know
- ✅ **Appropriate length**: Responses match the query type

### Domain Adaptation
- ✅ **Academic rigor**: Proper citations and formal language
- ✅ **Technical precision**: Step-by-step procedures and warnings
- ✅ **Concise summaries**: Brief, focused answers when needed
- ✅ **Comprehensive analysis**: Detailed exploration when required

### User Experience
- ✅ **Consistent formatting**: Structured responses with bullet points
- ✅ **Clear limitations**: Honest about missing information
- ✅ **Context awareness**: Responses match the document domain
- ✅ **Error reduction**: Fewer incorrect or irrelevant answers

## Best Practices

### Choosing the Right Style

| Use Case | Recommended Style | Example |
|----------|------------------|---------|
| Customer support | Default | General product questions |
| Research analysis | Detailed | Academic paper analysis |
| Mobile app | Concise | Quick fact lookups |
| Scientific papers | Academic | Research methodology questions |
| Developer docs | Technical | API configuration guides |

### Custom Instructions

For specialized domains, you can add custom instructions:

```python
from src.prompts import create_enhanced_prompt

prompt = create_enhanced_prompt(
    context=context,
    question=question,
    style=PromptStyle.DEFAULT,
    custom_instructions="Always mention regulatory compliance when discussing financial topics."
)
```

### Testing and Optimization

1. **Test different styles** with your specific documents
2. **Monitor response quality** using Logfire observability
3. **Compare styles** using the CLI test command
4. **Gather user feedback** on response appropriateness

## Advanced Features

### Few-Shot Examples

The system can include examples for better performance:

```python
from src.prompts import add_few_shot_examples

enhanced_prompt = add_few_shot_examples(
    prompt=base_prompt,
    task_type="factual_qa"
)
```

### Source Formatting

Retrieved documents are formatted with clear attribution:

```
[Document 1: Company Policy Manual, Page 3]
Employees are entitled to 15 days of paid vacation per year...

---

[Document 2: Benefits Guide, Page 1]  
Health insurance covers employee and family members...
```

### Conversation Context

The system maintains conversation history for multi-turn interactions:

```
Previous Conversation:
Q: What is the vacation policy?
A: According to the Company Policy Manual, employees get 15 days...

Provided Context: [current documents]

User Question: How do I request vacation?
```

## Troubleshooting

### Common Issues

1. **Responses too long/short**
   - Solution: Try "concise" or "detailed" style

2. **Missing source citations**
   - Check if documents have proper metadata
   - Verify source information is being passed

3. **Too formal/informal language**
   - Switch between "academic" and "default" styles
   - Add custom instructions for tone

4. **Technical responses unclear**
   - Use "technical" style for docs
   - Use "default" for general audiences

### Debug Prompts

View the exact prompt being sent:
```bash
# Enable prompt logging
export LOGFIRE_LOG_PROMPTS=true

# Run query and check logs
poetry run python -m src.cli chat
```

## Migration Guide

### From Basic to Enhanced

If upgrading from the basic prompt system:

1. **No breaking changes** - Enhanced prompts work automatically
2. **Better responses** - Expect improved quality immediately  
3. **New configuration** - Optionally set `PROMPT_STYLE` in `.env`
4. **Test styles** - Use CLI to find the best style for your use case

### Custom Prompts

If you have custom prompts, you can integrate them:

```python
from src.prompts import create_enhanced_prompt, PromptStyle

# Your custom instructions
custom_instructions = "Always include cost estimates when discussing services."

# Create enhanced prompt with your additions
prompt = create_enhanced_prompt(
    context=context,
    question=question,
    style=PromptStyle.DEFAULT,
    custom_instructions=custom_instructions
)
```

## Industry Comparison

The enhanced prompt system brings the RAG implementation in line with industry best practices:

- ✅ **Context boundaries**: Prevents hallucination (OpenAI best practices)
- ✅ **Source attribution**: Enables verification (Anthropic guidelines)
- ✅ **Uncertainty handling**: Builds trust (AI safety principles)
- ✅ **Domain adaptation**: Matches user needs (Microsoft Research)
- ✅ **Format consistency**: Improves UX (Google AI recommendations)

This places the system at the forefront of RAG prompt engineering, exceeding many commercial implementations.