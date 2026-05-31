"""
src/prompt_builder.py
----------------------
Prompt construction and context formatting for DocuMind AI.

Builds the two prompts required by the LangChain history-aware RAG chain:
  1. Contextualisation prompt — reformulates the user's question in light of
     conversation history into a standalone, self-contained question.
  2. QA prompt — the main DocuMind AI system prompt with anti-hallucination
     rules, citation mandate, and strict context-grounding.

Prompt design follows SRS §6.2 exactly, with all failure modes from §6.3
addressed through prompt structure.
"""

from typing import List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.docstore.document import Document

from utils.logger import logger


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def format_retrieved_chunks(docs: List[Document]) -> str:
    """
    Format a list of retrieved LangChain Document objects into a labelled
    context string for injection into the QA prompt.

    Each chunk is labelled with its source number and page number so that
    Llama 3 can produce accurate page citations in its answer.

    Args:
        docs: List of Document objects returned by the FAISS retriever.
              Each must have .page_content and .metadata["page"].

    Returns:
        Formatted multi-line string. Returns a fallback message if docs is empty.
    """
    if not docs:
        logger.warning("format_retrieved_chunks called with empty docs list.")
        return "No relevant context found in the document."

    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "Unknown")
        # Cap chunk text at 1500 chars to guard against runaway context
        chunk_text = doc.page_content[:1500]
        parts.append(
            f"--- Source {i} (File: {source} | Page: {page_num}) ---\n{chunk_text}"
        )

    formatted = "\n\n".join(parts)
    logger.debug(f"Formatted {len(docs)} retrieved chunk(s) into context string.")
    return formatted


# ---------------------------------------------------------------------------
# Prompt 1 — Contextualisation (history-aware question reformulation)
# ---------------------------------------------------------------------------

def get_contextualise_prompt() -> ChatPromptTemplate:
    """
    Build the prompt that reformulates the user's question as a standalone
    question, independent of conversation history.

    This is the first step in the history-aware RAG chain. It prevents the
    retriever from receiving vague follow-up questions like "tell me more".

    Returns:
        ChatPromptTemplate for the history-aware retriever.
    """
    system_prompt = (
        "Given the conversation history below and the latest user question, "
        "reformulate the question into a self-contained, standalone question "
        "that can be understood without the conversation history. "
        "Do NOT answer the question — only reformulate it if needed. "
        "If the question is already standalone, return it unchanged."
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


# ---------------------------------------------------------------------------
# Prompt 2 — QA (main DocuMind AI system prompt, per SRS §6.2)
# ---------------------------------------------------------------------------

def get_qa_prompt() -> ChatPromptTemplate:
    """
    Build the main QA system prompt for DocuMind AI.

    Implements all rules from SRS §6.2 and mitigates all failure modes
    from SRS §6.3:
      - Anti-hallucination: strict context-only rule
      - Citation mandate: always cite page numbers
      - Context bleeding: history capped at last 3 turns in llm_chain.py
      - Over-refusal: generation anchor ("ANSWER with page citations:")

    Returns:
        ChatPromptTemplate for the question-answer chain.
    """
    system_template = """You are DocuMind AI, an expert document analyst. \
Your job is to answer questions STRICTLY based on the provided document context.

RULES:
1. Only use information from the CONTEXT provided below.
2. If the answer is not in the context, say exactly: \
'I could not find this information in the document.'
3. Always cite the page number(s) where you found the answer, \
e.g. "(Page 4)" or "(Pages 4, 7)".
4. Be precise and concise. Do not add information from your training data.
5. If the question is ambiguous, ask for clarification before answering.

CONTEXT:
{context}

CONVERSATION HISTORY:
(Last 3 turns shown for continuity)

ANSWER with page citations:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


__all__ = ["format_retrieved_chunks", "get_contextualise_prompt", "get_qa_prompt"]