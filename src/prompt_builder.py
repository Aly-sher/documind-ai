"""
Prompt construction for DocuMind AI.

Provides:
  - get_contextualise_prompt()          History-aware retriever prompt
  - get_qa_prompt(complexity, language) QA chain prompt with tuner + language injection
  - format_retrieved_chunks(docs)       Source chunk formatter for the UI expander

Phase 1 additions:
  - Complexity Tuner: Simple / Balanced / Expert depth instructions
  - Multilingual injection: Urdu (Nastaliq), Roman Urdu, Arabic
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# Complexity tuner
# ---------------------------------------------------------------------------

COMPLEXITY_LEVELS: List[str] = ["Simple", "Balanced", "Expert"]

_COMPLEXITY_INSTRUCTIONS: dict[str, str] = {
    "Simple": (
        "RESPONSE DEPTH — SIMPLE:\n"
        "Explain your answer as if the reader has no prior knowledge of the topic. "
        "Use short sentences, everyday language, and concrete examples. "
        "Avoid all jargon. If a technical term is unavoidable, immediately define it "
        "in plain language."
    ),
    "Balanced": (
        "RESPONSE DEPTH — BALANCED:\n"
        "Explain your answer at a clear undergraduate level. "
        "Balance technical accuracy with accessibility. "
        "You may use domain terms but briefly define anything that is not common knowledge."
    ),
    "Expert": (
        "RESPONSE DEPTH — EXPERT:\n"
        "Explain your answer at a professional expert level. "
        "Use precise technical terminology without simplification. "
        "Assume the reader has advanced domain knowledge. "
        "Be concise and technically rigorous."
    ),
}

# ---------------------------------------------------------------------------
# Language injection
# ---------------------------------------------------------------------------

_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ur": (
        "\n\nCRITICAL LANGUAGE RULE: The user wrote in Urdu (Nastaliq script). "
        "Write your ENTIRE response in Urdu script — do not switch to English at any point. "
        "Page citation format: صفحہ [number]"
    ),
    "roman_ur": (
        "\n\nCRITICAL LANGUAGE RULE: The user wrote in Roman Urdu (Urdu in Latin letters). "
        "Write your ENTIRE response in Roman Urdu. "
        "Example style: 'Is document mein yeh likha gaya hai ke...'. "
        "Page citation format: 'Page [number] par'"
    ),
    "ar": (
        "\n\nCRITICAL LANGUAGE RULE: The user wrote in Arabic. "
        "Write your ENTIRE response in Arabic. "
        "Page citation format: صفحة [number]"
    ),
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def get_contextualise_prompt() -> ChatPromptTemplate:
    """
    History-aware retriever prompt.
    Reformulates follow-up questions into standalone queries — does NOT answer.
    """
    system = (
        "Given the conversation history and the latest user question "
        "(which may reference prior context), formulate a standalone question "
        "that can be understood without the history. "
        "Do NOT answer the question — only reformulate it if necessary. "
        "If it is already standalone, return it unchanged."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def get_qa_prompt(
    complexity_level: str = "Balanced",
    language_code: str = "en",
) -> ChatPromptTemplate:
    """
    Build the QA system prompt with complexity tuning and language injection.

    Args:
        complexity_level: "Simple" | "Balanced" | "Expert"
        language_code:    ISO code from language_detector, e.g. "en", "ur", "roman_ur"

    Returns:
        ChatPromptTemplate for use with create_stuff_documents_chain.
    """
    depth_instruction    = _COMPLEXITY_INSTRUCTIONS.get(
        complexity_level, _COMPLEXITY_INSTRUCTIONS["Balanced"]
    )
    language_instruction = _LANGUAGE_INSTRUCTIONS.get(language_code, "")

    system = (
        "You are DocuMind AI, an expert document analyst. "
        "Your sole task is to answer questions STRICTLY based on the provided document context.\n\n"

        "RULES:\n"
        "1. Only use information from the CONTEXT section below.\n"
        "2. If the answer is not in the context, say exactly:\n"
        "   'I could not find this information in the document.'\n"
        "3. Always cite the page number(s) where you found the answer, "
        "   e.g. [Page 4] or [Pages 4, 7].\n"
        "4. Do NOT add information from your training data.\n"
        "5. If the question is ambiguous, ask the user for clarification.\n\n"

        f"{depth_instruction}"
        f"{language_instruction}\n\n"

        "CONTEXT:\n{context}"
    )

    return ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


# ---------------------------------------------------------------------------
# Source chunk formatter
# ---------------------------------------------------------------------------

def format_retrieved_chunks(source_docs: List[Any]) -> str:
    """
    Format retrieved LangChain Document objects for the Streamlit source expander.

    Args:
        source_docs: Documents from rag_chain result["context"]

    Returns:
        Human-readable string for display inside st.expander.
    """
    if not source_docs:
        return "No source chunks were retrieved for this query."

    lines: List[str] = []
    seen:  set[str]  = set()

    for i, doc in enumerate(source_docs, start=1):
        # Normalise across LangChain Document objects and plain dicts
        if hasattr(doc, "page_content"):
            content  = doc.page_content
            metadata = getattr(doc, "metadata", {})
        elif isinstance(doc, dict):
            content  = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
        else:
            content  = str(doc)
            metadata = {}

        # Deduplicate identical chunks
        fingerprint = content[:80]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        page   = metadata.get("page", metadata.get("page_number", "?"))
        source = metadata.get("source", metadata.get("filename", "document"))

        snippet = content.strip()[:420]
        if len(content.strip()) > 420:
            snippet += "…"

        lines.append(f"[Chunk {i}]  Page {page}  |  {source}")
        lines.append(snippet)
        lines.append("")  # blank separator

    return "\n".join(lines)
