"""
Prompt construction for DocuMind AI.

Provides:
  - get_contextualise_prompt()
  - get_qa_prompt(complexity, language)
  - format_retrieved_chunks(docs)

Phase 1:
  - Complexity Tuner: Simple / Balanced / Expert with hard behavioural rules
  - Multilingual injection: Urdu (Nastaliq), Roman Urdu, Arabic
"""

from __future__ import annotations
from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# Complexity tuner — PRESCRIPTIVE rules, not vague tone descriptors
# ---------------------------------------------------------------------------

COMPLEXITY_LEVELS: List[str] = ["Simple", "Balanced", "Expert"]

_COMPLEXITY_INSTRUCTIONS: dict[str, str] = {
    "Simple": (
        "\n\n━━━ RESPONSE FORMAT — SIMPLE MODE (STRICTLY ENFORCED) ━━━\n"
        "You MUST follow ALL of these rules. No exceptions.\n"
        "1. MAXIMUM 3 short sentences in your entire answer.\n"
        "2. FORBIDDEN words: any technical term, acronym, or jargon. "
        "   If a technical word appears in the document, replace it with an everyday word.\n"
        "3. Write like you are explaining to a 10-year-old child.\n"
        "4. Use ONLY simple words (under 2 syllables where possible).\n"
        "5. Start your answer with: 'In simple terms, ...'\n"
        "6. End with one concrete real-life example if possible.\n"
        "EXAMPLE of correct Simple answer: "
        "'In simple terms, this document is a guide for writing a job application. "
        "It tells you what to include and how to present yourself. "
        "Think of it like instructions for introducing yourself at a job interview. [Page 1]'\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    ),
    "Balanced": (
        "\n\n━━━ RESPONSE FORMAT — BALANCED MODE (STRICTLY ENFORCED) ━━━\n"
        "You MUST follow ALL of these rules. No exceptions.\n"
        "1. Write 2-4 well-structured paragraphs.\n"
        "2. You may use technical terms BUT you must briefly define each one "
        "   the first time it appears (e.g. 'RAG (Retrieval-Augmented Generation), "
        "   a technique that...').\n"
        "3. Organise your answer clearly: start with the direct answer, "
        "   then explain supporting detail, then conclude.\n"
        "4. Use bullet points if listing 3 or more items.\n"
        "5. Tone: clear, professional, accessible to a university student.\n"
        "EXAMPLE of correct Balanced answer: "
        "'A CV should include five key sections: personal details, a professional summary, "
        "work experience, education, and skills. The professional summary (a 2-3 sentence "
        "overview of your background) appears at the top and gives the reader a quick "
        "snapshot of your profile. Work experience should be listed in reverse chronological "
        "order, starting with your most recent role. [Page 2]'\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    ),
    "Expert": (
        "\n\n━━━ RESPONSE FORMAT — EXPERT MODE (STRICTLY ENFORCED) ━━━\n"
        "You MUST follow ALL of these rules. No exceptions.\n"
        "1. Write a comprehensive, technically rigorous response with NO simplification.\n"
        "2. Use precise domain-specific terminology freely — do NOT define common "
        "   professional terms, assume expert-level background knowledge.\n"
        "3. Your answer MUST include:\n"
        "   a) A direct, precise answer in the first sentence.\n"
        "   b) Supporting technical detail with specific references to the document.\n"
        "   c) Nuance, caveats, or edge cases where relevant.\n"
        "   d) Exact page citations for every claim: [Page X].\n"
        "4. DO NOT hedge with phrases like 'it seems' or 'perhaps'. "
        "   Be definitive and precise.\n"
        "5. DO NOT over-explain or use analogies. Assume the reader is a "
        "   domain expert who values density and precision over accessibility.\n"
        "6. Tone: authoritative, analytical, peer-reviewed academic standard.\n"
        "EXAMPLE of correct Expert answer: "
        "'The document specifies a reverse-chronological CV structure with a mandatory "
        "competency-based professional summary not exceeding 80 words, positioned "
        "immediately below contact metadata. Work history entries must adhere to the "
        "CAR (Context-Action-Result) framework, with quantified outcomes where available. "
        "Soft-skill declarations without evidence are explicitly contraindicated. [Page 1, 3]'\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
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
    Reformulates follow-up questions into standalone queries.
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
        language_code:    ISO code from language_detector

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

        "CORE RULES (always apply regardless of mode):\n"
        "1. Only use information from the CONTEXT section below.\n"
        "2. If the answer is not in the context, say exactly: "
        "   'I could not find this information in the document.'\n"
        "3. Always cite the page number(s): [Page X] or [Pages X, Y].\n"
        "4. Do NOT add information from your training data.\n\n"

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
    """
    if not source_docs:
        return "No source chunks were retrieved for this query."

    lines: List[str] = []
    seen:  set[str]  = set()

    for i, doc in enumerate(source_docs, start=1):
        if hasattr(doc, "page_content"):
            content  = doc.page_content
            metadata = getattr(doc, "metadata", {})
        elif isinstance(doc, dict):
            content  = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
        else:
            content  = str(doc)
            metadata = {}

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
        lines.append("")

    return "\n".join(lines)