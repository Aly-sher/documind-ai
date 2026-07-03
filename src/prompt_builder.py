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
    "You are responding to a domain expert. Follow ALL rules below with zero deviation.\n\n"
    "STRUCTURAL REQUIREMENTS (mandatory):\n"
    "1. First sentence: state the direct, precise answer with no preamble whatsoever.\n"
    "2. Second paragraph: provide technical supporting detail using EXACT figures, dates, "
    "   names, percentages, and metrics from the document. Every specific claim MUST "
    "   include [Page X] citation immediately after the claim.\n"
    "3. Third paragraph (if applicable): state nuances, caveats, contradictions, "
    "   or edge cases present in the document. If none exist, omit this paragraph.\n"
    "4. Final line: a single-sentence synthesis conclusion.\n\n"
    "LANGUAGE REQUIREMENTS (mandatory):\n"
    "5. Use precise technical and domain-specific terminology throughout. "
    "   Do NOT define any term — assume expert-level reader.\n"
    "6. BANNED phrases: 'In simple terms', 'basically', 'it seems', 'appears to be', "
    "   'I think', 'perhaps', 'may', 'might', 'could be', 'it is worth noting', "
    "   'it is important to'. Be definitive.\n"
    "7. Minimum response length: 150 words. Maximum: 350 words.\n"
    "8. Use discipline-specific frameworks where applicable "
    "   (e.g. CAR framework for experience, STAR for achievements, "
    "   CGPA + institution for education).\n\n"
    "CITATION REQUIREMENTS (mandatory):\n"
    "9. Every factual claim must be followed immediately by [Page X].\n"
    "10. If a claim spans multiple pages: [Pages X, Y].\n\n"
    "CRITICAL OVERRIDE RULE:\n"
    "11. If retrieved context contains ANY relevant information, you MUST produce "
    "    a full expert-level answer. Returning 'I could not find this information' "
    "    when context exists is a FAILURE. Extract and synthesise every relevant "
    "    detail from the context before considering a not-found response.\n\n"
    "EXAMPLE of a correct Expert answer for 'What is Neha's experience?':\n"
    "'Neha Irshad's professional portfolio comprises two formal internships and one "
    "structured freelancing programme. [Page 2] At VirrgoTech (Oct 2025 – Feb 2026), "
    "she executed technical SEO audits, on-page optimisation, and KPI tracking via "
    "analytics platforms. [Page 2] Concurrently, her NITB engagement involved "
    "responsive UI development using HTML5/CSS3, applying mobile-first design "
    "principles. [Page 1] Her FYP operationalised a sustainable fashion e-commerce "
    "platform, demonstrating full-stack competency in Java backend integration with "
    "a CSS/HTML frontend. [Page 2] The Professional Freelancing Training Programme "
    "formalised her cross-domain marketability in web development, SEO, and Java. "
    "[Page 2] Collectively, her experience profile reflects a practitioner-level "
    "trajectory across frontend engineering and digital marketing verticals.'\n"
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
    system = (
        "Given the conversation history and the latest user question, "
        "your job is to reformulate the question into a standalone question "
        "ONLY if it contains pronouns or references that require context to understand "
        "(e.g. 'what about that?', 'tell me more about it'). "
        "\n\nCRITICAL RULES:"
        "\n1. If the question is already self-contained and clear, return it EXACTLY as written — do not change a single word."
        "\n2. If the same or very similar question has been asked before, return it EXACTLY as written."
        "\n3. NEVER rephrase, summarise, or change the meaning of the question."
        "\n4. Do NOT answer the question — only reformulate if strictly necessary."
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