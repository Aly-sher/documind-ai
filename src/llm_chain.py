"""
LangChain RAG chain assembly and query execution for DocuMind AI.

Phase 1 changes:
  - build_rag_chain now accepts complexity_level and language_code,
    forwarding them to get_qa_prompt() for the Complexity Tuner and
    Multilingual features.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

from src.prompt_builder import get_contextualise_prompt, get_qa_prompt
from utils.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_MODEL: str       = "llama-3.1-8b-instant"
MAX_HISTORY_TURNS: int = 3
MAX_RETRIES: int       = 1
RETRY_DELAY_SECONDS: float = 2.0


# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def get_llm(api_key: str) -> ChatGroq:
    """Initialise the Groq-hosted Llama 3.1 8B LLM."""
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=api_key,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------

def _cap_history(chat_history: List, max_turns: int = MAX_HISTORY_TURNS) -> List:
    """Return only the last *max_turns* human/AI message pairs."""
    max_messages = max_turns * 2
    if len(chat_history) > max_messages:
        logger.debug(
            f"Trimming chat history {len(chat_history)} → {max_messages} messages."
        )
        return chat_history[-max_messages:]
    return chat_history


# ---------------------------------------------------------------------------
# RAG chain builder  (Phase 1: complexity + language injection)
# ---------------------------------------------------------------------------

def build_rag_chain(
    retriever: VectorStoreRetriever,
    api_key: str,
    complexity_level: str = "Balanced",
    language_code: str    = "en",
):
    """
    Assemble the full history-aware RAG chain.

    Args:
        retriever:        Configured FAISS retriever from retriever.py.
        api_key:          Groq API key.
        complexity_level: "Simple" | "Balanced" | "Expert"  (Complexity Tuner)
        language_code:    Language code from language_detector  (Multilingual)

    Returns:
        Assembled LangChain retrieval chain ready for .invoke().
    """
    llm = get_llm(api_key)

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=get_contextualise_prompt(),
    )

    # Phase 1: pass complexity + language into the QA prompt
    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=get_qa_prompt(
            complexity_level=complexity_level,
            language_code=language_code,
        ),
    )

    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=qa_chain,
    )

    logger.info(
        f"RAG chain assembled — model={GROQ_MODEL} "
        f"complexity={complexity_level} lang={language_code}"
    )
    return rag_chain


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def run_query(
    rag_chain,
    question: str,
    chat_history: List,
) -> Dict[str, Any]:
    """
    Execute a question through the RAG chain with retry logic.

    Returns:
        Dict with keys: "answer", "sources", "question"

    Raises:
        RuntimeError: On rate limit, persistent timeout, or auth failure.
    """
    trimmed = _cap_history(chat_history)
    payload = {"input": question, "chat_history": trimmed}

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            logger.info(f"RAG chain invoke — attempt {attempt}/{MAX_RETRIES + 1}")
            response = rag_chain.invoke(payload)

            answer  = response.get("answer", "")
            sources = response.get("context", [])

            logger.info(
                f"Response received — {len(answer)} chars, "
                f"{len(sources)} source chunk(s)."
            )
            return {"answer": answer, "sources": sources, "question": question}

        except Exception as exc:
            err = str(exc).lower()

            if "rate limit" in err or "429" in err:
                logger.warning("Groq rate limit hit.")
                raise RuntimeError(
                    "rate_limit_error|Rate limit reached. Please wait 60 seconds."
                ) from exc

            if "timeout" in err or "timed out" in err:
                if attempt <= MAX_RETRIES:
                    logger.warning(f"Timeout — retrying in {RETRY_DELAY_SECONDS}s…")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                raise RuntimeError(
                    "timeout_error|Response timed out. Please try again."
                ) from exc

            if any(k in err for k in ("auth", "api key", "401")):
                logger.error("Groq API auth failure.")
                raise RuntimeError(
                    "auth_error|API key not configured or invalid. Contact support."
                ) from exc

            logger.error(f"Unexpected Groq error: {exc}")
            raise RuntimeError(f"unknown_error|Unexpected error: {exc}") from exc


__all__ = [
    "build_rag_chain",
    "run_query",
    "get_llm",
    "GROQ_MODEL",
    "MAX_HISTORY_TURNS",
]
