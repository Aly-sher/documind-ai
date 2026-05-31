"""
LangChain RAG chain assembly and query execution for DocuMind AI.

Wires together:
  - ChatGroq (Llama 3 8B via Groq API)
  - History-aware retriever (contextualises follow-up questions)
  - Stuff documents chain (injects retrieved chunks into QA prompt)
  - Conversation history (capped at last 3 turns per SRS §6.3)

All Groq API error cases from SRS §8 are handled here:
  - Timeout → retry once, then raise
  - Rate limit → raise with informative message
  - Missing API key → raise immediately with clear message
"""

import time
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.vectorstores import VectorStoreRetriever

from src.prompt_builder import get_contextualise_prompt, get_qa_prompt
from utils.logger import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SRS §6.1 — Groq model specification
GROQ_MODEL: str = "llama3-8b-8192"

# SRS §6.3 failure mode: "Context bleeding" — cap history at last N turns
MAX_HISTORY_TURNS: int = 3

# Retry config for Groq API timeouts
MAX_RETRIES: int = 1
RETRY_DELAY_SECONDS: float = 2.0


# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def get_llm(api_key: str) -> ChatGroq:
    """
    Initialise the Groq-hosted Llama 3 8B LLM.

    Args:
        api_key: Groq API key (validated before this call in app.py).

    Returns:
        ChatGroq instance with temperature=0 for deterministic, factual answers.
    """
    return ChatGroq(
        model=GROQ_MODEL,           # FIX: was model_name= (deprecated in langchain-groq >=0.2)
        temperature=0,              # Deterministic — reduces hallucination
        api_key=api_key,            # FIX: was groq_api_key= (deprecated in langchain-groq >=0.2)
        timeout=30,                 # FIX: was request_timeout= (deprecated in langchain-groq >=0.2)
    )


# ---------------------------------------------------------------------------
# History helper
# ---------------------------------------------------------------------------

def _cap_history(chat_history: List, max_turns: int = MAX_HISTORY_TURNS) -> List:
    """
    Return only the last `max_turns` human/AI message pairs.

    Each turn = 1 HumanMessage + 1 AIMessage = 2 items.
    Cap at max_turns * 2 items from the end of the list.

    Args:
        chat_history: List of HumanMessage / AIMessage objects.
        max_turns:    Maximum number of conversation turns to retain.

    Returns:
        Trimmed list of message objects.
    """
    max_messages = max_turns * 2
    if len(chat_history) > max_messages:
        logger.debug(
            f"Trimming chat history from {len(chat_history)} to "
            f"{max_messages} messages ({max_turns} turns)."
        )
        return chat_history[-max_messages:]
    return chat_history


# ---------------------------------------------------------------------------
# RAG chain builder
# ---------------------------------------------------------------------------

def build_rag_chain(retriever: VectorStoreRetriever, api_key: str):
    """
    Assemble the full history-aware RAG chain.

    Chain structure (per SRS §4.1 Phase B):
      User question
        → History-aware retriever (reformulates question if needed)
        → FAISS similarity search (top-5 chunks)
        → Stuff documents chain (injects context into QA prompt)
        → Llama 3 via Groq API
        → Answer with citations

    Args:
        retriever: Configured FAISS retriever from retriever.py.
        api_key:   Groq API key.

    Returns:
        Assembled LangChain retrieval chain, ready for .invoke().
    """
    llm = get_llm(api_key)

    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=get_contextualise_prompt(),
    )

    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=get_qa_prompt(),
    )

    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=qa_chain,
    )

    logger.info(f"RAG chain assembled (model={GROQ_MODEL}).")
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
    Execute a user question through the RAG chain with retry logic.

    Args:
        rag_chain:    Assembled chain from build_rag_chain().
        question:     The user's validated question string.
        chat_history: Full conversation history (will be capped internally).

    Returns:
        Dict with keys:
          "answer"   → str: LLM-generated answer with citations
          "sources"  → List[Document]: retrieved source chunks
          "question" → str: the (possibly reformulated) standalone question

    Raises:
        RuntimeError: On rate limit, persistent timeout, or unexpected API error.
    """
    trimmed_history = _cap_history(chat_history)

    payload = {
        "input": question,
        "chat_history": trimmed_history,
    }

    for attempt in range(1, MAX_RETRIES + 2):  # +2 so range covers 1 retry
        try:
            logger.info(f"Invoking RAG chain (attempt {attempt}/{MAX_RETRIES + 1})...")
            response = rag_chain.invoke(payload)

            answer = response.get("answer", "")
            sources = response.get("context", [])

            logger.info(
                f"Response received — {len(answer)} chars, "
                f"{len(sources)} source chunk(s) used."
            )
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
            }

        except Exception as e:
            error_str = str(e).lower()

            # --- Rate limit (SRS §8) ---
            if "rate limit" in error_str or "429" in error_str:
                logger.warning("Groq rate limit hit.")
                raise RuntimeError(
                    "rate_limit_error|Rate limit reached. Please wait 60 seconds before retrying."
                ) from e

            # --- Timeout with retry (SRS §8) ---
            if "timeout" in error_str or "timed out" in error_str:
                if attempt <= MAX_RETRIES:
                    logger.warning(f"Request timed out — retrying in {RETRY_DELAY_SECONDS}s...")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    logger.error("Request timed out after retry.")
                    raise RuntimeError(
                        "timeout_error|Response timed out. Please try again."
                    ) from e

            # --- Auth / missing key ---
            if "auth" in error_str or "api key" in error_str or "401" in error_str:
                logger.error("Groq API authentication failed.")
                raise RuntimeError(
                    "auth_error|API key not configured or invalid. Contact support."
                ) from e

            # --- Unknown error ---
            logger.error(f"Unexpected error from Groq API: {e}")
            raise RuntimeError(f"unknown_error|Unexpected error: {e}") from e


__all__ = ["build_rag_chain", "run_query", "get_llm", "GROQ_MODEL", "MAX_HISTORY_TURNS"]