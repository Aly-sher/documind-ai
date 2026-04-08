from langchain_core.prompts import ChatPromptTemplate
from typing import List

def format_retrieved_chunks(docs) -> str:
    """
    Formats the raw document objects from FAISS into a readable string.
    Crucially, it injects the page number metadata so Llama 3 can cite it.
    """
    formatted_text = ""
    for i, doc in enumerate(docs):
        # Extract the page number we saved back in Phase 2
        page_num = doc.metadata.get("page", "Unknown")
        formatted_text += f"--- Source {i+1} (Page {page_num}) ---\n"
        formatted_text += f"{doc.page_content}\n\n"
        
    return formatted_text

def get_qa_prompt() -> ChatPromptTemplate:
    """
    Builds the system prompt exactly as specified in the architecture document.
    """
    system_template = """You are DocuMind AI, an expert document analyst. Your job is to answer questions STRICTLY based on the provided document context.

RULES:
1. Only use information from the CONTEXT provided below.
2. If the answer is not in the context, say: 'I could not find this information in the document.'
3. Always cite the page number(s) where you found the answer.
4. Be precise and concise. Do not add information from your training data.
5. If the question is ambiguous, ask for clarification.

CONTEXT:
{context}"""

    # We use LangChain's ChatPromptTemplate to cleanly separate the System instructions
    # from the Human's input, and we leave a placeholder for the chat history.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    return prompt