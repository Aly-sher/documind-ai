from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

def create_chunks(documents: List[Dict], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict]:
    """
    Splits document pages into smaller chunks while inheriting page metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # This priority list ensures we don't cut sentences in half if possible
        separators=["\n\n", "\n", ".", " ", ""], 
        length_function=len
    )
    
    chunks = []
    for doc in documents:
        # Split the text of a single page
        page_chunks = text_splitter.split_text(doc["text"])
        
        # Senior Tip: We must re-attach the page metadata to *every single chunk*. 
        # When FAISS retrieves a chunk later, this is how it knows what page it came from.
        for chunk_text in page_chunks:
            chunks.append({
                "text": chunk_text,
                "metadata": doc["metadata"] 
            })
            
    return chunks