from typing import List
from langchain_core.documents import Document
from ..utils.logger import log_info

class ReRanker:
    def __init__(self):
        pass

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Re-ranks the documents based on relevance to the query.
        Currently a pass-through that slices the top_k, but can be extended with Cross-Encoders.
        """
        log_info(f"Re-ranking {len(documents)} documents...")
        
        # Placeholder for actual re-ranking logic (e.g., using a CrossEncoder)
        # For now, we trust the retriever's order (Hybrid Search is usually good)
        
        return documents[:top_k]
