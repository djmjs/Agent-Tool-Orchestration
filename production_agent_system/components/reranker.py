#can create a micro service for this component later if needed

from typing import List
from langchain_core.documents import Document
from fastembed.rerank.cross_encoder import TextCrossEncoder
from ..utils.logger import log_info, log_error

class ReRanker:
    def __init__(self):
        self.encoder = None
        self._setup()

    def _setup(self):
        """Initialize the FastEmbed CrossEncoder."""
        try:
            log_info("Initializing ReRanker (BAAI/bge-reranker-base)...")
            
            # Check available providers first to avoid error logs if DML is missing
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            
            providers = ["CPUExecutionProvider"]
            if "DmlExecutionProvider" in available_providers:
                providers = ["DmlExecutionProvider"]
                log_info("Using DmlExecutionProvider for ReRanker")
            else:
                log_info("DmlExecutionProvider not found, using CPUExecutionProvider for ReRanker")

            self.encoder = TextCrossEncoder(
                model_name="BAAI/bge-reranker-base", 
                providers=providers,
                cache_dir="fastembed_storage"
            )
            log_info("ReRanker initialized successfully.")
        except Exception as e:
            log_error(f"Failed to setup ReRanker: {e}")
            # Fallback to CPU if DML fails, or just raise
            try:
                log_info("Retrying ReRanker on CPU...")
                self.encoder = TextCrossEncoder(
                    model_name="BAAI/bge-reranker-base", 
                    providers=["CPUExecutionProvider"],
                    cache_dir="fastembed_storage"
                )
            except Exception as e2:
                log_error(f"Failed to setup ReRanker on CPU: {e2}")
                # If we can't initialize the model, we can't rerank. 
                # We'll handle this in the rerank method by checking if self.encoder is None.
                self.encoder = None

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Re-ranks the documents using the local FastEmbed model.
        """
        if not documents:
            return []
        
        if self.encoder is None:
            log_error("ReRanker not initialized. Returning original order.")
            return documents[:top_k]

        log_info(f"Re-ranking {len(documents)} documents...")
        
        try:
            doc_texts = [doc.page_content for doc in documents]
            
            # FastEmbed rerank returns an iterator of scores
            scores = list(self.encoder.rerank(query, doc_texts))
            
            # Pair documents with their scores
            doc_score_pairs = list(zip(documents, scores))
            
            # Sort by score descending
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Select top_k
            reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]
            
            return reranked_docs
            
        except Exception as e:
            log_error(f"Re-ranking failed: {e}. Returning original order.")
            return documents[:top_k]

    def compute_score(self, query: str, text: str) -> float:
        """
        Computes the relevance score between a query and a text.
        Returns a float score.
        """
        if self.encoder is None:
            return 0.0
        
        try:
            # rerank returns an iterator of scores
            scores = list(self.encoder.rerank(query, [text]))
            return float(scores[0])
        except Exception as e:
            log_error(f"Error computing score: {e}")
            return 0.0
