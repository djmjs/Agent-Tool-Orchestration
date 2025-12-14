import os
import onnxruntime as ort
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from ..utils.logger import log_error, log_info

# Configuration (Should ideally be in a config file)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ai_articles_collection_hybrid_test3"

class Retriever:
    def __init__(self):
        self.retriever = None
        self._setup()

    def _setup(self):
        """Initialize the Qdrant retriever."""
        try:
            log_info("Initializing Retriever...")
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
            
            # Check available providers
            available_providers = ort.get_available_providers()
            log_info(f"Available ONNX providers: {available_providers}")
            
            providers = ["DmlExecutionProvider"] if "DmlExecutionProvider" in available_providers else ["CPUExecutionProvider"]
            log_info(f"Using providers: {providers}")

            dense_embeddings = FastEmbedEmbeddings(
                model_name="BAAI/bge-large-en-v1.5",
                providers=providers,
                cache_dir="fastembed_storage"
            )

            sparse_embeddings = FastEmbedSparse(
                model_name="prithivida/Splade_PP_en_v1",
                providers=providers,
                cache_dir="fastembed_storage"
            )

            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=dense_embeddings,
                sparse_embedding=sparse_embeddings,
                retrieval_mode=RetrievalMode.HYBRID
            )
            
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # Fetch a bit more for re-ranking
        except Exception as e:
            log_error(f"Failed to setup retriever: {e}")
            raise

    async def retrieve(self, query: str):
        if not self.retriever:
            raise ValueError("Retriever not initialized")
        return await self.retriever.ainvoke(query)
