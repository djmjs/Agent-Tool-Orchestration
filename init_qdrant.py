from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, SparseVectorParams, Distance, SparseIndexParams
import os

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "ai_articles_collection_hybrid_test3"

def init_qdrant():
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    else:
        print(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            sparse_vectors_config={
                "langchain-sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")

if __name__ == "__main__":
    init_qdrant()
