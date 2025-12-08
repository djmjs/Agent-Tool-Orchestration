import asyncio
import os
import gc
from typing import List

from dotenv import load_dotenv 
from tavily import TavilyClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, RetrievalMode

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse

from big_ingestion_one_day.logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

# Load environment variables
load_dotenv()

# Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ai_articles_collection_hybrid"

# Initialize Dense Embeddings (BAAI/bge-m3)
dense_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize Sparse Embeddings (SPLADE via FastEmbed)
sparse_embeddings = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")

# Initialize Qdrant Client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure collection exists with HYBRID configuration
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        sparse_vectors_config={
            "langchain-sparse": SparseVectorParams(index=True)
        }
    )

# Initialize Vector Store with Hybrid Mode
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID
)

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Semaphore to limit concurrent embedding tasks on CPU
cpu_semaphore = asyncio.Semaphore(2)

async def index_documents_async(documents: List[Document], batch_size: int = 32):
    """Process documents in batches asynchronously with resource limits."""
    if not documents:
        return

    log_info(
        f"📚 VectorStore: Indexing {len(documents)} chunks (Hybrid)...",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def add_batch(batch: List[Document], batch_num: int):
        async with cpu_semaphore:
            try:
                await vectorstore.aadd_documents(batch)
                if batch_num % 5 == 0 or batch_num == len(batches):
                    log_success(f"   Indexed batch {batch_num}/{len(batches)}")
            except Exception as e:
                log_error(f"   Failed to add batch {batch_num} - {e}")
                return False
            return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)
    log_success(
        f"VectorStore: Batch processing complete! ({successful}/{len(batches)} batches)"
    )

async def process_topic(topic: str):
    """Fetch, split, and index data for a single topic using Tavily."""
    log_header(f"PROCESSING TOPIC (TAVILY): {topic}")
    
    docs_to_process = []

    # Tavily Search
    try:
        log_info(f"🔍 Tavily: Searching for '{topic}'...")
        # Using advanced search to get high quality results
        response = tavily.search(query=topic, search_depth="advanced", max_results=10)
        results = response.get("results", [])
        
        if results:
            log_success(f"   Found {len(results)} Tavily articles")
            for result in results:
                content = result.get("content")
                if content:
                    docs_to_process.append(
                        Document(
                            page_content=content,
                            metadata={
                                "title": result.get("title"),
                                "source": result.get("url"),
                                "topic": topic,
                                "type": "web_article"
                            },
                        )
                    )
    except Exception as e:
        log_error(f"Tavily error for {topic}: {e}")

    if not docs_to_process:
        log_warning(f"No documents found for topic: {topic}")
        return

    # Split Documents
    log_info(f"✂️  Splitting {len(docs_to_process)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(docs_to_process)
    
    log_info(f"   Created {len(splitted_docs)} chunks")

    # Index Documents
    await index_documents_async(splitted_docs, batch_size=32)

    # Memory Cleanup
    del docs_to_process
    del splitted_docs
    gc.collect()
    log_success(f"Finished processing topic: {topic}")


async def main():
    """Main async function to orchestrate the Tavily ingestion."""
    log_header("TAVILY DATA INGESTION PIPELINE")
    log_info("Starting ingestion process (Web Articles Only)...", Colors.PURPLE)

    # List of topics
    topics = [
        "Artificial Intelligence Agents",
        "Large Language Models (LLMs)",
        "Retrieval Augmented Generation (RAG)",
        "Generative Adversarial Networks (GANs)",
        "Transformer Neural Networks",
        "Reinforcement Learning",
        "Computer Vision and CNNs",
        "Natural Language Processing (NLP)",
        "AI Ethics and Safety",
        "Multi-agent Systems"
    ]

    total_topics = len(topics)
    for i, topic in enumerate(topics):
        log_info(f"--- Progress: Topic {i+1}/{total_topics} ---", Colors.BOLD)
        await process_topic(topic)
        await asyncio.sleep(2)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 All topics processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
