import asyncio
import os
import gc
import tempfile
import arxiv
from typing import List
from pypdf import PdfReader

from dotenv import load_dotenv 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

# Load environment variables
load_dotenv()

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ai_articles_collection_hybrid"

# Initialize Dense Embeddings (BAAI/bge-m3)
# This handles the semantic understanding
dense_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize Sparse Embeddings (SPLADE via FastEmbed)
# This handles the keyword matching (Hybrid Search)
sparse_embeddings = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")

# Initialize Qdrant Client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure collection exists with HYBRID configuration
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        # Dense Vectors (1024 dim for BGE-M3)
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        # Sparse Vectors (for Keyword Search)
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

# Semaphore to limit concurrent embedding tasks on CPU
# 2 is a safe number for most CPUs to avoid freezing the system
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
        # Use semaphore to prevent too many parallel CPU-heavy embedding tasks
        async with cpu_semaphore:
            try:
                await vectorstore.aadd_documents(batch)
                # Log progress periodically
                if batch_num % 5 == 0 or batch_num == len(batches):
                    log_success(f"   Indexed batch {batch_num}/{len(batches)}")
            except Exception as e:
                log_error(f"   Failed to add batch {batch_num} - {e}")
                return False
            return True

    # Process batches concurrently (but limited by semaphore)
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)
    log_success(
        f"VectorStore: Batch processing complete! ({successful}/{len(batches)} batches)"
    )

def download_and_parse_arxiv(topic: str, max_docs: int = 100) -> List[Document]:
    """Robustly download and parse Arxiv papers using raw client."""
    docs = []
    
    # Use raw arxiv client for better control
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_docs,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = list(client.results(search))
    log_info(f"   Arxiv API returned {len(results)} candidates. Downloading PDFs...")
    
    for i, result in enumerate(results):
        try:
            # Create a temp file for the PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_path = temp_pdf.name
                
            # Download
            result.download_pdf(filename=temp_path)
            
            # Extract text
            text = ""
            try:
                reader = PdfReader(temp_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                log_warning(f"   Failed to parse PDF for '{result.title}': {e}")
                continue
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            if len(text) > 500: # Only keep substantial documents
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "title": result.title,
                        "source": result.entry_id,
                        "published": str(result.published),
                        "authors": ", ".join([a.name for a in result.authors]),
                        "summary": result.summary,
                        "topic": topic,
                        "type": "research_paper"
                    }
                ))
                if (i + 1) % 10 == 0:
                    log_info(f"   Downloaded & Parsed {i+1}/{len(results)} papers...")
            
        except Exception as e:
            log_error(f"   Error processing paper {i}: {e}")
            
    return docs


async def process_topic(topic: str):
    """Fetch, split, and index data for a single topic to manage memory."""
    log_header(f"PROCESSING TOPIC: {topic}")
    
    docs_to_process = []

    # 1. Arxiv Search
    try:
        log_info(f"🔍 Arxiv: Searching for '{topic}' (Target: 200 papers)...")
        
        # Run blocking download in thread
        arxiv_docs = await asyncio.to_thread(download_and_parse_arxiv, topic, 200)
        
        if arxiv_docs:
            log_success(f"   Successfully processed {len(arxiv_docs)} Arxiv papers")
            docs_to_process.extend(arxiv_docs)
        else:
            log_warning("   No Arxiv papers successfully processed.")
            
    except Exception as e:
        log_error(f"Arxiv Search failed for {topic}: {e}")

    if not docs_to_process:
        log_warning(f"No documents found for topic: {topic}")
        return

    # 2. Split Documents
    log_info(f"✂️  Splitting {len(docs_to_process)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(docs_to_process)
    
    log_info(f"   Created {len(splitted_docs)} chunks")

    # 3. Index Documents
    await index_documents_async(splitted_docs, batch_size=32)

    # 4. Memory Cleanup
    # Explicitly delete large lists and collect garbage to free up RAM
    del docs_to_process
    del splitted_docs
    gc.collect()
    log_success(f"Finished processing topic: {topic}")


async def main():
    """Main async function to orchestrate the massive ingestion."""
    log_header("MASSIVE DATA INGESTION PIPELINE (HYBRID)")
    log_info("Starting ingestion process with Dense + Sparse vectors...", Colors.PURPLE)

    # List of topics to cover a wide range of AI data
    topics = [
        "Large Language Models (LLMs)",
        "Autonomous AI Agents",
        "Multi-Agent Systems",
        "Retrieval Augmented Generation (RAG)",
        "Chatbot Architecture and Design",
        "Prompt Engineering Techniques",
        "Reinforcement Learning from Human Feedback (RLHF)",
        "Chain of Thought Reasoning",
        "Transformer Architectures",
        "AI Safety and Alignment",
        "Tool Use in AI Agents",
        "Memory Systems for LLMs",
        "Vision-Language Models (VLMs)",
        "Fine-tuning LLMs"
    ]

    total_topics = len(topics)
    for i, topic in enumerate(topics):
        log_info(f"--- Progress: Topic {i+1}/{total_topics} ---", Colors.BOLD)
        await process_topic(topic)
        
        # Small pause to be polite to APIs and let CPU cool down slightly
        await asyncio.sleep(2)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 All topics processed successfully!")

    # Final Verification
    log_header("VERIFICATION")
    test_query = "What are AI agents?"
    results = await vectorstore.asimilarity_search(test_query, k=1)
    if results:
        log_success(f"Verification search found: {results[0].metadata.get('title')}")

if __name__ == "__main__":
    asyncio.run(main())
