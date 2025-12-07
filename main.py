"""
AI Articles Ingestion Pipeline - Production Ready
Ingests AI research papers from arXiv into Qdrant with hybrid search.

KEY FEATURES:
- AMD GPU acceleration via DirectML
- Hybrid search (dense + sparse vectors)
- Memory-safe batch processing
- Automatic error recovery
- Progress tracking and logging
"""

import asyncio
import os
import gc
import tempfile
import arxiv
from typing import List
from pypdf import PdfReader

from dotenv import load_dotenv 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode

from logger import log_error, log_header, log_info, log_success, log_warning

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION - Tuned for AMD GPU stability
# ============================================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ai_articles_collection_hybrid"

# Batch size reduced from 8 to 4 to prevent GPU OOM
BATCH_SIZE = 4

# Number of papers per topic
PAPERS_PER_TOPIC = 200

# ============================================================================
# EMBEDDING MODELS - AMD GPU Accelerated
# ============================================================================
dense_embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    providers=["DmlExecutionProvider"]  # AMD GPU via DirectML
)

sparse_embeddings = FastEmbedSparse(
    model_name="prithivida/Splade_PP_en_v1",
    providers=["DmlExecutionProvider"]  # AMD GPU via DirectML
)

# ============================================================================
# QDRANT CLIENT & COLLECTION SETUP
# ============================================================================
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Create collection if it doesn't exist
if not client.collection_exists(COLLECTION_NAME):
    log_info(f"Creating new collection: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        sparse_vectors_config={
            "langchain-sparse": SparseVectorParams(index=SparseIndexParams())
        }
    )
    log_success(f"Collection '{COLLECTION_NAME}' created")
else:
    log_info(f"Using existing collection: {COLLECTION_NAME}")

# Initialize Vector Store
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID
)

# Concurrency control
embedding_semaphore = asyncio.Semaphore(1)

# ============================================================================
# INDEXING FUNCTIONS
# ============================================================================
async def index_documents_async(documents: List[Document], batch_size: int = BATCH_SIZE):
    """
    Index documents in batches with memory safety.
    
    CRITICAL FIXES:
    1. Reduced batch size from 8 to 4 (prevents GPU OOM)
    2. Aggressive garbage collection every 10 batches
    3. Per-batch error handling (one failure doesn't kill the whole process)
    """
    if not documents:
        return

    log_info(f"📚 Indexing {len(documents)} chunks in batches of {batch_size}...")

    batches = [
        documents[i : i + batch_size] 
        for i in range(0, len(documents), batch_size)
    ]

    successful_batches = 0
    failed_batches = 0

    async def add_batch(batch: List[Document], batch_num: int):
        """Add a single batch with error isolation."""
        async with embedding_semaphore:
            try:
                await vectorstore.aadd_documents(batch)
                
                # Progress logging
                if batch_num % 5 == 0 or batch_num == len(batches):
                    log_success(f"   ✓ Batch {batch_num}/{len(batches)}")
                
                # Aggressive garbage collection to prevent memory buildup
                if batch_num % 10 == 0:
                    gc.collect()
                
                return True
                
            except Exception as e:
                log_error(f"   ✗ Batch {batch_num} failed: {e}")
                return False

    # Process batches with error recovery
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes
    for result in results:
        if result is True:
            successful_batches += 1
        else:
            failed_batches += 1

    # Force cleanup
    gc.collect()

    # Summary
    if failed_batches > 0:
        log_warning(f"⚠️  {successful_batches}/{len(batches)} batches succeeded, {failed_batches} failed")
    else:
        log_success(f"✅ All {successful_batches} batches indexed successfully")


# ============================================================================
# ARXIV DOWNLOAD FUNCTIONS
# ============================================================================
async def download_and_parse_arxiv(topic: str, max_results: int = PAPERS_PER_TOPIC) -> List[Document]:
    """
    Download and parse arXiv papers.
    
    IMPROVEMENTS:
    1. Per-paper error handling (one failure doesn't stop the download)
    2. Temp file cleanup even on errors
    3. Rich metadata including arxiv_id, categories, etc.
    """
    log_info(f"📥 Downloading {max_results} papers on '{topic}' from arXiv...")

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    documents = []
    successful = 0
    failed = 0

    for idx, result in enumerate(search.results(), 1):
        temp_path = None
        try:
            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_path = temp_file.name

            result.download_pdf(filename=temp_path)

            # Parse PDF
            reader = PdfReader(temp_path)
            full_text = "\n".join([page.extract_text() for page in reader.pages])

            # Create document with rich metadata
            doc = Document(
                page_content=full_text,
                metadata={
                    "title": result.title,
                    "authors": ", ".join([a.name for a in result.authors]),
                    "published": str(result.published.date()),
                    "arxiv_id": result.entry_id.split("/")[-1],
                    "url": result.entry_id,
                    "categories": ", ".join(result.categories),
                    "summary": result.summary,
                    "topic": topic,
                    "source": "arxiv",
                },
            )
            documents.append(doc)
            successful += 1

            # Progress logging every 10 papers
            if idx % 10 == 0:
                log_info(f"   [{idx}/{max_results}] Downloaded...")

        except Exception as e:
            failed += 1
            log_error(f"   ✗ Paper {idx} failed: {str(e)[:100]}")

        finally:
            # Always cleanup temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    log_success(f"✅ Downloaded {successful} papers (failed: {failed})")
    return documents


# ============================================================================
# TOPIC PROCESSING
# ============================================================================
async def process_topic(topic: str, max_papers: int = PAPERS_PER_TOPIC):
    """Process a single topic: download → split → index."""
    log_header(f"TOPIC: {topic}")

    try:
        # Step 1: Download papers
        documents = await download_and_parse_arxiv(topic, max_papers)

        if not documents:
            log_warning(f"⚠️  No papers downloaded for '{topic}', skipping...")
            return

        # Step 2: Split documents into chunks
        log_info("✂️  Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        split_docs = text_splitter.split_documents(documents)
        log_success(f"✅ Created {len(split_docs)} chunks from {len(documents)} papers")

        # Step 3: Index chunks
        await index_documents_async(split_docs)

        # Step 4: Cleanup
        del documents
        del split_docs
        gc.collect()

        log_success(f"✅ Topic '{topic}' completed")

    except Exception as e:
        log_error(f"❌ Topic '{topic}' failed: {e}")
        # Continue with next topic instead of crashing


# ============================================================================
# MAIN PIPELINE
# ============================================================================
async def main():
    """Main ingestion pipeline."""
    log_header("🚀 AI RESEARCH PAPERS INGESTION PIPELINE")
    log_info(f"Collection: {COLLECTION_NAME}")
    log_info(f"Batch Size: {BATCH_SIZE}")
    log_info(f"Papers per Topic: {PAPERS_PER_TOPIC}")
    log_info("Hybrid Search: Dense (BGE-large) + Sparse (SPLADE)")
    log_info("GPU Acceleration: AMD DirectML")

    # Topics covering various AI/ML research areas
    topics = [
        "Large Language Models",
        "AI Agents and Multi-Agent Systems",
        "Retrieval Augmented Generation RAG",
        "Prompt Engineering and In-Context Learning",
        "Chain of Thought Reasoning",
        "AI Alignment and Safety",
        "Reinforcement Learning from Human Feedback RLHF",
        "Multimodal AI and Vision Language Models",
        "Few-Shot Learning and Transfer Learning",
        "Emergent Abilities in Large Models",
        "Tool Use and Function Calling in AI",
        "Memory and Context Management in LLMs",
        "Fine-tuning and Instruction Tuning",
        "AI Ethics and Responsible AI",
    ]

    # Process each topic
    total_topics = len(topics)
    for idx, topic in enumerate(topics, 1):
        log_info(f"\n{'='*70}")
        log_info(f"Progress: {idx}/{total_topics} topics")
        log_info(f"{'='*70}\n")
        
        await process_topic(topic)
        
        # Pause between topics (be nice to arXiv API)
        await asyncio.sleep(2)

    # Final summary
    log_header("✅ INGESTION COMPLETE")
    try:
        total_count = client.count(COLLECTION_NAME).count
        log_success(f"🎉 Total vectors in collection: {total_count:,}")
        
        # Collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        log_info(f"Status: {collection_info.status}")
        log_info(f"Vectors config: {collection_info.config.params.vectors}")
        
    except Exception as e:
        log_error(f"Failed to get collection stats: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log_warning("\n⚠️  Interrupted by user. Progress saved to Qdrant.")
    except Exception as e:
        log_error(f"\n❌ Fatal error: {e}")
        raise
