"""
RAG Test with 'Free OpenAI' (Local LLM)
This script tests if an LLM can extract information from your Qdrant collection.

PREREQUISITES FOR "FREE OPENAI":
You need a local LLM server running that mimics the OpenAI API.
Common options:
1. LM Studio (https://lmstudio.ai/) -> Start server on port 1234
2. Ollama (https://ollama.com/) -> Start server on port 11434
3. LocalAI (https://localai.io/)

If you have a real OpenAI key, you can add OPENAI_API_KEY to your .env file.
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from logger import log_header, log_info, log_success, log_error, log_warning, Colors

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ai_articles_collection_hybrid_test"

# LLM Configuration (Adjust these for your local setup)
# Using Ollama running in Docker
LOCAL_LLM_URL = "http://localhost:11434/v1" 
MODEL_NAME = "llama3.2"

# ============================================================================
# SETUP
# ============================================================================
def setup_retriever():
    """Initialize the Qdrant retriever."""
    log_info("Initializing Retriever (Dense + Sparse)...")
    
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    dense_embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        providers=["DmlExecutionProvider"]
    )

    sparse_embeddings = FastEmbedSparse(
        model_name="prithivida/Splade_PP_en_v1",
        providers=["DmlExecutionProvider"]
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )
    
    # Return retriever that fetches top 3 results
    return vectorstore.as_retriever(search_kwargs={"k": 3})

def setup_llm():
    """Initialize the LLM (Local or OpenAI)."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        log_info(f"No OpenAI Key found. Attempting to connect to Local LLM at {LOCAL_LLM_URL}...", Colors.YELLOW)
        # Point to local server (Ollama)
        return ChatOpenAI(
            base_url=LOCAL_LLM_URL,
            api_key="ollama",  # Placeholder key
            model=MODEL_NAME,  # Must match the model pulled in Ollama
            temperature=0.0
        )
    else:
        log_info("Using Real OpenAI API...", Colors.GREEN)
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# ============================================================================
# RAG PIPELINE
# ============================================================================
async def run_rag_test(query: str):
    log_header(f"❓ QUERY: {query}")
    
    # 1. Setup
    retriever = setup_retriever()
    llm = setup_llm()
    
    # 2. Retrieve Context
    log_info("🔍 Retrieving documents...")
    docs = await retriever.ainvoke(query)
    
    if not docs:
        log_error("No documents found! Check your collection.")
        return

    log_success(f"Found {len(docs)} relevant documents.")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"Source: {doc.metadata.get('url', 'Unknown')}")
        print(f"Content Preview: {doc.page_content[:150]}...")

    # 3. Generate Answer
    log_info("\n🤖 Generating Answer with LLM...")
    
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": lambda x: "\n\n".join([d.page_content for d in docs]), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    try:
        response = await chain.ainvoke(query)
        log_header("💡 LLM ANSWER")
        print(response)
        log_success("\n✅ RAG Pipeline Test Complete!")
    except Exception as e:
        log_error(f"\n❌ LLM Generation Failed: {e}")
        log_warning("Make sure your Local LLM server (LM Studio/Ollama) is running!")
        log_warning(f"Target URL: {LOCAL_LLM_URL}")

if __name__ == "__main__":
    # Test Query based on the "AI Agents" topic we ingested
    TEST_QUERY = "What are the key challenges in multi-agent reinforcement learning?"
    
    asyncio.run(run_rag_test(TEST_QUERY))
