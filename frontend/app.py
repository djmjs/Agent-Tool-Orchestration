import os
import sys
import asyncio
import uvicorn
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
import onnxruntime as ort

# Add parent directory to path to import logger and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from big_ingestion_one_day.logger import log_header, log_info, log_success, log_error, log_warning, Colors

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "ai_articles_collection_hybrid_test3"
LOCAL_LLM_URL = "http://localhost:11434/v1" 
MODEL_NAME = "llama3.2"

# ============================================================================
# FASTAPI SETUP
# ============================================================================
# app definition moved to lifespan section
# app = FastAPI(title="Local RAG Interface")
# templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

class QueryRequest(BaseModel):
    query: str

class SourceDocument(BaseModel):
    title: str
    url: str
    content_preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

# ============================================================================
# RAG COMPONENTS
# ============================================================================
def setup_retriever():
    """Initialize the Qdrant retriever."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
        
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
        
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        log_error(f"Failed to setup retriever: {e}")
        raise

def setup_llm():
    """Initialize the LLM (Local or OpenAI)."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        log_info(f"Connecting to Local LLM at {LOCAL_LLM_URL}...")
        return ChatOpenAI(
            base_url=LOCAL_LLM_URL,
            api_key="ollama",
            model=MODEL_NAME,
            temperature=0.0
        )
    else:
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# Initialize components globally to avoid reloading on every request
# In a production app, you might want to handle this differently (e.g., lifespan events)
retriever = None
llm = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm
    log_info("Starting up FastAPI application...")
    try:
        retriever = setup_retriever()
        llm = setup_llm()
        log_success("RAG components initialized successfully.")
    except Exception as e:
        log_error(f"Startup failed: {e}")
    yield
    # Clean up if needed
    log_info("Shutting down FastAPI application...")

# ============================================================================
# ROUTES
# ============================================================================

app = FastAPI(title="Local RAG Interface", lifespan=lifespan)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not retriever or not llm:
        raise HTTPException(status_code=503, detail="RAG components not initialized")

    query_text = request.query
    log_header(f"Processing Query: {query_text}")

    try:
        # 1. Retrieve Documents
        docs = await retriever.ainvoke(query_text)
        
        sources = []
        for doc in docs:
            sources.append(SourceDocument(
                title=doc.metadata.get('title', 'Unknown'),
                url=doc.metadata.get('url', '#'),
                content_preview=doc.page_content[:200]
            ))

        if not docs:
            return QueryResponse(answer="I couldn't find any relevant information in the database.", sources=[])

        # 2. Generate Answer
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

        answer = await chain.ainvoke(query_text)
        
        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        log_error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
