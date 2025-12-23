import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
# Calculate project root (2 levels up from frontend/api.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
dotenv_path = os.path.join(project_root, '.env')
# Ensure we don't override Docker env vars
load_dotenv(dotenv_path, override=False)

from ..utils.logger import log_info

# Debug Logging
log_info(f"LANGFUSE_HOST: {os.getenv('LANGFUSE_HOST')}")
log_info(f"LANGFUSE_BASE_URL: {os.getenv('LANGFUSE_BASE_URL')}")

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from ..controller.graph_orchestrator import GraphOrchestrator
from ..utils.logger import log_info

app = FastAPI(title="Production Agent System")

# Mount Static Files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize Orchestrator
orchestrator = GraphOrchestrator()

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(static_dir, "favicon.ico"))

class QueryRequest(BaseModel):
    query: str

class SourceDocument(BaseModel):
    title: str
    url: str
    content_preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    user_prompt_history: List[str]
    chat_answers_history: List[str]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    result = await orchestrator.process_query(request.query)
    return QueryResponse(**result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
