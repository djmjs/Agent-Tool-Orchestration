import os
import sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

from ..controller.orchestrator import Orchestrator
from ..utils.logger import log_info

app = FastAPI(title="Production Agent System")

# Initialize Orchestrator
orchestrator = Orchestrator()

# Templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

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
