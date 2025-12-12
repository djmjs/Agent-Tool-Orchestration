# Production Agent System

This directory contains a modular, production-ready architecture for an **Agentic RAG System**. Unlike simple linear pipelines, this system uses a **Graph-based Orchestrator** (LangGraph) to dynamically decide the best course of action for each user query.

## 🏗️ Architecture

The system follows a **Router-based Agentic Workflow**:

```mermaid
graph TD
    User[User / Frontend] --> API[FastAPI Endpoint]
    API --> Orch[Graph Orchestrator]
    
    subgraph "Agent Graph"
        Orch --> Reform[Query Reformulator]
        Reform --> Router[Semantic Router]
        
        Router -->|Technical/Papers| RAG[Vector DB Retrieval]
        Router -->|News/Live Info| Web[Web Search (Tavily)]
        Router -->|Profile Updates| Tools[Tool Execution]
        Router -->|Chit-Chat| Gen[Generation]
        
        RAG --> Gen
        Web --> Gen
        Tools --> Gen
    end
    
    Gen --> |Final Response| Orch
    Orch --> User
```

## 📂 Component Breakdown

### 1. Frontend (`frontend/`)
- **`api.py`**: The entry point. Uses **FastAPI** to expose the REST API. It initializes the `GraphOrchestrator` and loads environment variables (including API keys).
- **`templates/index.html`**: A clean web interface for chatting with the agent.

### 2. Controller (`controller/`)
- **`graph_orchestrator.py`**: The new "brain" of the system, built with **LangGraph**.
    -   Manages the state of the conversation (`AgentState`).
    -   Defines the nodes (Reformulate, Route, Retrieve, Search, Generate) and conditional edges.
    -   Prevents hallucinations by strictly controlling when tools are called.

### 3. Components (`components/`)
These are specialized classes that handle specific tasks.

-   **`router.py`**: A specialized agent that classifies user intent into 4 categories:
    1.  **DATABASE**: Technical queries (triggers RAG).
    2.  **WEB**: Current events/news (triggers Tavily Search).
    3.  **TOOL**: Profile updates (triggers `update_user_info`).
    4.  **GENERAL**: Conversational chit-chat.

-   **`web_search.py`**: Integrates **Tavily API** to perform real-time web searches when the local database is insufficient.

-   **`query_reformulator.py`**: Uses the LLM to rewrite vague follow-up questions (e.g., "How does it work?") into standalone queries based on chat history.

-   **`retriever.py`**: Handles communication with **Qdrant**.
    -   Implements **Hybrid Search** (Dense + Sparse Embeddings).
    -   Uses persistent model storage in `fastembed_storage/` to avoid re-downloading models.

-   **`short_term_memory.py`**: Manages conversation history using LangChain's `ConversationSummaryBufferMemory`.

-   **`prompt_gen.py`**: Constructs the final prompt, injecting context from RAG or Web Search results into the system message.

-   **`llm.py`**: The interface for the Large Language Model (supports Ollama/Llama 3.2 and OpenAI).

### 4. Utilities (`utils/`)
-   **`logger.py`**: Centralized color-coded logging for debugging.

## 🚀 How to Run

### Prerequisites
1.  **Environment Setup**:
    Create a `.env` file in the project root:
    ```env
    TAVILY_API_KEY=tvly-xxxxxxxxxxxx
    OPENAI_API_KEY=sk-xxxxxxxx (Optional, if not using Ollama)
    ```

2.  **Docker Services**:
    Ensure Qdrant and Ollama are running:
    ```powershell
    docker-compose up -d
    ```

3.  **Python Environment**:
    ```powershell
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
    *Note: If you have issues with ONNX Runtime, reinstall it:*
    ```powershell
    pip uninstall -y onnxruntime
    pip install --force-reinstall onnxruntime-directml
    ```

4.  **Start the Server**:
    ```powershell
    python production_agent_system/main.py
    ```
    Access the UI at `http://127.0.0.1:8000`.

## 🧠 Key Features
-   **Adaptive Routing**: Doesn't just RAG everything. It knows when to search the web or just talk.
-   **Self-Correction**: If a tool doesn't exist, the system catches the error and instructs the LLM to answer directly.
-   **Persistent Caching**: Embedding models are cached locally to speed up startup.
-   **Hybrid Search**: Combines semantic understanding with keyword matching for better retrieval accuracy.


