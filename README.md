# Production Agent System 

This directory contains a modular, production-ready architecture for an **Agentic RAG System**. Unlike simple linear pipelines, this system uses a **Graph-based Orchestrator** (LangGraph) to dynamically decide the best course of action for each user query.

## 🧠 System Workflow (Step-by-Step)

1.  **Context Management**: The system starts by fetching the user's **Long-Term Memory (LTM)** profile from Postgres and checking **Short-Term Memory (STM)** for topic switches.
2.  **Intelligent Routing**: A specialized "Router Agent" analyzes the query + context to classify the intent:
    *   **Direct Answer**: For greetings, trivial questions, or questions answerable from memory (skips search).
    *   **Database**: For technical questions about LLMs/Agents.
    *   **Web Search**: For current events, stocks, or specific facts about people/companies.
    *   **Tools**: For updating user profile info.
    *   **General**: For general knowledge or conversation requiring history.
3.  **Reformulation (Conditional)**: If search is needed, the query is rewritten specifically for the target (e.g., keyword-rich for DB, natural language for Web).
4.  **Retrieval & Grading**:
    *   If **Database** is chosen, it retrieves documents and **Grades** them. If they are irrelevant, it automatically falls back to **Web Search**.
5.  **Generation**: The LLM generates a final answer using the retrieved context (if any).
6.  **Memory Update**: After the response, the system analyzes the interaction to extract and save new user details to LTM.

## 🏗️ Architecture 

The system follows a **Router-First Agentic Workflow**:

```mermaid
graph TD
    Start([User Query]) --> Context[Manage Context\n(Fetch LTM + STM)]
    Context --> Router{Router Decision}

    %% Fast Track
    Router -- Direct Answer --> Generate[Generate Response]

    %% Standard Tracks
    Router -- DB/Web/Tool/General --> Reformulate[Reformulate Query\n(Route Specific)]
    Reformulate --> Dispatch((Dispatch))

    %% Specific Handling
    Dispatch -- Vector DB --> Retrieve[Retrieve Docs]
    Retrieve --> Grade[Grade Relevance]
    Grade -- Relevant --> Generate
    Grade -- Irrelevant --> WebSearch

    Dispatch -- Web Search --> WebSearch[Web Search]
    WebSearch --> Generate

    Dispatch -- Tool/General --> Generate

    %% Tool Loop
    Generate -- Call Tool --> Tools[Execute Tools]
    Tools --> Generate

    Generate --> End([Final Answer])
```


## 🔧 Technical Deep Dive

### 1. Memory Systems (STM & LTM)
*   **Short-Term Memory (STM)**: Uses `ConversationSummaryBufferMemory` to track the immediate conversation. It includes a **Topic Switch Detector** that analyzes the **full chat history** to prevent "topic bleeding" (e.g., asking "How much is it?" after switching from Bitcoin to Weather).
*   **Long-Term Memory (LTM)**: Uses **PostgreSQL** to store persistent User Profiles.
    *   **Structure**: Stores `preferences`, `projects`, `expertise`, `constraints`, `environment`, and **`personal_info`** (Name, Age, Location, etc.).
    *   **Read**: At the start of every turn, the system fetches the profile and injects relevant details (e.g., "User is a Python dev") into the context.
    *   **Write**: After every turn, a background process extracts new facts from the conversation and updates the database.

### 2. Router-First Architecture
Instead of always reformulating or searching, the **Router** is the first decision maker. It sees the Query + Profile + History and decides:
*   *"Do I know this already?"* -> **Direct Answer** (Fastest).
*   *"Is this technical?"* -> **Vector DB**.
*   *"Is this news?"* -> **Web Search**.
This minimizes latency and cost by skipping unnecessary steps.

### 3. Hybrid Retrieval & Re-Ranking
*   **Hybrid Search**: We use **Qdrant** for dense vector search (semantic) + sparse search (keyword/SPLADE).
*   **Cross-Encoder Re-Ranking**: The top results are re-ranked by `BAAI/bge-reranker-base` to ensure high precision.
*   **Relevance Grading**: An "LLM-as-a-Judge" evaluates retrieved docs. If they are irrelevant, the system falls back to Web Search.

## 📂 Component Breakdown

### 1. Controller (`controller/`)
- **`graph_orchestrator.py`**: The "brain" built with **LangGraph**.
    -   **`manage_context`**: Fetches LTM and checks STM.
    -   **`route_query`**: Decides the path.
    -   **`reformulate_query`**: Optimizes queries based on the route.
    -   **`generate_response`**: Produces the final answer.

### 2. Components (`components/`)
-   **`router.py`**: Classifies intent into 5 categories (`DATABASE`, `WEB`, `TOOL`, `GENERAL`, `DIRECT_ANSWER`).
-   **`postgres_storage.py`**: Manages persistent user profiles in PostgreSQL.
-   **`profile_extractor.py`**: Extracts user details (preferences, projects) from chat text.
-   **`query_reformulator.py`**: Rewrites queries with specialized prompts for each route.
-   **`retriever.py`**: Handles Qdrant Hybrid Search.
-   **`web_search.py`**: Integrates Tavily API for real-time info.
-   **`relevance_grader.py`**: Evaluates document and history relevance.

### 3. Infrastructure
-   **Docker Compose**: Orchestrates the entire stack.
    -   `agent-system`: The main Python application.
    -   `postgres`: Database for LTM.
    -   `qdrant`: Vector database for RAG.
    -   `ollama`: Local LLM inference.

## 🚀 How to Run

### Prerequisites
1.  **Environment Setup**:
    Create a `.env` file in the project root:
    ```env
    TAVILY_API_KEY=tvly-xxxxxxxxxxxx
    LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
    LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
    LANGFUSE_HOST=http://localhost:3000
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=password
    POSTGRES_DB=agents_db
    ```

2.  **Start Langfuse (Observability)**:
    ```powershell
    cd langfuse_docker
    docker-compose up -d
    cd ..
    ```

3.  **Start the System**:
    ```powershell
    docker-compose up -d --build
    ```
    The app will be available at `http://localhost:8000`.

    **First Run Only**: Pull the LLM model:
    ```powershell
    docker exec -it ollama ollama pull llama3.2
    ```

## � Quick Access Links

| Service | URL | Description |
| :--- | :--- | :--- |
| **Chat Interface** | [http://localhost:8000](http://localhost:8000) | Main UI for interacting with the agent. |
| **LTM Profiles** | [http://localhost:8000/profiles](http://localhost:8000/profiles) | View raw JSON of all stored user profiles (Postgres). |
| **Langfuse UI** | [http://localhost:3000](http://localhost:3000) | Observability dashboard (Traces, Logs). |
| **Qdrant API** | [http://localhost:6333](http://localhost:6333) | Vector Database API endpoint. |

## �🛠️ Observability (Langfuse)
This project uses **Langfuse** to trace every step of the agent's thought process.
-   **Traces**: See the exact path taken (Router -> Reformulate -> Search).
-   **Inputs/Outputs**: View the exact prompts sent to the LLM.
-   **Latency**: Identify bottlenecks in the graph.

Access the dashboard at `http://localhost:3000`.
