# System Architecture

```mermaid
graph TD
    %% Styles
    classDef db fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef llm fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef app fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef obs fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef ext fill:#f5f5f5,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5;

    %% User
    User([👤 User / Browser])

    %% LLM Application Group
    subgraph "LLM Application (Agent System)"
        direction TB
        FastAPI[⚡ FastAPI Server]:::app
        Guardrails[🛡️ Input Guardrails]:::app
        Orchestrator[🤖 LangGraph Orchestrator]:::app
        LangChain[🦜🔗 LangChain]:::app
        
        subgraph "Tools"
            Tavily[🔍 Tavily Search]:::ext
            Arxiv[📄 Arxiv]:::ext
            PDF[📑 PDF Reader]:::ext
        end
    end

    %% Inference Engine Group
    subgraph "Inference Engine"
        Ollama[🦙 Ollama (Llama 3.2)]:::llm
        FastEmbed[🔢 FastEmbed (Embeddings)]:::llm
    end

    %% Databases Group
    subgraph "Databases & Memory"
        Qdrant[(💠 Qdrant Vector DB)]:::db
        Postgres[(🐘 PostgreSQL)]:::db
    end

    %% Observability Group
    subgraph "Observability"
        Langfuse[🔭 Langfuse]:::obs
    end

    %% Connections
    User <-->|HTTP/JSON| FastAPI
    FastAPI --> Orchestrator
    Orchestrator --> Guardrails
    Guardrails -->|Safe| LangChain
    Guardrails -->|Unsafe| User
    Orchestrator --> LangChain
    
    %% Tool Connections
    LangChain --> Tavily
    LangChain --> Arxiv
    LangChain --> PDF

    %% Inference Connections
    Orchestrator <-->|Generate| Ollama
    Orchestrator <-->|Embed| FastEmbed

    %% Database Connections
    Orchestrator <-->|RAG / Context| Qdrant
    Orchestrator <-->|User Profiles| Postgres

    %% Observability Connections
    Orchestrator -.->|Trace| Langfuse
    LangChain -.->|Trace| Langfuse

```
