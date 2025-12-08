# AI Article Aggregator & Vectorizer

This project searches for AI/ML/LLM articles using Tavily, vectorizes them using a local embedding model (FastEmbed), and stores them in a local Qdrant vector database.

## Prerequisites

- Docker & Docker Compose
- Python 3.8+
- A Tavily API Key (Get one at [tavily.com](https://tavily.com/))

## Setup

1.  **Start Qdrant:**
    ```bash
    docker-compose up -d
    ```

2. Venv
    ```bash
    .\venv\Scripts\Activate.ps1
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    - Open `.env`
    - Paste your Tavily API key: `TAVILY_API_KEY=tvly-...`

## Usage

Run the main script:

```bash
python main.py
```

    When wanna use llama3.2

        docker exec -it agents_test-ollama-1 ollama pull llama3.2

In Qdrant run

```bash
{
  "limit": 5,
  "sample": 900,
  "tree": true
}
```


The script will:
1.  Fetch articles related to AI/LLMs.
2.  Download the "FastEmbed" model (first run only).
3.  Vectorize the article content.
4.  Store vectors and metadata in Qdrant.
5.  Perform a test query to verify.
