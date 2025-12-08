# Local RAG Frontend

This is a simple FastAPI interface for your Local RAG application.

## Prerequisites

Ensure you have the required packages installed:

```bash
pip install -r ../requirements.txt
```

Make sure your Qdrant and Ollama instances are running.

## Running the App

From the root of the project (`agents_test`):

```bash
python frontend/app.py
```

Or using uvicorn directly:

```bash
uvicorn frontend.app:app --reload
```

Then open your browser at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
