# DocuMind

A **Retrieval-Augmented Generation (RAG)** pipeline with GPU acceleration.  
Loads documents, chunks them, embeds into vectors, indexes with FAISS, and generates grounded answers using an LLM.

## Architecture

```
User Question
     │
     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Ingestion   │────▶│   Chunking   │────▶│  Embedding   │
│ (load file)  │     │ (split text) │     │ (GPU / CPU)  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Answer     │◀────│    LLM       │◀────│   Retriever  │
│              │     │ (GPU / CPU)  │     │ (FAISS top-k)│
└──────────────┘     └──────────────┘     └──────────────┘
```

## Project Structure

```
DocuMind/
├── app/
│   ├── __init__.py          # Package init
│   ├── config.py            # Centralized settings (device, models, params)
│   ├── ingestion.py         # Document loading
│   ├── chunking.py          # Text splitting
│   ├── embedding.py         # SentenceTransformer embeddings (GPU)
│   ├── vector_store.py      # FAISS index (GPU)
│   ├── retriever.py         # Similarity search
│   ├── prompt.py            # Prompt builder
│   ├── llm.py               # LLM text generation (GPU)
│   └── main.ipynb           # Full pipeline notebook
├── data/
│   └── bangladesh.txt       # Sample document
├── env/                     # Virtual environment
└── README.md
```

## GPU Support

All compute-heavy modules automatically detect and use CUDA when available:

| Module           | GPU Feature                                       |
|------------------|---------------------------------------------------|
| `embedding.py`   | SentenceTransformer runs on CUDA device            |
| `vector_store.py`| FAISS index moved to GPU via `faiss-gpu`           |
| `llm.py`         | Model loaded in float16 with `device_map="auto"`  |

Falls back to CPU seamlessly if no GPU is detected.

## Quick Start

```bash
# Activate virtual environment
.\env\Scripts\Activate.ps1

# Run the notebook
# Open app/main.ipynb in VS Code and run all cells

# Or run a test
python app/test_chunk.py
```
