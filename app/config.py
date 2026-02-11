"""All project-level settings live here — models, devices, hyperparams."""

import torch

# Pick GPU if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model (runs via SentenceTransformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# LLM for answer generation
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_MAX_NEW_TOKENS = 200
LLM_TEMPERATURE = 0.7

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# How many chunks to retrieve per query
TOP_K = 3
