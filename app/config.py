# all project-level settings — models, devices, hyperparams

import torch

# pick gpu if available, otherwise fall back to cpu
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# embedding model (runs via sentencetransformers)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# llm for answer generation
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_MAX_NEW_TOKENS = 200
LLM_TEMPERATURE = 0.7

# chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# how many chunks to retrieve per query
TOP_K = 3
