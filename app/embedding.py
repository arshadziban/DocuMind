"""Turns text into dense vectors using SentenceTransformers (GPU-accelerated)."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, DEVICE


class EmbeddingModel:
    """Wraps a SentenceTransformer for document and query embedding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.device = DEVICE
        print(f"Loading embedding model on [{self.device.upper()}]...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"Embedding model '{model_name}' loaded successfully.")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text chunks into float32 embeddings."""
        if not texts:
            raise ValueError("Text list is empty.")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            device=self.device,
            convert_to_numpy=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a float32 embedding."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        embedding = self.model.encode(
            [query],
            device=self.device,
            convert_to_numpy=True,
        )
        return np.array(embedding, dtype=np.float32)
