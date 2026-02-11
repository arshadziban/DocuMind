# faiss-based vector store with optional gpu acceleration

import faiss
import numpy as np

from config import DEVICE


class VectorStore:
    # builds and searches a faiss similarity index over document embeddings

    def __init__(self):
        self.index = None
        self.text_chunks: list[str] = []
        self.use_gpu = (
            DEVICE == "cuda"
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() > 0
        )

    def build_index(self, embeddings: np.ndarray, text_chunks: list[str]):
        # create a flat l2 index from the given embeddings
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings are empty.")

        dimension = embeddings.shape[1]
        cpu_index = faiss.IndexFlatL2(dimension)

        if self.use_gpu:
            gpu_resource = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index)
            device_label = "GPU"
        else:
            self.index = cpu_index
            device_label = "CPU"

        self.index.add(embeddings.astype(np.float32))
        self.text_chunks = text_chunks

        print(f"FAISS index built on [{device_label}] with {self.index.ntotal} vectors (dim={dimension}).")

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[str]:
        # return the top-k most similar chunks for a query embedding
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")

        distances, indices = self.index.search(
            query_embedding.astype(np.float32), k
        )
        # faiss returns -1 for unfilled slots when k > total vectors
        results = [self.text_chunks[i] for i in indices[0] if i != -1]
        return results
