# connects the embedder and vector store to find relevant chunks

from embedding import EmbeddingModel
from vector_store import VectorStore
from config import TOP_K


class Retriever:
    # embeds a user query and searches the vector store for matches

    def __init__(self, embedder: EmbeddingModel, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = TOP_K) -> list[str]:
        # return the top-k most relevant chunks for a query
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, k)
