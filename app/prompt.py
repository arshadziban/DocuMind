"""Builds the RAG prompt from retrieved context and the user's question."""


def build_prompt(query: str, context_chunks: list[str]) -> str:
    """Build a grounded RAG prompt using retrieved context chunks."""
    if not context_chunks:
        context = "No relevant context found."
    else:
        context = "\n\n".join(context_chunks)

    prompt = f"""You are a helpful AI assistant.

Use ONLY the information provided in the context below to answer the question.
If the answer is not found in the context, say: "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt
