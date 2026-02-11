# splits raw text into overlapping chunks for embedding

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_document(
    document: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    # break a document into overlapping chunks using recursive splitting
    if not isinstance(document, str):
        raise ValueError("Input document must be a string.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(document)

    print(f"Total chunks created: {len(chunks)}")
    return chunks
