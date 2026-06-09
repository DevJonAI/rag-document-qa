from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    """
    Request schema for the /query endpoint.

    Attributes:
        question (str): The natural language question to ask about the indexed documents.
        filter_document (Optional[str]): If provided, restricts the search to chunks
            from this specific document. If None, searches across all indexed documents.
    """
    question: str
    filter_document: Optional[str] = None


class QueryResponse(BaseModel):
    """
    Response schema for the /query endpoint.

    Attributes:
        answer (str): The generated answer from Claude.
        sources (list[str]): List of text excerpts used as context for the answer.
            Each excerpt is truncated to 200 characters.
    """
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    """
    Response schema for the /ingest endpoint.

    Attributes:
        chunks_indexed (int): Number of text chunks added to the FAISS vector store.
        filename (str): Name of the processed file.
    """
    chunks_indexed: int
    filename: str


class HealthResponse(BaseModel):
    """
    Response schema for the /health endpoint.

    Attributes:
        status (str): Service status. Always 'ok' if the service is running.
        vectorstore (bool): Whether the FAISS vector store exists on disk.
        documents (int): Number of documents currently indexed in memory.
    """
    status: str
    vectorstore: bool
    documents: int


class DocumentsResponse(BaseModel):
    """
    Response schema for the /documents endpoint.

    Attributes:
        documents (list[str]): List of filenames currently indexed in the vector store.
    """
    documents: list[str]
