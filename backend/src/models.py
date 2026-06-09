from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

class IngestResponse(BaseModel):
    chunks_indexed: int
    filename: str

class HealthResponse(BaseModel):
    status: str
    vectorstore: bool
    documents: int

class DocumentsResponse(BaseModel):
    documents: list[str]
