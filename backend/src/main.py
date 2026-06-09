from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from ingestion import ingest_document
from retrieval import query_rag, clear_memory
from models import QueryRequest, QueryResponse, IngestResponse, HealthResponse, DocumentsResponse

app = FastAPI(title='RAG API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

VECTORSTORE_PATH = './vectorstore'
indexed_docs = []


@app.get('/health', response_model=HealthResponse)
def health():
    """
    Check the health status of the service and the vector store.

    Returns:
        HealthResponse: Service status, whether the vector store exists, and document count.
    """
    vs_exists = os.path.exists(VECTORSTORE_PATH)
    return HealthResponse(status='ok', vectorstore=vs_exists, documents=len(indexed_docs))


@app.get('/documents', response_model=DocumentsResponse)
def get_documents():
    """
    Retrieve the list of documents currently indexed in the vector store.

    Returns:
        DocumentsResponse: Dictionary with list of indexed filenames.
    """
    return DocumentsResponse(documents=indexed_docs)


@app.post('/ingest', response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload and process a document, indexing it into the vector store.

    Each chunk is stored with the source filename as metadata to enable
    document-level filtering at query time.

    Args:
        file (UploadFile): The document file to ingest. Supported: PDF, TXT, DOCX.

    Returns:
        IngestResponse: Number of chunks indexed and the filename.

    Raises:
        HTTPException: If the file format is not supported.
    """
    allowed = ['.pdf', '.txt', '.docx']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, 'Unsupported format. Use PDF, TXT or DOCX.')
    tmp_path = f'./tmp_{file.filename}'
    with open(tmp_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    result = ingest_document(tmp_path, file.filename)
    os.remove(tmp_path)
    if file.filename not in indexed_docs:
        indexed_docs.append(file.filename)
    return IngestResponse(chunks_indexed=result['chunks_indexed'], filename=file.filename)


@app.post('/query', response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Perform a RAG query against the indexed documents.

    Optionally filters retrieval to a specific document using the
    filter_document field in the request body.

    Args:
        request (QueryRequest): Request body containing the question and optional document filter.

    Returns:
        QueryResponse: Generated answer and source document excerpts.

    Raises:
        HTTPException: If no documents have been indexed yet.
    """
    if not os.path.exists(VECTORSTORE_PATH):
        raise HTTPException(400, 'No documents indexed yet. Upload a document first.')
    result = query_rag(request.question, request.filter_document)
    return QueryResponse(answer=result['answer'], sources=result['sources'])


@app.delete('/memory')
def clear_conversation_memory():
    """
    Reset the conversation memory without clearing the vector store.

    Returns:
        dict: Confirmation message.
    """
    clear_memory()
    return {'message': 'Conversation memory cleared successfully'}


@app.delete('/vectorstore')
def clear_vectorstore():
    """
    Delete all indexed documents and reset the vector store and memory.

    Returns:
        dict: Confirmation message.
    """
    global indexed_docs
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
    indexed_docs = []
    clear_memory()
    return {'message': 'Vector store cleared successfully'}
