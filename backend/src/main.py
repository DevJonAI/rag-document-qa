from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from ingestion import ingest_document
from retrieval import query_rag
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
    """Check the health status of the service."""
    vs_exists = os.path.exists(VECTORSTORE_PATH)
    return HealthResponse(status='ok', vectorstore=vs_exists, documents=len(indexed_docs))

@app.get('/documents', response_model=DocumentsResponse)
def get_documents():
    """Retrieve the list of indexed documents."""
    return DocumentsResponse(documents=indexed_docs)

@app.post('/ingest', response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload and index a document into the vector store."""
    allowed = ['.pdf', '.txt', '.docx']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, 'Unsupported format. Use PDF, TXT or DOCX.')
    tmp_path = f'./tmp_{file.filename}'
    with open(tmp_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    result = ingest_document(tmp_path)
    os.remove(tmp_path)
    indexed_docs.append(file.filename)
    return IngestResponse(chunks_indexed=result['chunks_indexed'], filename=file.filename)

@app.post('/query', response_model=QueryResponse)
def query(request: QueryRequest):
    """Perform a RAG query against the indexed documents."""
    if not os.path.exists(VECTORSTORE_PATH):
        raise HTTPException(400, 'No documents indexed yet. Upload a document first.')
    result = query_rag(request.question)
    return QueryResponse(answer=result['answer'], sources=result['sources'])

@app.delete('/vectorstore')
def clear_vectorstore():
    """Delete all indexed documents and reset the vector store."""
    global indexed_docs
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
    indexed_docs = []
    return {'message': 'Vector store cleared successfully'}
