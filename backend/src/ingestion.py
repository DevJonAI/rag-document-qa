from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

VECTORSTORE_PATH = './vectorstore'


def ingest_document(file_path: str) -> dict:
    """
    Load, chunk, embed and index a document into the FAISS vector store.

    Supports PDF, TXT and DOCX formats. If a vector store already exists,
    the new chunks are added to it. Otherwise a new index is created.

    Args:
        file_path (str): Absolute or relative path to the document file.

    Returns:
        dict: A dictionary with a single key:
            - 'chunks_indexed' (int): Number of text chunks added to the index.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif ext == 'docx':
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f'Unsupported format: {ext}')

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    if os.path.exists(VECTORSTORE_PATH):
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)
    return {'chunks_indexed': len(chunks)}
