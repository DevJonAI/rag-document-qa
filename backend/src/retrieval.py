from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
import os

VECTORSTORE_PATH = './vectorstore'

def query_rag(question: str) -> dict:
    """
    Execute the full RAG pipeline for a given question.
    """
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    llm = ChatAnthropic(
        model='claude-sonnet-4-5',
        temperature=0,
        api_key=os.environ.get('ANTHROPIC_API_KEY')
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({'query': question})

    return {
        'answer': result['result'],
        'sources': [doc.page_content[:200] for doc in result['source_documents']]
    }
