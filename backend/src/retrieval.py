from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from typing import Optional

VECTORSTORE_PATH = './vectorstore'

# Global memory instance shared across requests
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)


def query_rag(question: str, filter_document: Optional[str] = None) -> dict:
    """
    Execute the full RAG pipeline for a given question with conversation memory.

    Loads the FAISS vector store, optionally filters chunks by source document,
    retrieves the 4 most relevant chunks, and passes them along with the
    conversation history to Claude.

    Args:
        question (str): The natural language question asked by the user.
        filter_document (Optional[str]): If provided, restricts retrieval to chunks
            from this specific document filename. If None, searches all documents.

    Returns:
        dict: A dictionary with two keys:
            - 'answer' (str): The generated answer from Claude.
            - 'sources' (list[str]): List of up to 4 text excerpts (first 200 chars each)
              used as context to generate the answer.

    Raises:
        Exception: If the vector store does not exist or the Anthropic API call fails.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Apply document filter if specified
    if filter_document:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                'k': 4,
                'filter': {'source_filename': filter_document}
            }
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    llm = ChatAnthropic(
        model='claude-sonnet-4-5',
        temperature=0,
        api_key=os.environ.get('ANTHROPIC_API_KEY')
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )

    result = qa_chain.invoke({'question': question})

    return {
        'answer': result['answer'],
        'sources': [doc.page_content[:200] for doc in result['source_documents']]
    }


def clear_memory() -> None:
    """
    Reset the conversation memory.

    Clears all stored chat history so the next query starts
    a fresh conversation without context from previous exchanges.
    """
    memory.clear()
