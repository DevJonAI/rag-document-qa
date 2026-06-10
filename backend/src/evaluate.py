from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import anthropic
import os


VECTORSTORE_PATH = './vectorstore'


def run_evaluation() -> dict:
    """
    Run RAGAS evaluation on the RAG pipeline using a predefined test dataset.

    Evaluates four metrics:
        - faithfulness: whether the answer is grounded in the retrieved context
        - answer_relevancy: whether the answer addresses the question
        - context_precision: whether retrieved chunks are relevant to the question
        - context_recall: whether all necessary information was retrieved

    Returns:
        dict: Dictionary with metric names as keys and float scores as values.
    """
    # Test dataset — questions and reference answers based on the indexed document
    test_cases = [
        {
            'question': 'What is RAG?',
            'ground_truth': 'RAG is a technique that combines a retrieval system that locates relevant text fragments with a generative model that produces coherent responses using those fragments as context.'
        },
        {
            'question': 'What vector stores does the project support?',
            'ground_truth': 'The project supports FAISS and ChromaDB as vector stores.'
        },
        {
            'question': 'What document formats does the system accept?',
            'ground_truth': 'The system accepts PDF, TXT, and DOCX document formats.'
        },
        {
            'question': 'What is the recommended chunk size?',
            'ground_truth': 'The recommended chunk size is between 500 and 1000 tokens with an overlap of 100 to 200 tokens.'
        },
    ]

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for case in test_cases:
        question = case['question']
        docs = retriever.invoke(question)
        context = [doc.page_content for doc in docs]

        prompt = (
            'Answer the following question based exclusively on the provided context.\n\n'
            'Context:\n' + '\n\n'.join(context) + '\n\n'
            'Question: ' + question + '\n\nAnswer:'
        )

        response = client.messages.create(
            model='claude-sonnet-4-5',
            max_tokens=512,
            messages=[{'role': 'user', 'content': prompt}]
        )

        questions.append(question)
        answers.append(response.content[0].text)
        contexts.append(context)
        ground_truths.append(case['ground_truth'])

    dataset = Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths,
    })

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    return dict(results)


if __name__ == '__main__':
    scores = run_evaluation()
    print('\nRAGAS Evaluation Results')
    print('=' * 40)
    for metric, score in scores.items():
        print(f'{metric}: {score:.4f}')