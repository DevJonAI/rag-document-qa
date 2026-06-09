import streamlit as st
import requests

API_URL = 'http://backend:8000'

st.set_page_config(page_title='RAG Document Q&A', page_icon='📄')
st.title('RAG Document Q&A')

st.header('1. Upload a document')
uploaded = st.file_uploader('Select a file', type=['pdf', 'txt', 'docx'])

if uploaded and st.button('Index document'):
    with st.spinner('Processing document...'):
        r = requests.post(
            f'{API_URL}/ingest',
            files={'file': (uploaded.name, uploaded.getvalue())}
        )
        if r.status_code == 200:
            st.success(f'Indexed {r.json()["chunks_indexed"]} chunks from "{uploaded.name}"')
        else:
            st.error(f'Error: {r.json()}')

with st.expander('View indexed documents'):
    try:
        r = requests.get(f'{API_URL}/documents')
        docs = r.json().get('documents', [])
        if docs:
            for doc in docs:
                st.write(f'- {doc}')
        else:
            st.write('No documents indexed yet.')
    except Exception:
        st.write('Backend not reachable.')

if st.button('Clear vector store'):
    requests.delete(f'{API_URL}/vectorstore')
    st.warning('Vector store cleared.')

st.divider()

st.header('2. Ask questions')

if 'messages' not in st.session_state:
    st.session_state.messages = []

if st.button('Clear conversation'):
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if question := st.chat_input('Ask a question about your documents...'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    st.chat_message('user').write(question)
    with st.spinner('Generating answer...'):
        r = requests.post(f'{API_URL}/query', json={'question': question})
        if r.status_code == 200:
            data = r.json()
            answer = data['answer']
            sources = data.get('sources', [])
            st.session_state.messages.append({'role': 'assistant', 'content': answer})
            st.chat_message('assistant').write(answer)
            with st.expander('Sources used'):
                for i, src in enumerate(sources):
                    st.write(f'Fragment {i+1}: {src}')
        else:
            st.error(f'Error: {r.json()}')
