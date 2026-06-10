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

try:
    r = requests.get(f'{API_URL}/documents')
    docs = r.json().get('documents', [])
except Exception:
    docs = []

filter_options = ['All documents'] + docs
selected = st.selectbox('Search in:', filter_options)
filter_document = None if selected == 'All documents' else selected

streaming_mode = st.toggle('Streaming mode', value=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

if st.button('Clear conversation'):
    requests.delete(f'{API_URL}/memory')
    st.session_state.messages = []
    st.rerun()

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if question := st.chat_input('Ask a question about your documents...'):
    st.session_state.messages.append({'role': 'user', 'content': question})
    st.chat_message('user').write(question)

    payload = {'question': question}
    if filter_document:
        payload['filter_document'] = filter_document

    if streaming_mode:
        with st.chat_message('assistant'):
            placeholder = st.empty()
            full_answer = ''
            sources = []

            with requests.post(f'{API_URL}/stream', json=payload, stream=True) as r:
                skip_next_data = False
                for line in r.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode('utf-8')
                    if decoded.startswith('event: sources') or decoded.startswith('event: done'):
                        skip_next_data = True
                        continue
                    if skip_next_data:
                        skip_next_data = False
                        continue
                    if decoded.startswith('data: '):
                        token = decoded[6:]
                        full_answer += token
                        display = full_answer.replace('\\n', '\n').replace('\\t', '\t')
                        placeholder.markdown(display + '▌')

            placeholder.markdown(full_answer.replace('\\n', '\n').replace('\\t', '\t'))
            st.session_state.messages.append({'role': 'assistant', 'content': full_answer})

    else:
        with st.spinner('Generating answer...'):
            r = requests.post(f'{API_URL}/query', json=payload)
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