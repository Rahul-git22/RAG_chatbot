import tempfile
import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

lm = Ollama(
    model = 'llama3.1',
    temperature=0
)

embeddings = HuggingFaceEmbeddings()

def get_answer(uploaded_file,query):

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name
    
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(separator='/n',chunk_size = 1000, chunk_overlap=200)

    text_chunks = text_splitter.split_documents(documents)

    knowledge_base = FAISS.from_documents(text_chunks, embeddings)


    # retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=knowledge_base.as_retriever())

    response = qa_chain.invoke({"query": query})

    return response['result']




st.set_page_config(
    page_title="Chat with Doc",
    page_icon="0",
    layout="centered"
)

st.title('Document Q&A LLAMA 3.1')

uploaded_file = st.file_uploader(label='Upload your file',type = ["pdf"])

user_query = st.text_input('Ask your Question')

if st.button("Run"):
    answer = get_answer(uploaded_file, user_query)

    st.success(answer)  