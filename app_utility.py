import os
from langchain_community.llms import Ollama
from langchain_community.vectorstores import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# Set working directory (same as main.py)
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load the LLAMA 3.1 model with temperature 0 for determinism
llm = Ollama(
    model = 'llama3.1',
    temperature=0
)

# Create HuggingFace Embeddings instance
embeddings = HuggingFaceEmbeddings()

def get_answer(file_name,query):
    """
    This function processes a PDF document, extracts relevant information
    based on the user query using a Retrieval-Augmented Generation (RAG) model,
    and returns the answer.

    Args:
        file_name (str): Name of the uploaded PDF file.
        query (str): User's question about the document content.

    Returns:
        str: Answer extracted from the document based on the query.
    """

    # Construct the full file path
    file_path = f"{working_dir}/{file_name}"

    # Load the PDF document
    loader = PyPDFLoader(file_path)

    documents = loader.load()

    # Text splitter configuration for splitting the document
    text_splitter = CharacterTextSplitter(separator='/n',chunk_size = 1000, chunk_overlap=200)

    # Split the document into text chunks
    text_chunks = text_splitter.split_documents(documents)

    # Generate embeddings for the text chunks
    knowledge_base = FAISS.from_documents(text_chunks, embeddings)


    # Create a retrieval QA chain using the LLAM and FAISS vector store
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=knowledge_base.as_retriever())


    # Invoke the QA chain to retrieve and generate an answer
    response = qa_chain.invoke({"query": query})

    # Return the extracted answer
    return response['result']