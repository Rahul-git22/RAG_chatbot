import streamlit as st
import os
import tempfile
from typing import List

# Import all necessary LangChain components from your notebook
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")

# --- API KEY and MODEL INITIALIZATION ---
try:
    # Attempt to get the API key from Streamlit secrets
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    # Fallback for local development
    st.warning("GROQ_API_KEY not found in Streamlit secrets. Please enter it below.")
    GROQ_API_KEY = st.text_input("Enter your GROQ API Key:", type="password")

if not GROQ_API_KEY:
    st.info("Please provide a GROQ API Key to proceed.")
    st.stop()

os.environ['GROQ_API_KEY'] = GROQ_API_KEY

@st.cache_resource
def get_llm():
    """Initializes and caches the ChatGroq LLM."""
    return ChatGroq(model="llama3-70b-8192", temperature=0)

llm = get_llm()

# --- CORE RAG FUNCTIONS ---

def load_documents_from_uploads(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Document]:
    """Loads documents from Streamlit uploaded files into a temporary directory."""
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
            elif uploaded_file.name.endswith('.docx'):
                loader = Docx2txtLoader(temp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            
    return documents

def build_rag_chain(uploaded_files):
    """
    Builds the full RAG chain from uploaded documents.
    This function encapsulates loading, splitting, embedding, and chaining.
    """
    # 1. Load Documents
    documents = load_documents_from_uploads(uploaded_files)
    if not documents:
        st.error("No documents could be loaded. Please check the files and try again.")
        return None

    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    splits = text_splitter.split_documents(documents)

    # 3. Create Hybrid Retriever
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embedding_model)
    
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10
    
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    # 4. Add Reranker
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid_retriever)

    # 5. Create History-Aware Chain
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, contextualize_q_prompt)
    
    # 6. Create Answering Chain
    qa_system_prompt = (
        "You are an expert assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Be concise and helpful."
        "\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- STREAMLIT UI ---
st.title("📄 Hybrid RAG Chatbot")
st.write("Upload your documents (PDF, DOCX) and chat with them using an advanced RAG pipeline.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lc_chat_history" not in st.session_state:
    st.session_state.lc_chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Sidebar for controls ---
with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a moment."):
                st.session_state.rag_chain = build_rag_chain(uploaded_files)
                if st.session_state.rag_chain:
                    st.session_state.messages = [{"role": "assistant", "content": "Documents processed! How can I help you?"}]
                    st.session_state.lc_chat_history = []
                    st.success("Processing complete!")
                else:
                    st.error("Failed to process documents.")
    
    if st.session_state.rag_chain:
        if st.button("Clear Chat History"):
            st.session_state.messages = [{"role": "assistant", "content": "History cleared. Ask me a new question!"}]
            st.session_state.lc_chat_history = []
            st.rerun()

# --- Main chat interface ---
if not st.session_state.rag_chain:
    st.info("Please upload your documents and click 'Process Documents' to begin.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.invoke({
                    "input": prompt, 
                    "chat_history": st.session_state.lc_chat_history
                })
                response = result['answer']
                st.markdown(response)

                # Update histories
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.lc_chat_history.extend([HumanMessage(content=prompt), AIMessage(content=response)])