# Import required libraries
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore

# Available Groq models
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "qwen-2.5-32b", 
    "deepseek-r1-distill-qwen-32b"
]

# Set page configuration
st.set_page_config(
    page_title="RAG Chatbot with Pinecone",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = GROQ_MODELS[0]

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # Model selection
    st.session_state.selected_model = st.selectbox(
        "Select Groq Model",
        GROQ_MODELS,
        index=GROQ_MODELS.index(st.session_state.selected_model)
    )
    
    # API keys
    groq_api_key = st.text_input("Groq API Key", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    os.environ['GROQ_API_KEY'] = groq_api_key
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    
    # Document upload
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )
    
    # Pinecone index name
    index_name = st.text_input("Pinecone Index Name", "langchain-rag-index")
    
    # Initialize system button
    if st.button("Initialize System"):
        if not groq_api_key or not pinecone_api_key:
            st.error("Please provide both API keys")
        elif not uploaded_files:
            st.error("Please upload at least one document")
        else:
            with st.spinner("Initializing system..."):
                try:
                    # Save uploaded files temporarily
                    temp_dir = "./temp_docs"
                    os.makedirs(temp_dir, exist_ok=True)
                    for file in uploaded_files:
                        file_path = os.path.join(temp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                    
                    # Load documents
                    def load_documents(folder_path: str) -> List[Document]:
                        documents = []
                        for filename in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, filename)
                            if filename.endswith('.pdf'):
                                loader = PyPDFLoader(file_path)
                            elif filename.endswith('.docx'):
                                loader = Docx2txtLoader(file_path)
                            else:
                                continue
                            documents.extend(loader.load())
                        return documents
                    
                    documents = load_documents(temp_dir)
                    
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    # Initialize embeddings and Pinecone
                    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    
                    # Create Pinecone vector store
                    vectorstore = PineconeVectorStore.from_documents(
                        documents=splits,
                        embedding=embedding_function,
                        index_name=index_name
                    )
                    
                    # Create retriever
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                    
                    # Initialize LLM with selected model
                    llm = ChatGroq(
                        model=st.session_state.selected_model,
                        temperature=0
                    )
                    
                    # Create contextual question chain
                    contextualize_q_system_prompt = """
                    Given a chat history and the latest user question
                    which might reference context in the chat history,
                    formulate a standalone question which can be understood
                    without the chat history. Do NOT answer the question,
                    just reformulate it if needed and otherwise return it as is.
                    """
                    
                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    
                    # Create history-aware retriever
                    history_aware_retriever = create_history_aware_retriever(
                        llm, retriever, contextualize_q_prompt
                    )
                    
                    # Create QA chain
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
                        ("system", "Context: {context}"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}")
                    ])
                    
                    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                    
                    # Store in session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = retriever
                    st.session_state.rag_chain = rag_chain
                    st.session_state.messages = []
                    
                    st.success(f"System initialized successfully with {st.session_state.selected_model}!")
                    
                except Exception as e:
                    st.error(f"Error initializing system: {str(e)}")

# Main chat interface
st.title("🤖 RAG Chatbot with Pinecone")

# Display current model
st.caption(f"Current model: {st.session_state.selected_model}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    if not st.session_state.rag_chain:
        st.error("Please initialize the system first in the sidebar")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare chat history for the chain
        chat_history = []
        for msg in st.session_state.messages[:-1]:  # all messages except the current one
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        
        # Get response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })['answer']
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

# Add clear chat button
if st.session_state.messages and st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()