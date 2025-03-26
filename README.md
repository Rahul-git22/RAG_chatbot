# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, Pinecone, and Groq. This chatbot leverages vector embeddings and a vector database to provide context-aware responses to user queries.

## Overview

This project demonstrates how to build a RAG chatbot using:
- **LangChain**: For managing documents, embeddings, and retrieval.
- **Pinecone**: As the vector database for storing and querying embeddings.
- **Groq**: For the LLM to generate responses.
- **Streamlit**: For creating an interactive web interface.

The chatbot can load documents (PDF and DOCX), split them into chunks, create embeddings, and store them in Pinecone. It then uses these embeddings to retrieve relevant context for answering user queries.

## Project Flow

1. **Install Required Dependencies:**
   - Install all necessary Python packages from the `requirements.txt` file.

2. **Initialize Environment Variables:**
   - Set environment variables for Groq and Pinecone API keys.

3. **Initialize the LLM and Vector Store:**
   - Initialize the Groq LLM and Pinecone vector store.
   - Load documents, split them into chunks, and create embeddings.
   - Store the embeddings in Pinecone.

4. **Create Retrieval Chain:**
   - Create a retrieval chain to search the vector store and retrieve relevant context.

5. **Create QA Chain:**
   - Create a QA chain to generate responses based on the retrieved context.

6. **Run Streamlit App:**
   - Run the Streamlit app to interact with the chatbot via a web interface.

## Prerequisites

Before you begin, ensure you have the following:
- Python 3.11 or higher
- API keys for Groq and Pinecone
- A Pinecone index created and accessible

## Running the Streamlit App
Install Streamlit:
```bash
pip install streamlit
```

Run the Streamlit App
```bash
streamlit run app.py
```
