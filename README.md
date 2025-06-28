
# Advanced Hybrid RAG Chatbot

This project is a sophisticated, full-stack conversational AI chatbot designed to interact with a private collection of documents. It leverages an advanced Retrieval-Augmented Generation (RAG) architecture that goes beyond standard vector search to deliver highly accurate, context-aware, and low-latency answers.

The core of this system is a **two-stage hybrid retrieval pipeline** that combines keyword and semantic search for broad recall, followed by a precision-focused reranking step to ensure only the most relevant information is passed to the Large Language Model.

![Screenshot (46)](https://github.com/user-attachments/assets/c2fe1009-4808-487c-bb88-3350b002188b)

## 🚀 Key Features

-   **Hybrid Retrieval:** Combines sparse (BM25) and dense (FAISS) retrievers to get the best of both keyword and semantic search.
-   **Cross-Encoder Reranking:** Implements a second-stage reranker to drastically improve the precision of the retrieved context before generation.
-   **History-Aware Conversations:** Remembers previous turns in the conversation to understand follow-up questions and maintain context.
-   **Multi-Format Document Support:** Ingests and processes both PDF (`.pdf`) and Microsoft Word (`.docx`) files.
-   **Interactive Web Interface:** Built with Streamlit for an intuitive user experience, including file uploading and a real-time chat window.
-   **High-Performance Generation:** Utilizes the Groq API for near-instantaneous LLM responses, ensuring a smooth user interaction.

## 🏗️ Project Architecture

The system is composed of two main pipelines: an offline **Indexing Pipeline** and a real-time **Querying Pipeline**.

#### 1. Indexing Pipeline (Offline Processing)

```
[PDFs/DOCX in Folder] -> [Document Loader] -> [Recursive Text Splitter] -> [Text Chunks]
                                                                              |
                                        +-------------------------------------+------------------------------------+
                                        |                                                                          |
                              [HuggingFace Embeddings]                                                       [BM25 Algorithm]
                                        |                                                                          |
                           [FAISS Vector Store (Semantic)]                                              [BM25 Index (Keyword)]
```

#### 2. Querying Pipeline (Real-time Interaction)

```
[User Query + Chat History]
          |
[1. Query Reformulation (LLM)] -> (Creates a standalone query)
          |
[2. Hybrid Retrieval] -> (Queries both FAISS & BM25) -> (Returns top K candidate chunks)
          |
[3. Reranking] -> (Cross-Encoder scores and re-orders candidates) -> (Returns top N most relevant chunks)
          |
[4. Final Context + Final Query] -> [LLM (Groq)]
          |
[5. Final Answer] -> [User Interface]
```

## 🛠️ Tech Stack

-   **Core Frameworks:** `LangChain`, `Streamlit`
-   **LLM & Generation:** `Groq API` (with Llama3-70b)
-   **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
-   **Vector Store:** `FAISS` (Facebook AI Similarity Search)
-   **Sparse Retriever:** `rank_bm25`
-   **Reranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
-   **Document Loading:** `PyPDF`, `python-docx2txt`
-   **Deployment:** `Streamlit Cloud` / `Google Colab` + `ngrok`

## ⚙️ Setup and Installation

Follow these steps to get the application running locally.

#### Prerequisites

-   Python 3.9+
-   A [Groq API Key](https://console.groq.com/keys)

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hybrid-rag-chatbot.git
cd hybrid-rag-chatbot
```

#### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For Unix/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Configure API Keys

The application uses Streamlit's secrets management. Create a `.streamlit` directory and a `secrets.toml` file within it.

```
your-project-directory/
├── .streamlit/
│   └── secrets.toml
├── app.py
└── ...
```

Add your Groq API key to the `secrets.toml` file:

```toml
# .streamlit/secrets.toml
GROQ_API_KEY="gsk_..."
```

#### 5. Run the Application

```bash
streamlit run app.py
```

Your browser should automatically open to the application running on `localhost`.

## 📖 Usage

1.  **Launch the application** using the command above.
2.  Use the **sidebar** to upload one or more PDF or DOCX files.
3.  The application will automatically begin **processing the documents**. A spinner will indicate progress.
4.  Once processing is complete, a message will appear in the chat window.
5.  **Start asking questions!** Type your query into the chat input at the bottom of the screen.

## 💡 How It Works in Detail

### Indexing

1.  **Document Loading:** `PyPDFLoader` and `Docx2txtLoader` are used to extract raw text from the uploaded files.
2.  **Chunking:** The text is split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`. This method is superior to fixed-size chunking as it attempts to preserve semantic boundaries (paragraphs, sentences). An overlap is maintained between chunks to ensure context is not lost.
3.  **Dual Indexing:**
    -   **FAISS (Dense):** Each chunk is converted into a vector embedding using the `all-MiniLM-L6-v2` model. These vectors are stored in a FAISS index for efficient semantic similarity search.
    -   **BM25 (Sparse):** A separate BM25 index is created from the text chunks for efficient keyword-based retrieval.

### Querying

1.  **Query Reformulation:** The system first takes the user's latest query and the chat history. It uses an LLM to reformulate this into a self-contained question that can be understood without the prior conversation. For example, "What about its main feature?" becomes "What is the main feature of the Hybrid RAG system?".
2.  **Hybrid Retrieval (Recall):** The reformulated query is sent to the `EnsembleRetriever`, which queries both the FAISS and BM25 indexes simultaneously. This fetches a diverse set of candidate documents—some that are semantically similar and others that match exact keywords. This "recall-focused" step ensures we don't miss any potentially relevant information.
3.  **Reranking (Precision):** The list of candidate chunks is then passed to a `CrossEncoderReranker`. The cross-encoder model takes the query and each candidate chunk *together* to compute a highly accurate relevance score. This computationally intensive but precise step filters and reorders the chunks, ensuring only the top 3-5 most relevant ones proceed.
4.  **Generation:** The final, reranked context is formatted and passed along with the query to the `llama3-70b` model via the Groq API, which generates the final, concise answer.

## 📈 Potential Future Improvements

-   **Parent Document Retriever:** Implement a strategy to retrieve the most relevant small chunks but provide the larger "parent" document (the full paragraph or section) to the LLM. This would give the model more context for reasoning without sacrificing retrieval precision.
-   **Advanced Document Parsing:** Integrate a document intelligence model (like Nougat or LayoutLM) to parse tables, figures, and complex layouts from PDFs and store them as structured data, enabling more precise answers to questions about that content.
-   **Automated Evaluation Suite:** Build a rigorous evaluation pipeline using a framework like RAGAs to continuously measure metrics like context precision, faithfulness, and answer relevance, allowing for data-driven improvements to the system.
