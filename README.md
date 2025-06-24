# 📄 Chat with Document (RAG System using PostgreSQL + pgvector)

This project implements a **Retrieval-Augmented Generation (RAG)** system for PDF documents using **PostgreSQL with the pgvector extension** to store and retrieve semantic embeddings.

Users can upload a scientific or technical PDF and ask questions. The system retrieves relevant context from the document using **semantic similarity** and passes it to a language model (LLM) to generate accurate responses.

---

## 🧠 Key Features

- 📄 **PDF Upload Only** – Parse and embed PDF documents.
- 🔍 **RAG Pipeline** – Retrieve relevant chunks from documents to enhance LLM answers.
- 🧠 **LLM Integration** – Used ollama's llama3.2 to generate responses.
- 💾 **PostgreSQL + pgvector** – Store vector embeddings in a PostgreSQL database.
- 🐧 **Ubuntu VirtualBox** – PostgreSQL backend hosted in an Ubuntu VM.

---

## 🧱 Tech Stack

| Component         | Technology                |
|------------------|---------------------------|
| Document Parsing | PyMuPDF / pdfplumber      |
| Embedding Model  | HuggingFace               |
| Vector Store     | PostgreSQL + pgvector     |
| LLM              | LLaMA3.2                  |
| UI               | Streamlit / Flask         |
| Deployment       | Ubuntu VirtualBox         |

---

## 🗃️ Architecture Overview

1. User uploads a PDF in a folder, every pdf is read
2. Document is split into chunks and converted into vector embeddings after preprocessing
3. Embeddings are stored in **PostgreSQL with pgvector**
4. On user query:
    - Query is embedded
    - Similar chunks are retrieved via vector similarity search
    - Context + query is sent to the LLM for a final answer

---

## 🐘 PostgreSQL + pgvector Setup

1. **Install PostgreSQL**:

    ```bash
    sudo apt update
    sudo apt install postgresql postgresql-contrib
    ```

2. **Install pgvector Extension**:

    ```bash
    sudo -u postgres psql
    CREATE EXTENSION vector;
    ```

3. **Create a Table**:

    ```sql
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(1536) -- adjust depending on embedding size
    );
    ```

4. **Insert Embeddings via Python** (example):

    ```python
    import psycopg2
    from pgvector.psycopg2 import register_vector
    conn = psycopg2.connect(...)
    register_vector(conn)

    cursor.execute("INSERT INTO documents (content, embedding) VALUES (%s, %s)", (chunk_text, embedding.tolist()))
    ```

---

## 🚀 Getting Started

1. **Clone the repo**:

    ```bash
    git clone https://github.com/keya714/ChatwithDocument.git
    cd ChatwithDocument
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set environment variables**:

    ```bash
    export OPENAI_API_KEY=your_key_here
    export DATABASE_URL=postgresql://user:pass@localhost/dbname
    ```

4. **Run the app** (example with Streamlit):

    ```bash
    streamlit run app.py
    ```

5. **Upload a PDF and start chatting!**

---

## 📁 Project Structure
.
├── app.py # Main app (Streamlit or Flask)
├── db/ # PostgreSQL connection and queries
├── utils/ # PDF parsing and embedding functions
├── requirements.txt
└── README.md
