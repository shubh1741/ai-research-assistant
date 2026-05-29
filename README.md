# AI Research Assistant

A fully local RAG (Retrieval-Augmented Generation) based AI Research Assistant built using LangChain, ChromaDB, Ollama, and Llama3.

This project allows users to upload and query PDF documents using a conversational AI interface powered by local LLMs.

---

# Features

* PDF document ingestion
* Intelligent text chunking
* Semantic search using embeddings
* Vector database storage with ChromaDB
* Local LLM inference using Ollama + Llama3
* Streamlit chat interface
* Fully local execution (no paid APIs required)
* Retrieval-Augmented Generation (RAG)

---

# Tech Stack

| Component            | Technology                             |
| -------------------- | -------------------------------------- |
| Programming Language | Python                                 |
| LLM Framework        | LangChain                              |
| Vector Database      | ChromaDB                               |
| Embedding Model      | sentence-transformers/all-MiniLM-L6-v2 |
| Local LLM Runtime    | Ollama                                 |
| Language Model       | Llama3                                 |
| Frontend             | Streamlit                              |
| PDF Parsing          | PyPDFLoader                            |

---

# Project Architecture

```text
PDF Documents
      ↓
Document Loader
      ↓
Text Chunking
      ↓
Embedding Generation
      ↓
Chroma Vector Database
      ↓
Retriever
      ↓
Llama3 via Ollama
      ↓
Streamlit Chat Interface
```

---

# What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that combines:

* Retrieval systems
* Large Language Models

Instead of relying only on the model’s internal knowledge, relevant information is first retrieved from external documents and then passed to the LLM as context.

This improves:

* Accuracy
* Context awareness
* Reduced hallucinations
* Domain-specific answering

---

# Folder Structure

```text
ai-research-assistant/
│
├── app/
│   ├── ingest.py
│   ├── rag.py
│   └── main.py
│
├── data/
│   └── sample.pdf
│
├── chroma_db/
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/shubh1741/ai-research-assistant.git
cd ai-research-assistant
```

---

## 2. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Ollama

Download and install:

[https://ollama.com/](https://ollama.com/)

Pull Llama3 model:

```bash
ollama pull llama3
```

Run Ollama:

```bash
ollama run llama3
```

---

# Add PDF Documents

Place your PDF inside:

```text
/data
```

---

# Create Vector Database

Run:

```bash
python app/ingest.py
```

This:

* loads PDFs
* splits text into chunks
* generates embeddings
* stores vectors in ChromaDB

---

# Run the Application

```bash
python -m streamlit run app/main.py --server.fileWatcherType none
```

Open browser:

```text
http://localhost:8501
```

---

# Example Questions

* What is this paper about?
* Summarize the document.
* Explain the methodology.
* What are the key findings?
* What limitations are discussed?

---

# Key Concepts Learned

## 1. Retrieval-Augmented Generation (RAG)

Understanding how LLMs can use external knowledge.

## 2. Embeddings

Converting text into high-dimensional vectors for semantic search.

## 3. Vector Databases

Using ChromaDB to store and retrieve embeddings efficiently.

## 4. Semantic Search

Finding contextually relevant information instead of exact keyword matches.

## 5. Local LLM Inference

Running Llama3 locally using Ollama without external APIs.

## 6. LangChain Pipelines

Building AI workflows using retrievers, prompts, chains, and parsers.

## 7. Streamlit UI

Creating interactive AI applications quickly.

---

# Challenges Faced

During development, several real-world engineering issues were handled:

* Python dependency conflicts
* LangChain version migrations
* Streamlit environment issues
* Mac + Anaconda conflicts
* Local model setup
* Vector database integration
* GitHub authentication with PAT tokens

---

# Future Improvements

* Multi-PDF support
* Citation highlighting
* Chat memory
* Agentic workflows
* Web search integration
* Cloud deployment
* Docker support
* Authentication system
* Hybrid search
* Reranking pipeline

---

# Resume Impact

This project demonstrates:

* Applied AI engineering
* LLM application development
* Retrieval systems
* Vector databases
* Local model deployment
* End-to-end AI workflow building

---

# Author

## Shubham Sahu

GitHub:

[https://github.com/shubh1741](https://github.com/shubh1741)
