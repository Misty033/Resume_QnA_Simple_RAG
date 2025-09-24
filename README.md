# Resume QA System

A **simple RAG-based Resume Question-Answering system** that allows users to upload PDF resumes and ask questions about their content.  
It uses **PDF extraction**, **text cleaning & chunking**, **FAISS semantic search**, and **Ollama LLM** for generating answers. **Langfuse** is integrated for monitoring and logging. **Streamlit** is used for frontend UI.

---

## Features

- Upload PDF resumes and extract clean text.
- Split text into semantic chunks using `langchain.text_splitter`.
- Store embeddings in FAISS vectorstore for efficient similarity search.
- Query the resume content using an LLM (Ollama: model_name = "gemma3:27b").
- Track operations and LLM interactions using Langfuse.
- Streamlit interface with multi-question session support.

---

## Demo

Watch a demo of the Resume QA System:

<video width="480" controls>
  <source src="demo_videos/demo_version1.webm" type="video/webm">
  Your browser does not support the video tag.
</video>

---

## Tech Stack

- **PDF Extraction:** pdfplumber  
- **Text Processing & Chunking:** langchain.text_splitter  
- **Vector Database:** FAISS  
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`  
- **LLM for QA:** Ollama  
- **Monitoring / Logging:** Langfuse
-  **Frontend / UI:** Streamlit   
- **Environment Management:** Python `.env` + python-dotenv  

## Installation

```bash
# Clone the repo
git clone https://github.com/Misty033/Resume_QnA_Simple_RAG.git
cd Resume_QnA_Simple_RAG

# Create virtual environment (Linux/macOS)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```
## Usage

```bash
streamlit run app1.py

```

## Author
Misty Roy

