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


