# Resume QA System

A **Streamlit-based Resume Question-Answering system** that allows users to upload PDF resumes and ask questions about their content.  
It uses **PDF extraction**, **text cleaning & chunking**, **FAISS semantic search**, and **Ollama LLM** for generating answers. **Langfuse** is integrated for monitoring and logging.

---

## Features

- Upload PDF resumes and extract clean text.
- Split text into semantic chunks using `langchain.text_splitter`.
- Store embeddings in FAISS vectorstore for efficient similarity search.
- Query the resume content using an LLM (Ollama).
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


