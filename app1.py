import streamlit as st
import os
import pickle
import re
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langfuse import Langfuse


# -------------------------
# Langfuse setup
# -------------------------
load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("PUBLIC_KEY"),
    secret_key=os.getenv("SECRET_KEY"),
    host=os.getenv("HOST")
)

# -------------------------
# PDF extraction & cleaning
# -------------------------
def clean_text(text):
    text = re.sub(r'\(cid:\d+\)', '', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', text)
    text = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'[§•#]', '', text)
    text = re.sub(r'\s*—\s*', ' - ', text)
    text = re.sub(r'\s*[\|]\s*', ' | ', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def extract_pdf_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = pdf.pages[page.page_number - 1].extract_text(x_tolerance=1, y_tolerance=1)
            if page_text:
                text += page_text + "\n"
    cleaned_text = clean_text(text)

    # REPLACEMENT for langfuse.log(...)
    with langfuse.start_as_current_span(name="pdf_extracted") as span:
        span.update(input={"filename": pdf_file.name, "length": len(cleaned_text)})

    return cleaned_text

# -------------------------
# Text splitting
# -------------------------
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_text(text)

    # REPLACEMENT for langfuse.log(...)
    with langfuse.start_as_current_span(name="text_split") as span:
        span.update(input={"num_chunks": len(chunks)})

    return chunks

# -------------------------
# Vectorstore functions
# -------------------------
def build_vectorstore(chunks, vectorstore_path="vectorstore.pkl"):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = FAISS.from_documents(documents, embedder)
    with open(vectorstore_path, "wb") as f:
        pickle.dump(vectorstore, f)

    # REPLACEMENT for langfuse.log(...)
    with langfuse.start_as_current_span(name="vectorstore_built") as span:
        span.update(input={"num_docs": len(documents), "vectorstore_path": vectorstore_path})

    return vectorstore

def load_vectorstore(vectorstore_path="vectorstore.pkl"):
    if os.path.exists(vectorstore_path):
        with open(vectorstore_path, "rb") as f:
            return pickle.load(f)
    return None

def retrieve_chunks(query, vectorstore, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    return " ".join([doc.page_content for doc in docs])

# -------------------------
# Ollama QA & Prompting
# -------------------------
model_name = "gemma3:270m"

def answer_query(query, vectorstore):
    context = retrieve_chunks(query, vectorstore, k=3)
    prompt = f"""
You are an expert HR assistant. Answer the user's question strictly based on the candidate's resume provided below.
Do NOT make assumptions or add information not present in the resume.

Name extraction (important):
- Treat the candidate name as the primary heading usually at the top.
- Heuristics to identify the name when formatting is lost:
  - Pick the first non-empty line that does NOT contain an email, phone number, URL, or words like "Resume", "Curriculum Vitae", "CV", "Summary", "Objective", "Profile".
  - If a line like "First Last | email | phone" exists, take the tokens before the first separator as the name.
  - Prefer a 2–4 token capitalized phrase without digits near the contact block.
- If no confident name is found, state: "Name not clearly present in the resume."

Source snippet requirement:
- For every answer, include a Source section that quotes the minimal snippet(s) from the resume used to answer (1–3 lines each). Do not fabricate or paraphrase the quoted snippet.

Resume Context:
{context}

Question:
{query}

Instructions:
- Answer concisely and clearly.
- Use only the resume content above. Do not use outside knowledge.
- If the information is not available in the resume, say: "The resume does not provide this information."
- Use bullet points only if listing multiple items (e.g., skills, programming languages, certifications).
- Always append a Source section quoting the exact resume text used. If multiple snippets were used, list them as separate bullet points under Source.

Answer:
"""



    response = ollama.generate(
        model=model_name,
        prompt=prompt
    )

    # REPLACEMENT for langfuse.log_llm(...)
    with langfuse.start_as_current_generation(
        name="ollama_generate",
        model=model_name,
        input={"query": query, "context": context}
    ) as gen:
        gen.update(output={"answer": response['response']})

    return response['response']

# -------------------------
# Streamlit UI
# -------------------------
st.title("Resume QA System")

uploaded_file = st.file_uploader("Upload a PDF Resume", type="pdf")

if uploaded_file is not None:
    # Extract text
    with st.spinner("Processing PDF..."):
        resume_text = extract_pdf_text(uploaded_file)
        resume_chunks = split_text(resume_text)
        vectorstore = build_vectorstore(resume_chunks)

    st.success("Resume processed successfully!")

    # -------------------------
    # Multi-question interface
    # -------------------------
    st.subheader("Ask Questions About the Resume")
    MAX_QUESTIONS = 7

    # Initialize session state for tracking
    if "questions" not in st.session_state:
        st.session_state.questions = []  # store past questions
        st.session_state.answers = []    # store corresponding answers

    # Show past questions and answers
    for q, a in zip(st.session_state.questions, st.session_state.answers):
        st.markdown(f"**Question:** {q}")
        st.markdown(f"**Answer:** {a}")

    # Only allow adding more questions if under MAX_QUESTIONS
    if len(st.session_state.questions) < MAX_QUESTIONS:
        # New input for next question
        new_query = st.text_input(
            f"Question {len(st.session_state.questions)+1}",
            key=f"question_{len(st.session_state.questions)}",
            value=""  # start empty each time
        )

        if new_query:
            # Generate answer
            answer = answer_query(new_query, vectorstore)

            # Save to session state
            st.session_state.questions.append(new_query)
            st.session_state.answers.append(answer)

            # Display immediately
            st.markdown(f"**Answer:** {answer}")
