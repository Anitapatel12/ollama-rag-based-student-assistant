# app.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import io
import tempfile
import psutil

import streamlit as st
from pypdf import PdfReader
import docx2txt

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LCFAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


# -----------------------------
# Document Loading
# -----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        text = docx2txt.process(tmp.name)
    return text

def extract_text(file) -> str:
    file_bytes = file.read()
    if file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif file.name.lower().endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif file.name.lower().endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


# -----------------------------
# Embeddings + FAISS
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

def build_vectorstore(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    docs = splitter.create_documents([text])
    embeddings = get_embedding_model()
    vectorstore = LCFAISS.from_documents(docs, embeddings)
    return vectorstore


# -----------------------------
# Ollama LLM Setup
# -----------------------------
def get_system_memory_gb():
    mem = psutil.virtual_memory()
    return mem.available / (1024 ** 3)  # GB

def get_llm(model_name: str):
    available_mem = get_system_memory_gb()
    
    # List of commonly available Ollama models to try as fallbacks
    FALLBACK_MODELS = [
        "llama3:8b",
        "llama3",
        "mistral",
        "gemma:2b",
        "phi3"
    ]
    
    # Try the requested model first
    models_to_try = [model_name] + FALLBACK_MODELS
    
    for current_model in models_to_try:
        try:
            # Adjust model based on available memory for 8b models
            if current_model.endswith("8b") and available_mem < 2.0:
                st.warning(f"Not enough memory for {current_model}, trying smaller models...")
                continue
                
            st.info(f"Trying to load model: {current_model}")
            llm = OllamaLLM(model=current_model)
            st.success(f"Successfully loaded model: {current_model}")
            return llm
            
        except Exception as e:
            if current_model == models_to_try[-1]:  # If this was the last model to try
                error_msg = (
                    f"Failed to load any Ollama model.\n\n"
                    f"Tried models: {', '.join(models_to_try)}\n\n"
                    f"Error details: {str(e)}\n\n"
                    "Please ensure:\n"
                    "1. Ollama is installed (https://ollama.com)\n"
                    "2. Ollama service is running: `ollama serve`\n"
                    f"3. You have pulled the model: `ollama pull {model_name}`\n"
                    "4. You have sufficient system resources"
                )
                raise ValueError(error_msg)
            continue


# -----------------------------
# Prompts
# -----------------------------
SUMMARY_PROMPT_TEMPLATE = """
You are an expert placement mentor. Read the JD and provide a focused, actionable summary.

Focus on:
- Key responsibilities
- Required skills and experience
- What to prioritize for preparation (top 5 topics)
- A short 30-day study plan (weekly breakdown)

JD:
{document_text}

Summary:
"""

QA_PROMPT_TEMPLATE = """
You are a helpful placement assistant. Use ONLY the context provided from the JD unless user allows 'LLM knowledge beyond JD'.

Context:
{context}

Question:
{question}

Answer concisely and include actionable steps if applicable.
If the context does not contain the answer, reply "Not available in the JD."
"""

QUESTION_PROMPT_TEMPLATE = """
You are an interviewer for role: {role}. Based on this JD context, generate:
- 10 likely interview questions (mix of HR, behavioral, domain, and technical)
- For each question, give a 1-2 line hint on how to answer well.

Context:
{context}
"""

ROADMAP_PROMPT_TEMPLATE = """
You are a placement coach. Create a {days}-day roadmap to prepare for the role described in the JD.
Make it day-by-day and actionable (topics + practice + revision + mock interviews).

Context:
{context}
"""


# -----------------------------
# Chains
# -----------------------------
def get_qa_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_PROMPT_TEMPLATE
    )

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def summarize_document(llm, document_text: str) -> str:
    truncated = document_text[:8000]
    prompt = PromptTemplate(input_variables=["document_text"], template=SUMMARY_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"document_text": truncated})


def generate_interview_questions(llm, document_text: str) -> str:
    truncated = document_text[:8000]
    prompt = PromptTemplate(input_variables=["context","role"], template=QUESTION_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": truncated, "role": "the role"})


def generate_roadmap(llm, document_text: str, days: int = 30) -> str:
    truncated = document_text[:8000]
    prompt = PromptTemplate(input_variables=["context","days"], template=ROADMAP_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": truncated, "days": days})


# -----------------------------
# Intent Classifier
# -----------------------------
def classify_intent(text: str) -> str:
    t = text.lower()
    if "roadmap" in t or "plan" in t:
        return "roadmap"
    if "question" in t or "interview" in t:
        return "questions"
    if "summary" in t or "focus" in t:
        return "summary"
    return "qa"


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="JD Agentic Placement Assistant", layout="wide")
    st.title("🤖 JD Agentic Placement Assistant (Ollama + RAG)")

    st.sidebar.header("⚙️ Ollama Config")
    model_name = st.sidebar.text_input(
    "Ollama Model Name",
    value="llama3.1:8b",  # <- match the model you have
    help="Example: llama3.1:8b | gemma2 | mistral | gpt-4"
)


    if st.sidebar.button("Initialize Ollama LLM"):
        try:
            llm = get_llm(model_name)
            st.session_state["llm"] = llm
            st.sidebar.success(f"{model_name} loaded ✅")
        except Exception as e:
            st.sidebar.error(str(e))

    uploaded_file = st.file_uploader("Upload JD (PDF / DOCX / TXT)", type=["pdf","docx","txt"])
    if uploaded_file is not None and st.button("📄 Extract & Index JD"):
        with st.spinner("Extracting text..."):
            text = extract_text(uploaded_file)
        st.session_state["document_text"] = text
        with st.spinner("Building FAISS index..."):
            vectorstore = build_vectorstore(text)
            st.session_state["vectorstore"] = vectorstore
        st.success("✅ JD indexed successfully")

    if "llm" not in st.session_state:
        st.info("➡️ Step 1: Start Ollama and initialize LLM.")
        return
    if "vectorstore" not in st.session_state:
        st.info("➡️ Step 2: Upload and index a JD")
        return

    llm = st.session_state["llm"]
    document_text = st.session_state["document_text"]
    vectorstore = st.session_state["vectorstore"]

    col1, col2 = st.columns([1,2])

    with col1:
        st.subheader("One-Click Actions")
        if st.button("🔍 Generate Summary"):
            st.session_state["last_summary"] = summarize_document(llm, document_text)

        days = st.number_input("Days for Roadmap", 7, 90, 30)
        if st.button("🗺️ Generate Roadmap"):
            st.session_state["last_roadmap"] = generate_roadmap(llm, document_text, days)

        if st.button("💡 Generate Interview Questions"):
            st.session_state["last_questions"] = generate_interview_questions(llm, document_text)

    with col2:
        st.subheader("Chat with JD Assistant")
        user_q = st.text_input("Ask something about the JD")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        if st.button("Ask"):
            intent = classify_intent(user_q)
            if intent == "summary":
                answer = summarize_document(llm, document_text)
            elif intent == "roadmap":
                answer = generate_roadmap(llm, document_text, days)
            elif intent == "questions":
                answer = generate_interview_questions(llm, document_text)
            else:
                qa_chain = get_qa_chain(vectorstore, llm)
                answer = qa_chain.invoke(user_q)

            st.session_state["chat_history"].append(("You", user_q))
            st.session_state["chat_history"].append(("Bot", answer))

        for sender, msg in st.session_state["chat_history"][-6:]:
            if sender == "You":
                st.markdown(f"**🧑 You:** {msg}")
            else:
                st.markdown(f"**🤖 Bot:** {msg}")

        if "last_summary" in st.session_state:
            st.subheader("Summary")
            st.write(st.session_state["last_summary"])
        if "last_roadmap" in st.session_state:
            st.subheader("Roadmap")
            st.write(st.session_state["last_roadmap"])
        if "last_questions" in st.session_state:
            st.subheader("Interview Questions")
            st.write(st.session_state["last_questions"])


if __name__ == "__main__":
    main()
