"""
SCWGL Chatbot - PDF-first with Google fallback (summarised by LLM)

Notes:
- Requires OPENAI_API_KEY and SERPER_API_KEY set in Streamlit secrets.
- Put your SCWGL PDFs in a folder named "pdfs" in the app root.
- Use Python 3.10+ on Streamlit Cloud.
"""

import os
from pathlib import Path
import streamlit as st

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback if you prefer
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.chains import ConversationalRetrievalChain

# ---------------------------------------------------------------------
# Configuration & context
# ---------------------------------------------------------------------
st.set_page_config(page_title="SCWGL Chatbot", layout="wide")
st.title("⚽ SCWGL Assistant")
st.markdown("Ask questions about the Surrey County Women & Girls League (SCWGL).")

# SCWGL system context used when summarising Google results or LLM fallback
SCWGL_CONTEXT = (
    "You are an assistant for the Surrey County Women and Girls Football League (SCWGL). "
    "Answer concisely in the context of the SCWGL (age groups, ball sizes, match formats, squad rules). "
    "If not known, say 'I don't know'."
)

# ---------------------------------------------------------------------
# Secrets check (fail fast)
# ---------------------------------------------------------------------
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY not found in Streamlit secrets. Add it to .streamlit/secrets.toml")
    st.stop()
if "SERPER_API_KEY" not in st.secrets:
    st.warning("SERPER_API_KEY not found in secrets — Google fallback will not work. Add SERPER_API_KEY to secrets.")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)

# ---------------------------------------------------------------------
# Sidebar: PDFs & debug
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Documents")
    st.markdown("Drop PDF files into the `pdfs/` folder (or upload them) and the app will index them.")
    st.markdown("Secrets required: `OPENAI_API_KEY`, (optional) `SERPER_API_KEY` for Google fallback.")
    st.write("---")
    st.markdown("**Index status:**")
    # show later after indexing

# ---------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": "..."}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "google_search" not in st.session_state:
    st.session_state.google_search = None
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0

# ---------------------------------------------------------------------
# Utility: format chat history to list of tuples for chain
# ---------------------------------------------------------------------
def format_chat_history(messages):
    """
    Convert Streamlit-style messages (dicts) to list of (user, assistant) tuples expected by chain.
    """
    history = []
    user_msg = None
    for msg in messages:
        if msg["role"] == "user":
            user_msg = msg["content"]
        elif msg["role"] == "assistant" and user_msg:
            history.append((user_msg, msg["content"]))
            user_msg = None
    return history

# ---------------------------------------------------------------------
# Load and index PDFs (cached resource)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore(pdf_folder: str, openai_api_key: str):
    """Load PDFs, split into docs and build FAISS vectorstore using OpenAI embeddings."""
    p = Path(pdf_folder)
    if not p.exists() or not any(p.glob("*.pdf")):
        return None, 0

    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()  # list of Document objects

    # Choose embeddings: OpenAIEmbeddings (accurate) but note cost; fallback to HuggingFace allowed
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    except Exception:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store, len(documents)

# Build vectorstore if pdfs exist
vector_store, doc_count = build_vectorstore("pdfs", OPENAI_API_KEY)
st.session_state.vector_store = vector_store
st.session_state.indexed_chunks = doc_count
if vector_store:
    st.sidebar.success(f"Indexed {doc_count} PDF document(s).")
else:
    st.sidebar.info("No PDFs found in ./pdfs (index not built).")

# ---------------------------------------------------------------------
# Initialize LLM and Google Serper
# ---------------------------------------------------------------------
if not st.session_state.llm:
    st.session_state.llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0)

if SERPER_API_KEY and not st.session_state.google_search:
    st.session_state.google_search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

# ---------------------------------------------------------------------
# Build conversational retrieval chain if vectorstore exists
# ---------------------------------------------------------------------
if st.session_state.vector_store and not st.session_state.qa_chain:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    # return_source_documents=True so we can check if any docs were actually retrieved
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=retriever,
        return_source_documents=True
    )

# ---------------------------------------------------------------------
# Helper: extract filenames/pages from source_documents (if present)
# ---------------------------------------------------------------------
def format_sources(source_docs):
    """
    source_docs are LangChain Document objects. Try to show filename or source metadata.
    Returns a short string or empty if none.
    """
    if not source_docs:
        return ""
    parts = []
    for d in source_docs:
        md = d.metadata or {}
        src = md.get("source") or md.get("filename") or md.get("file") or md.get("path")
        if src:
            parts.append(str(src))
    if not parts:
        # fallback to providing number of source docs
        return f"{len(source_docs)} document(s)"
    # unique and short
    uniq = sorted(set(parts))
    return ", ".join(uniq[:5])

# ---------------------------------------------------------------------
# Core handler: try PDF -> Google -> LLM
# ---------------------------------------------------------------------
def google_then_summarise(user_text: str) -> (str, str):
    """
    Query Google (Serper) and summarise the returned text via LLM.
    Returns (answer_text, source_label)
    """
    if not st.session_state.google_search:
        return None, None

    # ask Serper to search SCWGL site explicitly (improves relevance)
    google_query = f"site:scwgl.org.uk {user_text}"
    try:
        raw = st.session_state.google_search.run(google_query)  # string summarised by Serper wrapper
    except Exception as e:
        st.warning(f"Google search error: {e}")
        return None, None

    if not raw or not raw.strip():
        return None, None

    # Now summarise/clean the Google result through the LLM using SCWGL context
    prompt = (
        f"{SCWGL_CONTEXT}\n\n"
        "Below are snippets/pages from SCWGL website search results. "
        "Answer the user question concisely (1-2 sentences). If the exact info isn't present, say 'I don't know'.\n\n"
        f"Search snippets:\n{raw}\n\nUser question: {user_text}\n\nAnswer:"
    )
    try:
        llm_out = st.session_state.llm.invoke(prompt)
        summary = llm_out.content if hasattr(llm_out, "content") else str(llm_out)
    except Exception:
        # as a fallback, return the raw google text (shorten)
        summary = raw if len(raw) < 800 else raw[:800] + "..."

    if summary and summary.strip():
        return summary.strip(), "Google (summarised)"
    return None, None

def handle_user_input():
    user_text = st.session_state.user_input.strip()
    if not user_text:
        return

    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_text})

    response_text = None
    source_label = None

    # --------- 1) PDF Conversational Retrieval (preferred) ----------
    if st.session_state.qa_chain:
        try:
            # ConversationalRetrievalChain expects question + chat_history (list of tuples)
            chat_history = format_chat_history(st.session_state.messages)
            result = st.session_state.qa_chain.invoke({"question": user_text, "chat_history": chat_history})
            # result usually a dict with "answer" and "source_documents"
            if isinstance(result, dict):
                answer = result.get("answer", "").strip()
                source_docs = result.get("source_documents", []) or []
            else:
                answer = str(result).strip()
                source_docs = []

            # Use the PDF answer ONLY if source_docs is non-empty AND answer is substantive
            if source_docs and answer and "I don't know" not in answer.lower():
                response_text = answer
                src_info = format_sources(source_docs)
                source_label = f"PDF{': ' + src_info if src_info else ''}"
        except Exception as e:
            # do not abort — we'll fall back to Google / LLM
            st.warning(f"PDF QA error: {e}")

    # --------- 2) Google fallback (only if PDFs returned no usable sources) ----------
    if not response_text:
        google_answer, google_source = google_then_summarise(user_text)
        if google_answer:
            response_text = google_answer
            source_label = google_source

    # --------- 3) LLM fallback (last resort) ----------
    if not response_text:
        # direct LLM in SCWGL context
        prompt = f"{SCWGL_CONTEXT}\n\nUser question: {user_text}\n\nAnswer concisely:"
        try:
            llm_out = st.session_state.llm.invoke(prompt)
            response_text = llm_out.content if hasattr(llm_out, "content") else str(llm_out)
            source_label = "LLM"
        except Exception as e:
            response_text = f"LLM failed: {e}"
            source_label = "Error"

    # Append assistant response (with source)
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{response_text}\n\n_Source: {source_label}_"
    })

    # clear input
    st.session_state.user_input = ""

# ---------------------------------------------------------------------
# UI: render chat history and input
# ---------------------------------------------------------------------
st.markdown(f"**Indexed PDF documents:** {st.session_state.indexed_chunks}")

chat_box = st.container()
with chat_box:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

# Input area
st.text_input("Ask a question...", key="user_input", on_change=handle_user_input)
