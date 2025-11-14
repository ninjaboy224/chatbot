import streamlit as st
from pathlib import Path
import json

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSerperAPIWrapper

# ---------------------------------------------------------------------
# ⚙️ Streamlit Setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="SCWGL Chatbot", page_icon="⚽", layout="wide")
st.title("⚽ SCWGL Football")
st.markdown("### 💬 Chat with SCWGL Assistant")

# ---------------------------------------------------------------------
# 📚 SCWGL Context
# ---------------------------------------------------------------------
SCWGL_CONTEXT = """
You are an assistant for the Surrey County Women and Girls Football League (SCWGL). 
Answer all questions in the context of SCWGL, focusing on girls’ football teams, age groups, fixtures, league rules, ball sizes, match formats, and policies.
"""

# ---------------------------------------------------------------------
# 📚 Session State Initialization
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for key in ["user_input", "vector_store", "llm", "google_search"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.user_input is None:
    st.session_state.user_input = ""

# ---------------------------------------------------------------------
# 🔹 Load PDFs into Vectorstore (cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_vectorstore(pdf_folder: str):
    path = Path(pdf_folder)
    if not path.exists():
        return None
    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(documents, embeddings)

# Load initial PDFs
if not st.session_state.vector_store:
    st.session_state.vector_store = load_vectorstore("pdfs")

# Optionally upload more PDFs
uploaded_files = st.file_uploader("Upload additional PDF(s)", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    # Save uploaded PDFs to `pdfs/` folder
    pdf_dir = Path("pdfs")
    pdf_dir.mkdir(exist_ok=True)
    for file in uploaded_files:
        with open(pdf_dir / file.name, "wb") as f:
            f.write(file.getbuffer())
    # Reload vector store
    st.session_state.vector_store = load_vectorstore("pdfs")
    st.success(f"{len(uploaded_files)} PDF(s) uploaded and indexed!")

# ---------------------------------------------------------------------
# 🧠 Initialize LLM and Google Search
# ---------------------------------------------------------------------
if not st.session_state.llm:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

if not st.session_state.google_search:
    st.session_state.google_search = GoogleSerperAPIWrapper(
        serper_api_key=st.secrets["SERPER_API_KEY"]
    )

# ---------------------------------------------------------------------
# 🗂 Helpers
# ---------------------------------------------------------------------
def format_chat_history(messages):
    """Return chat history as string for LLM input."""
    return "\n".join([
        f"User: {msg['content']}" if msg["role"]=="user" else f"Assistant: {msg['content']}"
        for msg in messages
    ])

def extract_text_from_result(res):
    """Normalise outputs from .invoke() / .run() into a string safely."""
    if res is None:
        return ""
    if isinstance(res, dict):
        for key in ("answer", "output_text", "output", "text", "result"):
            if key in res and res[key] is not None:
                return str(res[key])
        for v in res.values():
            if isinstance(v, str) and v.strip():
                return v
        return json.dumps(res)
    if hasattr(res, "content"):
        return str(res.content)
    if hasattr(res, "text"):
        return str(res.text)
    if isinstance(res, str):
        return res
    return str(res)

# ---------------------------------------------------------------------
# 🧠 Handle User Input
# ---------------------------------------------------------------------
def handle_user_input():
    user_text = st.session_state.user_input.strip()
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    response_text = None
    source = None

    # Step 1: PDF-first retrieval
    if st.session_state.vector_store:
        try:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_text)
            if docs:
                retrieved_text = "\n\n".join([d.page_content for d in docs])
                history_text = format_chat_history(st.session_state.messages)
                llm_input = (
                    f"{SCWGL_CONTEXT}\n\n"
                    f"Chat history:\n{history_text}\n\n"
                    f"Retrieved PDF content:\n{retrieved_text}\n\n"
                    f"Question: {user_text}\nAnswer concisely:"
                )
                pdf_answer_obj = st.session_state.llm.invoke(llm_input)
                pdf_answer = extract_text_from_result(pdf_answer_obj).strip()
                if pdf_answer and "i don't know" not in pdf_answer.lower():
                    response_text = pdf_answer
                    source = "PDF"
        except Exception as e:
            st.warning(f"PDF QA failed: {e}")

    # Step 2: Google fallback → LLM summarization
    if not response_text:
        try:
            google_query = f"{SCWGL_CONTEXT}\n\nUser question: {user_text} site:scwgl.org.uk"
            raw_google = st.session_state.google_search.run(google_query)
            raw_google_text = extract_text_from_result(raw_google).strip()
            if raw_google_text:
                llm_summary_prompt = (
                    f"{SCWGL_CONTEXT}\n\n"
                    f"Below are search snippets from the SCWGL site. Use them to answer the user's question concisely.\n\n"
                    f"Search snippets:\n{raw_google_text}\n\nQuestion: {user_text}\n\nAnswer:"
                )
                raw_llm_sum = st.session_state.llm.invoke(llm_summary_prompt)
                llm_summary = extract_text_from_result(raw_llm_sum).strip()
                if llm_summary:
                    response_text = llm_summary
                    source = "Google → LLM Summary"
                else:
                    response_text = raw_google_text[:1500] + ("..." if len(raw_google_text) > 1500 else "")
                    source = "Google (raw)"
            else:
                response_text = "No good Google Search Result was found."
                source = "Google"
        except Exception as e:
            st.warning(f"Google Search failed: {e}")

    # Step 3: LLM fallback
    if not response_text:
        try:
            llm_prompt = f"{SCWGL_CONTEXT}\n\nUser question: {user_text}"
            raw_llm = st.session_state.llm.invoke(llm_prompt)
            response_text = extract_text_from_result(raw_llm).strip()
            source = "LLM"
        except Exception as e:
            response_text = f"LLM failed: {e}"
            source = "Error"

    # Append assistant response
    if response_text is None:
        response_text = "Sorry, I couldn't find an answer."

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{response_text}\n\n_Source: {source}_"
    })
    st.session_state.user_input = ""

# ---------------------------------------------------------------------
# 💬 Display Chat Messages
# ---------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------
# ⌨️ Input Box
# ---------------------------------------------------------------------
st.text_input("Ask your question...", key="user_input", on_change=handle_user_input)
