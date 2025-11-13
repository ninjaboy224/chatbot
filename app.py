import streamlit as st
from pathlib import Path

# LangChain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
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

for key in ["user_input", "vector_store", "qa_chain", "llm", "google_search"]:
    if key not in st.session_state:
        st.session_state[key] = None

if st.session_state.user_input is None:
    st.session_state.user_input = ""

# ---------------------------------------------------------------------
# 🔹 Load PDFs and Create Vectorstore (cached)
# ---------------------------------------------------------------------
@st.cache_resource
def load_vectorstore(pdf_folder: str):
    path = Path(pdf_folder)
    if not path.exists():
        return None
    loader = PyPDFDirectoryLoader(pdf_folder)
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

if not st.session_state.vector_store:
    st.session_state.vector_store = load_vectorstore("pdfs")

# ---------------------------------------------------------------------
# 🔹 Initialize LLMs and Chains
# ---------------------------------------------------------------------
if not st.session_state.llm:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )

if st.session_state.vector_store and not st.session_state.qa_chain:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=retriever,
        return_source_documents=True
    )

if not st.session_state.google_search:
    st.session_state.google_search = GoogleSerperAPIWrapper(
        serper_api_key=st.secrets["SERPER_API_KEY"]
    )

# ---------------------------------------------------------------------
# 🔹 Format chat history
# ---------------------------------------------------------------------
def format_chat_history(messages):
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
# 🔹 Handle user input
# ---------------------------------------------------------------------
def handle_user_input():
    user_text = st.session_state.user_input.strip()
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    response_text = None
    source = None

    # Step 1: PDF QA
    if st.session_state.qa_chain:
        try:
            pdf_result = st.session_state.qa_chain.invoke({
                "question": user_text,
                "chat_history": format_chat_history(st.session_state.messages)
            })
            pdf_answer = pdf_result.get("answer") if isinstance(pdf_result, dict) else str(pdf_result)
            sources = pdf_result.get("source_documents") if isinstance(pdf_result, dict) else []

            if sources and pdf_answer.strip() and "I don't know" not in pdf_answer:
                response_text = pdf_answer
                source = "PDF"
        except Exception as e:
            st.warning(f"PDF QA failed: {e}")

    # Step 2: Google fallback
    if not response_text:
        try:
            google_query = f"{SCWGL_CONTEXT}\n\nUser question: {user_text} site:scwgl.org.uk"
            google_result = st.session_state.google_search.run(google_query)
            if google_result and google_result.strip():
                response_text = google_result
                source = "Google (SCWGL site)"
            else:
                response_text = "No good Google Search Result was found."
                source = "Google (SCWGL site)"
        except Exception as e:
            st.warning(f"Google Search failed: {e}")

    # Step 3: LLM fallback
    if not response_text:
        try:
            llm_prompt = f"{SCWGL_CONTEXT}\n\nUser question: {user_text}"
            llm_result = st.session_state.llm.invoke(llm_prompt)
            response_text = llm_result.content if hasattr(llm_result, "content") else str(llm_result)
            source = "LLM"
        except Exception as e:
            response_text = f"LLM failed: {e}"
            source = "Error"

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{response_text}\n\n_Source: {source}_"
    })

    st.session_state.user_input = ""

# ---------------------------------------------------------------------
# 🔹 Display chat messages
# ---------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------
# 🔹 Input box
# ---------------------------------------------------------------------
st.text_input("Ask your question...", key="user_input", on_change=handle_user_input)
