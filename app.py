import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory

# ---------------------------------------------------------------------
# ⚙️ Setup and Environment
# ---------------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="SCWGL Chatbot", layout="centered")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_CSE_ID"] = st.secrets["GOOGLE_CSE_ID"]

# ---------------------------------------------------------------------
# 🎨 Header
# ---------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("scwgl_image.jpeg", width=60)
with col2:
    st.markdown(
        "<h1 style='color: red; text-align: center;'>SCWGL Football</h1>",
        unsafe_allow_html=True,
    )
with col3:
    st.image("walton_hersham_logo.png", width=60)

st.markdown("---")

# ---------------------------------------------------------------------
# 📚 Sidebar: File Upload
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Your Documents")
    files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

# Auto-load PDFs from folder
pdf_folder = "pdfs"
if os.path.exists(pdf_folder):
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            files = files + [os.path.join(pdf_folder, filename)] if files else [os.path.join(pdf_folder, filename)]

# ---------------------------------------------------------------------
# 💬 Session State Initialization
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------------------------------------------------------------------
# 📖 Extract PDF Text
# ---------------------------------------------------------------------
text = ""
if files:
    for file in files:
        try:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += (page.extract_text() or "") + "\n"
        except Exception as e:
            st.warning(f"Could not read {file}: {e}")

# ---------------------------------------------------------------------
# 🧭 Context
# ---------------------------------------------------------------------
scwgl_context = """
The Surrey County Women and Girls Football League (SCWGL) manages women's and girls' football leagues in Surrey, UK.
It provides fixtures, regulations, and guidance for teams and clubs.
Questions should be answered in this SCWGL context.
"""

# ---------------------------------------------------------------------
# 🧩 Create or Load Vector Store
# ---------------------------------------------------------------------
if text and not st.session_state.vector_store:
    with st.spinner("📚 Indexing your documents..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_text(scwgl_context + "\n" + text)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
        st.success("✅ Documents indexed and ready!")

# ---------------------------------------------------------------------
# 🧠 LLM and Google Search Setup
# ---------------------------------------------------------------------
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4o-mini",
    temperature=0.2,
)
google_search = GoogleSearchAPIWrapper()
memory = ConversationBufferWindowMemory(k=5)

# ---------------------------------------------------------------------
# 🔍 RetrievalQA Chain
# ---------------------------------------------------------------------
qa_chain = None
if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ---------------------------------------------------------------------
# 💬 Chat Interface (ChatGPT-style)
# ---------------------------------------------------------------------
st.markdown("### 💬 Chat with SCWGL Assistant")

# Display chat history using Streamlit’s native chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input field
if prompt := st.chat_input("Ask about SCWGL, fixtures, or upload-related queries..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the user input
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = None

            # Step 1: Try to answer from uploaded documents
            if qa_chain:
                local_answer = qa_chain.run(prompt)
                if local_answer and "no relevant" not in local_answer.lower():
                    response = local_answer

            # Step 2: Fallback to Google Search
            if not response:
                try:
                    response = google_search.run(prompt)
                except Exception as e:
                    response = f"Web search failed: {e}"

            st.markdown(response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
