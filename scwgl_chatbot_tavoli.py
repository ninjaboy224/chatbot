import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# LangChain 0.3.x imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search.tool import TavilySearchResults

# ---------------------------------------------------------------------
# 🧩 Setup and Environment
# ---------------------------------------------------------------------
#load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="SCWGL Chatbot", layout="wide")

SCWGL_LOGO_PATH = "scwgl_image.jpeg"
WALTON_HERSHAM_LOGO_PATH = "walton_hersham_logo.png"

# ---------------------------------------------------------------------
# 🎨 Page Layout with Logos
# ---------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image(SCWGL_LOGO_PATH, width=60)
with col2:
    st.markdown(
        "<h1 style='color: red; text-align: center; border-radius: 5px; max-width:100%; word-wrap: break-word;'>SCWGL Football</h1>",
        unsafe_allow_html=True,
    )
with col3:
    st.image(WALTON_HERSHAM_LOGO_PATH, width=60)

# ---------------------------------------------------------------------
# 📚 Sidebar: File Upload
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("Your Documents")
    files = st.file_uploader(
        "Upload PDF files and start asking questions",
        type="pdf",
        accept_multiple_files=True,
    )

# Autoload PDFs from folder
auto_pdf_folder = "pdfs"
if os.path.exists(auto_pdf_folder):
    for filename in os.listdir(auto_pdf_folder):
        if filename.endswith(".pdf"):
            files = files + [os.path.join(auto_pdf_folder, filename)] if files else [os.path.join(auto_pdf_folder, filename)]

# ---------------------------------------------------------------------
# 💬 Session Initialization
# ---------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------------------------------------------------------------------
# 📖 Extract Text from PDFs
# ---------------------------------------------------------------------
text = ""
if files:
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

if not text.strip() and files:
    st.warning("No text could be extracted from the uploaded PDFs. Check your files.")

# ---------------------------------------------------------------------
# 🧭 SCWGL Context
# ---------------------------------------------------------------------
scwgl_context = """
The Surrey County Women and Girls Football League (SCWGL) is an organization that manages women's and girls' football leagues in Surrey, UK. 
It provides fixtures, regulations, news, and support for teams and clubs participating in the league. 
All queries should be answered in the context of SCWGL, focusing on topics such as team registrations, league rules, fixtures, results, policies, FAQs, and club information.
"""

# ---------------------------------------------------------------------
# 🧩 Create or Load Vector Store
# ---------------------------------------------------------------------
if text and not st.session_state.vector_store:
    with st.spinner("Processing and indexing documents..."):
        full_text = scwgl_context + "\n" + text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
        st.success("Documents indexed and ready for search!")

# ---------------------------------------------------------------------
# 🧠 Initialize LLM and Memory
# ---------------------------------------------------------------------
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name="gpt-4"
)

memory = ConversationBufferWindowMemory(k=5)

# ---------------------------------------------------------------------
# 🔍 Define RetrievalQA Chain
# ---------------------------------------------------------------------
vector_store = st.session_state.vector_store
qa_chain = None
if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

# ---------------------------------------------------------------------
# 🌐 Tavily Fallback Search Tool
# ---------------------------------------------------------------------
tavily_tool = TavilySearchResults()

def fallback_web_search(query):
    """Use Tavily web search if no local SCWGL info found."""
    try:
        results = tavily_tool.run(query)
        if results:
            return results
    except Exception as e:
        return f"Tavily search failed: {e}"
    return "No web results found."

# ---------------------------------------------------------------------
# 💬 User Input
# ---------------------------------------------------------------------
user_question = st.text_input("Type your question here")

if user_question:
    response = None

    # Step 1: Try answering from uploaded documents
    if qa_chain:
        answer = qa_chain.run(user_question)
        if answer and "No relevant" not in answer:
            response = answer

    # Step 2: Fallback to web if nothing found locally
    if not response:
        response = fallback_web_search(user_question)

    # Step 3: Display answer
    st.session_state.chat_history.insert(0, f"Assistant: {response}")
    st.session_state.chat_history.insert(0, f"User: {user_question}")

    st.markdown(f"**Answer:** {response}")

# ---------------------------------------------------------------------
# 📜 Chat History Display
# ---------------------------------------------------------------------
if st.session_state.chat_history:
    with st.expander("Chat History"):
        for message in st.session_state.chat_history:
            st.text(message)
else:
    st.info("Upload PDFs or type a question to get started.")
