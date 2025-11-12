import streamlit as st
import os
import glob
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# --- Load environment variables ---
#load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Streamlit page config ---
st.set_page_config(page_title="SCWGL Chatbot", layout="wide")

# --- Responsive layout: two logos + title ---
SCWGL_LOGO_PATH = "scwgl_image.jpeg"
WALTON_HERSHAM_LOGO_PATH = "walton_hersham_logo.png"

col1, col2, col3 = st.columns([1, 3, 1])
with col1: st.image(SCWGL_LOGO_PATH, width=60)
with col2:
    st.markdown("<h1 style='color:red; text-align:center; font-size:20px;'>SCWGL Chatbot</h1>", unsafe_allow_html=True)
with col3: st.image(WALTON_HERSHAM_LOGO_PATH, width=120)

# --- Sidebar for PDF uploads ---
with st.sidebar:
    st.title("Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Preload PDFs from repo ---
preloaded_files = glob.glob("pdfs/*.pdf")
all_pdfs = preloaded_files + (uploaded_files or [])

pdf_text = ""
for file in all_pdfs:
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() or ""

# --- SCWGL context ---
scwgl_context = """
The Surrey County Women and Girls Football League (SCWGL) manages women's and girls' football leagues in Surrey, UK.
It provides fixtures, regulations, news, and support for teams and clubs participating in the league.
All queries should be answered in the context of SCWGL, including team registrations, league rules, fixtures, results, policies, FAQs, and club info.
"""

# --- Create vector store ---
if pdf_text.strip():
    text = scwgl_context + "\n" + pdf_text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)

# --- Initialize LLM ---
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4")

# --- RetrievalQA chain ---
qa_chain = None
if st.session_state.vector_store:
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# --- User input ---
user_question = st.text_input("Ask a question about SCWGL")
if user_question and qa_chain:
    response = qa_chain.run(user_question)

    # Only store last question/answer
    st.session_state.chat_history = [f"User: {user_question}", f"Assistant: {response}"]

    # Display response styled for mobile
    st.markdown(f"""
        <div style='background-color:black; color:gold; padding:10px; border-radius:5px; word-wrap:break-word;'>
            {response}
        </div>
    """, unsafe_allow_html=True)

# --- Chat history ---
if st.session_state.chat_history:
    with st.expander("Chat History"):
        for msg in st.session_state.chat_history:
            st.markdown(f"""
                <div style='background-color:black; color:gold; padding:5px; border-radius:5px; word-wrap:break-word;'>
                    {msg}
                </div>
            """, unsafe_allow_html=True)
