import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit UI Configuration
st.set_page_config(page_title="SCWGL Chatbot", layout="wide")

# Image Paths
SCWGL_LOGO_PATH = "scwgl_image.jpeg"
WALTON_HERSHAM_LOGO_PATH = "walton_hersham_logo.png"

# Layout with two logos
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    st.image(SCWGL_LOGO_PATH, width=80)

with col2:
    st.markdown(
        "<h1 style='color: red; text-align: center;'>Surrey County Women and Girls Football League Chatbot</h1>",
        unsafe_allow_html=True)

with col3:
    st.image(WALTON_HERSHAM_LOGO_PATH, width=130)

# Sidebar for file upload
with st.sidebar:
    st.title("Your Documents")
    files = st.file_uploader("Upload PDF files and start asking questions", type="pdf", accept_multiple_files=True)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Extract text from PDFs
text = ""
if files:
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text() or ""
            text += extracted_text + "\n"

# SCWGL Context
scwgl_context = """
The Surrey County Women and Girls Football League (SCWGL) is an organization that manages women's and girls' football leagues in Surrey, UK. 
It provides fixtures, regulations, news, and support for teams and clubs participating in the league. 
All queries should be answered in the context of SCWGL, focusing on topics such as team registrations, league rules, fixtures, results, policies, FAQs, and club information.
"""

# Initialize vector_store as None
vector_store = None

# Initialize the LLM (GPT-4 for conversational responses)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-4")

# Combine documents with context if PDFs are uploaded
if text:
    text = scwgl_context + text  # Prepend SCWGL context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)

# Define a safe function for SCWGL document search
def search_scwgl_docs(query):
    """Search SCWGL document vector store if available."""
    if vector_store:
        docs = vector_store.similarity_search(query)
        if docs:
            return "\n".join([doc.page_content for doc in docs])
        return "No relevant SCWGL document found."
    return "No SCWGL documents uploaded yet."

scwgl_tool = Tool(
    name="SCWGL Document Search",
    func=search_scwgl_docs,
    description="Use this tool to search SCWGL documents and league regulations."
)

tavily_tool = TavilySearchResults()  # External search replaces web scraping

# Initialize the agent with SCWGL search first, then Tavily

# Create an agent memory to limit chat history tokens
memory = ConversationBufferWindowMemory(k=5)  # Keep only the last 5 interactions

# Define the agent with limited memory
agent = initialize_agent(
    [scwgl_tool, tavily_tool],
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory  # Ensures conversation history is managed
)

# User input for queries
user_question = st.text_input("Type your question here")
if user_question:
    # Formulate the enhanced prompt
    prompt_with_context = f"""
    You are a chatbot for the Surrey County Women and Girls Football League (SCWGL). 
    Always assume that the user is asking about SCWGL unless explicitly stated otherwise.

    {scwgl_context}

    Chat History:
    {st.session_state.chat_history}

    User: {user_question}
    Assistant:
    """

    # Invoke the agent
    response = agent.run(prompt_with_context)

    # Store chat history for display
    st.session_state.chat_history.insert(0, f"Assistant: {response}")
    st.session_state.chat_history.insert(0, f"User: {user_question}")

    st.write(response)

# Display Chat History in Reverse Order
if st.session_state.chat_history:
    with st.expander("Chat History"):
        for message in st.session_state.chat_history:
            st.text(message)
else:
    st.warning("Enter a question to proceed.")
