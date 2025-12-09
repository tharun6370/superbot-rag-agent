import streamlit as st
import os
import tempfile
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Suppress warnings
warnings.filterwarnings("ignore")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Superbot AI ", page_icon="⚡")
st.title("⚡ Superbot AI")
# st.caption("Faster loading, faster searching, faster answers.")

# --- SIDEBAR: SETUP ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Google API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# --- FILE UPLOADER ---
uploaded_file = st.sidebar.file_uploader(
    "Upload a Document", 
    type=["pdf", "docx", "txt", "csv"]
)

# --- HELPER: SMART LOADER (The Speed Fix) ---
def get_fast_loader(file_path, file_extension):
    """Selects the fastest loader for the specific file type."""
    if file_extension == ".pdf":
        return PyPDFLoader(file_path) # 10x faster than Unstructured
    elif file_extension == ".txt":
        return TextLoader(file_path)
    elif file_extension == ".docx":
        return Docx2txtLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path) # Fallback for complex files

# --- CACHED AGENT BUILDER ---
@st.cache_resource
def configure_agent(file_path=None, file_ext=None):
    # 1. Setup LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        max_retries=2, 
        request_timeout=30
    )

    # 2. Setup RAG Tool
    if file_path:
        print(f"🔄 Indexing {file_path}...")
        
        try:
            # USE SMART LOADER
            loader = get_fast_loader(file_path, file_ext)
            pages = loader.load()
            
            # Clean text
            for page in pages:
                page.page_content = page.page_content.replace('\n', ' ')

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever()
            
            @tool
            def ask_my_documents(query: str):
                """Use this tool to find info in the uploaded document."""
                docs = retriever.invoke(query)
                return "\n\n".join([d.page_content for d in docs])
                
            print("✅ Indexing Complete!")
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
            @tool
            def ask_my_documents(query: str): 
                """Fallback tool."""
                return "Error loading document."

    else:
        @tool
        def ask_my_documents(query: str):
            """Use this tool to check the user's uploaded documents."""
            return "The user has not uploaded a document yet."

    # 3. Setup Web Search (OPTIMIZED)
    # Limiting to 3 results makes the agent read less, responding faster.
    # wrapper = DuckDuckGoSearchAPIWrapper(max_results=3, time="d") 
    # web_search = DuckDuckGoSearchResults(api_wrapper=wrapper, backend="news")
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5) 
    web_search = DuckDuckGoSearchResults(api_wrapper=wrapper, backend="news")
    # 4. Create Agent
    tools = [ask_my_documents, web_search]
    memory = MemorySaver()
    
    sys_msg = """
    You are a Superbot. You have access to a private document and the public web.

    YOUR TOOLS:
    1. 'ask_my_documents': 
       - PRIORITY: HIGH. 
       - USE FOR: Definitions, summaries, specific terms (like "GreenBench", "Green Score"), or anything that might be in the uploaded file.
    
    2. 'duckduckgo_search': 
       - PRIORITY: LOW (Fallback).
       - USE FOR: Real-time news, general facts, or when the document DOES NOT have the answer.

    ROUTING RULE (CRITICAL):
    - When the user asks "What is X?", ALWAYS try 'ask_my_documents' FIRST. 
    - Only use 'duckduckgo_search' if the document returns "I don't know" or "Not found".
    - If the user asks for "Current news" or "Stock price", go straight to search.
    """

    return create_react_agent(llm, tools, checkpointer=memory), sys_msg

# --- MAIN LOGIC ---

if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("Please enter your Google API Key in the sidebar to start.")
    st.stop()

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Pass extension so we can pick the fast loader
    agent_executor, system_instruction = configure_agent(file_path=tmp_path, file_ext=file_extension)
else:
    agent_executor, system_instruction = configure_agent(file_path=None)

# --- CHAT INTERFACE ---

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I'm ready! Ask me anything about your document or the latest news."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            config = {"configurable": {"thread_id": "streamlit-session-v1"}}
            
            try:
                response = agent_executor.invoke(
                    {"messages": [("system", system_instruction), ("user", prompt)]},
                    config=config #type: ignore
                )
                
                raw_content = response['messages'][-1].content
                if isinstance(raw_content, list):
                    final_answer = raw_content[0]['text']
                else:
                    final_answer = raw_content
                
                st.write(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                
            except Exception as e:
                st.error(f"Error: {e}")