# 🤖 Superbot: Universal RAG + Web Search Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![LangGraph](https://img.shields.io/badge/LangGraph-Agents-orange)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-purple)

**Superbot** is a multi-modal AI agent that intelligently routes user queries between a private knowledge base (your documents) and the live internet. It solves the problem of "Static RAG" by giving the AI the ability to search the web for real-time information when necessary.

## 🚀 Features

* **Universal Document Loading:** Upload **PDF, DOCX, TXT, or CSV** files. The bot automatically detects the format, cleans, and indexes them.
* **Intelligent Routing:** Uses **LangGraph** to decide if a question requires looking up the internal document or searching the web.
* **Real-Time Research:** Integrated **DuckDuckGo** search for current events, stocks, and news (with date validation to prevent stale data).
* **Conversational Memory:** Remembers context across turns (e.g., *"What is the price?"* -> *"Is **that** higher than yesterday?"*).
* **Modern UI:** Built with **Streamlit** for a clean, chat-like experience.

## 🛠️ Tech Stack

* **LLM:** Google Gemini 2.5 Flash (via LangChain)
* **Orchestration:** LangChain & LangGraph (ReAct Architecture)
* **RAG Engine:** FAISS (Vector Store) + HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* **Data Ingestion:** `unstructured` (for universal file parsing)
* **Frontend:** Streamlit

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tharun6370/superbot-rag-agent.git](https://github.com/tharun6370/superbot-rag-agent.git)
    cd superbot-rag-agent
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This handles complex dependencies like `unstructured[pdf]` and specific `Pillow` versions automatically.*

## 🏃‍♂️ Usage

1.  **Get a Google API Key:**
    * Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to get your free key.

2.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

3.  **Interact:**
    * Enter your API Key in the sidebar.
    * **Upload a document** (Lecture notes, Financial report, Resume, etc.).
    * **Ask questions:**
        * *"Summarize the uploaded file."* (Uses RAG)
        * *"What is the current stock price of Apple?"* (Uses Web Search)
        * *"Compare the concepts in my PDF to the latest news."* (Uses Both)

## 🧠 How It Works

The system uses a **Router Agent** architecture:

1.  **User Input** is analyzed by the Agent.
2.  **Router Logic:**
    * If the query is about the file → Calls `ask_my_documents` tool (Retrieval).
    * If the query is general/news → Calls `duckduckgo_search` tool.
    * If the query is hybrid → Uses both tools sequentially.
3.  **Synthesis:** The LLM combines the tool outputs into a final natural language response.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. Future roadmap includes:
* Adding support for "Chat with URL"
* Switching to Tavily for advanced search filtering
* Dockerizing the application

## 📄 License

MIT License
