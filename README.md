# 🤖 AI-Powered RAG Chatbot

This project is an AI-powered chatbot that allows users to query PDF documents conversationally using state-of-the-art LLMs and embeddings. Built with **LangChain**, **HuggingFace**, **FAISS**, **Gemini (Google Generative AI)**, and **Streamlit**.

---

## 📂 Project Structure

├── data/ # Directory containing your PDF files
├── vectorstore/ # FAISS vector database storage
├── Pipfile / Pipfile.lock # Pipenv environment and dependencies
├── .env # Environment variables (API keys, configs)
├── main.py # Terminal chatbot interface
├── app.py # Streamlit web app
└── README.md



---

## 🔧 Tech Stack

- **LangChain** – for chaining components and QA logic  
- **HuggingFace Transformers** – for semantic embeddings  
- **FAISS** – for vector-based similarity search  
- **Google Gemini** – for LLM-powered answer generation  
- **Streamlit** – for a lightweight web UI  
- **Pipenv** – for environment and dependency management  
- **dotenv** – for secure secret and config handling



## 🚀 Features

- 🧠 **Query-Response System**  
  Ask questions and get precise, context-aware answers based on your uploaded PDF documents.

- 📁 **Document Upload Support** *(Planned)*  
  Easily upload PDF files through the Streamlit interface (to be integrated, under development).

- 📝 **Conversational Memory**  
  Maintains context across turns using LangChain's `ConversationBufferMemory` for more natural, flowing interactions.

- 🗃️ **Source Document Citation**  
  Every answer includes references to the documents and the exact text chunk from which the answer was derived.

- 💬 **Dual Interface (CLI + Web UI)**  
  - Terminal-based interactive chat using `python connect_memory_with_llm.py`  
  - Web-based chat via `Streamlit` in `app.py`

- 💾 **Downloadable Chat History**   
  Enable users to download or export their chat conversations.

- ⚙️ **FAISS-based Vector Store**  
  Efficient document retrieval using FAISS for storing and querying semantic embeddings.

- 🔐 **Secure Environment Variable Handling**  
  All sensitive keys like `GOOGLE_API_KEY` are managed via a `.env` file.

- 🔄 **Easy Re-indexing of New PDFs**  
  Just run `python create_memory_for_llm.py` to process newly added PDFs (automatic embedding creation---> future work).


---


## ⚙️ .env Configuration


```env
HF_TOKEN=your_hugging_face_api_key_here
GOOGLE_API_KEY=your_google_api_key_here



# 1. Clone the Repository
git clone git@github.com:Utsuk7/AI-Chatbot.git
cd AI-Chatbot


# 2. Install Pipenv
pip install pipenv


# 3. Install Project Dependencies from Pipfile
pipenv install


# 4. Activate the Virtual Environment
pipenv shell


# 5. Add Your PDF Files
# 👉 Place all .pdf documents you want to query into the 'data/' directory


# 6. 🆕 Update Memory
After placing new PDF files in the `data/` directory, you need to create chunks and generate embeddings for them.

Run the following command to update the FAISS vector store with the new PDFs:
```bash
python create_memory_for_llm.py


# 7. Start the Chatbot

# ✅ Terminal Interface
python connect_memory_with_llm.py


# ✅ Web Interface (Streamlit)
streamlit run app.py
