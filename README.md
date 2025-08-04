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

---

## ⚙️ .env Configuration


```env
HF_TOKEN=your_hugging_face_api_key_here
GOOGLE_API_KEY=your_google_api_key_here



# 1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Install Pipenv
pip install pipenv

# 3. Install Project Dependencies from Pipfile
pipenv install

# 4. Activate the Virtual Environment
pipenv shell

# 5. Add Your PDF Files
# 👉 Place all .pdf documents you want to query into the 'data/' directory

# 6. Run the Indexer
# ⚙️ This script loads PDFs, splits them into chunks, embeds them, and stores them in a FAISS vector DB
python main.py

# 7. Start the Chatbot

# ✅ Terminal Interface
python main.py

# ✅ Web Interface (Streamlit)
streamlit run app.py
