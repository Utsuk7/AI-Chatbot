'''import os
import streamlit as st
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# --- Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Prompt templates ---
def set_custom_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say "I don't know", don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
    )

def set_condense_prompt():
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
Given the following conversation and a follow-up question, rephrase the follow-up question 
to be a standalone question.

Chat History:
{chat_history}
Follow-up question: {question}
Standalone question:
"""
    )

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# --- Build chatbot chain ---
def create_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    db = load_db()

    # Set output_key="answer" in memory too
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    question_generator = LLMChain(llm=llm, prompt=set_condense_prompt())
    combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=set_custom_prompt())

    convo_chain = ConversationalRetrievalChain(
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        question_generator=question_generator,
        combine_docs_chain=combine_docs_chain,
        memory=memory,
        return_source_documents=True,
        output_key="answer"  
    )

    return convo_chain, memory

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("🤖 Your Assistant")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain, st.session_state.memory = create_chain()
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")
submit = st.button("Send")

if submit and user_input:
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.qa_chain({
                "question": user_input,
                "chat_history": st.session_state.memory.chat_memory.messages
            })

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response["answer"]))

            st.markdown("### Chat History")
            for speaker, msg in st.session_state.chat_history:
                st.write(f"**{speaker}:** {msg}")

            st.markdown("### Source Documents")
            for i, doc in enumerate(response["source_documents"], 1):
                source = doc.metadata.get("source", "N/A")
                snippet = doc.page_content[:300]
                st.markdown(f"**[{i}] Source:** {source}\n\n`{snippet}`")

        except Exception as e:
            st.error(f" Error: {e}")
'''
import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import shutil
import datetime

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"

# Prompt templates
def set_custom_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say "I don't know", don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
    )

def set_condense_prompt():
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""
Given the following conversation and a follow-up question, rephrase the follow-up question 
to be a standalone question.

Chat History:
{chat_history}
Follow-up question: {question}
Standalone question:
"""
    )

# embedding for queries
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Build the question answer chain
def build_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    db = load_db()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    question_generator = LLMChain(llm=llm, prompt=set_condense_prompt())
    combine_docs_chain = load_qa_chain(llm, chain_type="stuff", prompt=set_custom_prompt())

# Create the ConversationalRetrievalChain to store memory and retrieve documents
    return ConversationalRetrievalChain(
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        question_generator=question_generator,
        combine_docs_chain=combine_docs_chain,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    ), memory

# Function to process uploaded PDFs and update the vectorstore (under developement)
def process_uploaded_pdfs(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    docs = []
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

    shutil.rmtree(temp_dir)
    return True

# Streamlit UI
st.set_page_config(page_title="Chatbot", page_icon="🧠")
st.title("🔍 AI-Chatbot ")

st.sidebar.subheader("Upload PDFs")
uploaded_pdfs = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
if st.sidebar.button("Process PDFs") and uploaded_pdfs:
    with st.spinner("Processing uploaded PDFs..."):
        success = process_uploaded_pdfs(uploaded_pdfs)
        if success:
            st.success("PDFs processed and vectorstore updated.")
            st.cache_resource.clear()
        else:
            st.error("Failed to process uploaded PDFs.")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain, st.session_state.memory = build_chain()
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question:")
submit = st.button("Send")

if submit and user_input:
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.qa_chain({
                "question": user_input,
                "chat_history": st.session_state.memory.chat_memory.messages
            })

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response["answer"]))

            st.markdown("### 💬 Chat History")
            for speaker, msg in st.session_state.chat_history:
                st.write(f"**{speaker}:** {msg}")

            st.markdown("### 📚 Source Documents")
            for i, doc in enumerate(response["source_documents"], 1):
                source = doc.metadata.get("source", "N/A")
                snippet = doc.page_content[:300]
                st.markdown(f"**[{i}] Source:** {source}\n\n`{snippet}`")

        except Exception as e:
            st.error(f"❌ Error: {e}")

if st.sidebar.button("Download Chat History") and st.session_state.chat_history:
    chat_log = "\n".join([f"{speaker}: {msg}" for speaker, msg in st.session_state.chat_history])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"chat_history_{timestamp}.txt"
    st.sidebar.download_button(
        label="Download Chat", data=chat_log, file_name=file_name, mime="text/plain"
    )
