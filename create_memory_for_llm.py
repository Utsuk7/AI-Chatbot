from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#1. Load Raw PDF(s)

DATA_PATH = 'data/'
def load_pdf_files(data):
    loader = DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)


#2. Create Chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)


#3.Create Vector Embeddings

def get_embedding_model():
    #embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()


#4. Store Embeddings in FAISS

#4. Store Embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)