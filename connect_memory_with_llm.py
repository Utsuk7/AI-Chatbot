# essential imports 
import os
import warnings
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.exceptions import OutputParserException 

# Load environment variables from a .env file
warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# --- Step 1: Configuration and Prompt Setup ---
DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say "I don't know", don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt():
    """Defines and returns the custom prompt template."""
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt_template,
    )
    return custom_prompt

# --- Step 2: Check for API Key and Load Models ---
# The Gemini LLM requires the GOOGLE_API_KEY environment variable.
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not found. Please set it in your .env file or environment.")

# Load the embedding model for document retrieval
print("Loading embedding model...")
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit()

# Load the FAISS vector store
# allow_dangerous_deserialization=True is set when you trust the data source.
print("Loading FAISS vector database...")
try:
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except FileNotFoundError:
    print(f"Error: FAISS vector database not found at '{DB_FAISS_PATH}'. Please ensure it exists.")
    exit()
except Exception as e:
    print(f"Error loading FAISS database: {e}")
    exit()

# Initialize the Gemini LLM
print("Initializing Gemini LLM")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1.5)

# --- Step 3: Create the Retrieval QA Chain ---
print("Creating retrieval QA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": set_custom_prompt()},
    return_source_documents=True,
)

# --- Step 4: Interactive Chatbot Loop ---
print("Chatbot is ready. Type 'exit' or 'quit' to end the session.")
while True:
    user_query = input("Write your query (or 'exit' to quit): ")
    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not user_query:
        print("Please enter a query.")
        continue

    print("Searching for an answer...")
    try:
        response = qa_chain.invoke({"query": user_query})
        print(f"\n Answer: {response['result']}\n")

        print(" Source Documents:")
        if response.get('source_documents'):
            for i, doc in enumerate(response['source_documents'], 1):
                source_info = doc.metadata.get('source', 'N/A')
                print(f"  [{i}] Source: {source_info}")
                print(f"      Content: {doc.page_content[:300]}...\n") # show snippet
        else:
            print("  No source documents were retrieved.")

    except OutputParserException as e:
        print(f"An error occurred while parsing the model's output: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")