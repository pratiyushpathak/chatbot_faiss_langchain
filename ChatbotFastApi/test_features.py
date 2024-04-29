import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
import streamlit as st  
from dotenv import load_dotenv
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders.text import TextLoader
from langchain.docstore.document import Document  # Import the Document class

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

embedding_for_vector_db = SentenceTransformerEmbeddings(model_name="all-miniLM-L6-v2")
vector_db_path = "faiss_index"

db = FAISS.load_local(folder_path=vector_db_path, embeddings=embedding_for_vector_db, allow_dangerous_deserialization=True)


retriever = db.as_retriever(k=2)

data = TextLoader('/home/samruddhak/Downloads/state_of_the_union.txt')

async def add_data():
    db.add_documents(data)

# def get_docs(question):
#     docs_data = retriever.get_relevant_documents("madam president")
#     return docs_data


# st.header("vector DB different search")
# question = st.text_input("Enter to search in vectorstore")

# ans = get_docs(question)
# st.write(ans)