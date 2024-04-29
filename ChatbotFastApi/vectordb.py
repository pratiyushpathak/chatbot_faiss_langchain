from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import faiss
from dotenv import load_dotenv



load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
# Reading the text from pdf page by page and storing it into various
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Getting the text into number of chunks as it is helpful in faster processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Storing the text chunks into embeddings to retrive the answer for the query outoff it
def get_vector_store(text_chunks):
    
    # vector_store = faiss.FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store = faiss.FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    vector_store.add_texts(text_chunks)
    vector_store.save_local("faiss_index")
    