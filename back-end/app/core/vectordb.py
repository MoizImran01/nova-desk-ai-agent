import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chroma_client = chromadb.HttpClient(host="localhost", port=8000)

def get_vector_store():
    return Chroma(client=chroma_client, collection_name="med_spa_faqs", embedding_function=embeddings)

