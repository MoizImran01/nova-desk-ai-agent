import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings

# 1. Initialize Google's Cloud Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.GOOGLE_GEMINI_API_KEY,
    output_dimensionality=768
)

os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY

def get_vector_store():

    return PineconeVectorStore(
        index_name="nova-spa-faqs", 
        embedding=embeddings
    )