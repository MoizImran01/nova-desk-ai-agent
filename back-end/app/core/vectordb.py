from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a cached singleton GoogleGenerativeAIEmbeddings client."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME,
        google_api_key=settings.GOOGLE_GEMINI_API_KEY,
        output_dimensionality=settings.EMBEDDING_DIMENSIONS,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    """Return a cached singleton PineconeVectorStore instance without mutating os.environ at import time."""
    return PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=get_embeddings(),
        pinecone_api_key=settings.PINECONE_API_KEY,
    )
