from fastapi import APIRouter, UploadFile, File
from app.core.vectordb import get_vector_store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import os

router = APIRouter()

@router.post("/upload-faqs", tags=["Admin"])
async def upload_faqs(file: UploadFile = File(...)):
    try:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:

            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = file.filename

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)
        os.remove(tmp_path)

        return {"message": f"Successfully ingested {len(chunks)} chunks from {file.filename}"}
        
    except Exception as e:
        return {"error": str(e)}