from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import verify_admin_api_key
from app.services.rag_service import RAGService

router = APIRouter()


@router.post(
    "/upload-faqs",
    tags=["Admin"],
    dependencies=[Depends(verify_admin_api_key)],
)
def upload_faqs(file: UploadFile = File(...)):
    file_bytes = file.file.read()
    rag_service = RAGService()
    result = rag_service.ingest_pdf(
        file_bytes=file_bytes,
        filename=file.filename or "",
    )
    return {
        "message": f"Successfully ingested {result.chunks_ingested} chunks from {result.filename}",
        "chunks_ingested": result.chunks_ingested,
    }
