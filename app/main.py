import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.rag import answer_question
from app.ingest import ingest_document

from app.conversation import add_message, get_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Intelligence RAG Assistant API",
    description=(
        "A Retrieval-Augmented Generation API that allows users "
        "to upload PDF documents and ask grounded questions with "
        "source citations."
    ),
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount(
    "/app",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Question to ask about the document"
    )

    document_id: str = Field(
        ...,
        min_length=1,
        description="ID of the document to search"
    )

    conversation_id: str = Field(
        ...,
        min_length=1,
        description="Conversation ID used to store chat history"
    )


class SourceResponse(BaseModel):
    source: str
    page: int


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    distances: list[float]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int


@app.get(
    "/",
    tags=["General"],
    summary="API information",
    description=(
        "Returns basic information about the "
        "Document Intelligence RAG Assistant API."
    )
)
def root():
    return {
        "message": "Document Intelligence RAG Assistant API"
    }


@app.get(
    "/health",
    tags=["General"],
    summary="Health check",
    description="Checks whether the API is running successfully."
)
def health():
    return {
        "status": "ok"
    }


@app.post(
    "/ask",
    response_model=AnswerResponse,
    tags=["RAG"],
    summary="Ask a document question",
    description=(
        "Retrieves relevant chunks from the selected document "
        "and generates a grounded answer with source citations."
    )
)
def ask_question(request: QuestionRequest):

    logger.info(
        "Question received for document_id=%s",
        request.document_id
    )

    result = answer_question(
        request.question,
        request.document_id
    )

    add_message(
        request.conversation_id,
        request.question,
        result["answer"]
    )

    logger.info(
        "Answer generated successfully for document_id=%s",
        request.document_id
    )

    return result


@app.get(
    "/history/{conversation_id}",
    tags=["RAG"],
    summary="Get conversation history",
    description="Returns all saved question-answer pairs for a conversation."
)
def conversation_history(conversation_id: str):
    return {
        "conversation_id": conversation_id,
        "messages": get_history(conversation_id)
    }


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Documents"],
    summary="Upload and index a PDF",
    description=(
        "Uploads a PDF document, validates it, extracts text, "
        "creates embeddings, and stores the chunks in Chroma."
    )
)
async def upload_document(
    file: UploadFile = File(...)
):

    # Validate extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    # Validate content type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a real PDF file."
        )

    # Create upload directory
    upload_dir = Path("data/uploads")

    upload_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_filename = Path(file.filename).name
    file_path = upload_dir / safe_filename

    # Read uploaded file
    contents = await file.read()

    # Reject empty file
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Validate PDF header
    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    # File size limit
    max_file_size = 10 * 1024 * 1024  # 10 MB

    if len(contents) > max_file_size:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 10 MB limit."
        )

    # Save file
    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    logger.info(
        "Uploading document filename=%s",
        file.filename
    )

    # Ingest document
    try:
        chunks, embeddings = ingest_document(
            str(file_path)
        )

    except ValueError as exc:
        logger.warning(
            "Document upload rejected filename=%s reason=%s",
            file.filename,
            str(exc)
        )

        raise HTTPException(
            status_code=409,
            detail=str(exc)
        )

    except Exception:
        logger.exception(
            "Unexpected Document ingestion failure filename=%s",
            file.filename
        )

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing the document."
        )

    logger.info(
        "Document indexed successfully filename=%s chunks=%s",
        file.filename,
        len(chunks)
    )

    return {
        "message": "Document uploaded and indexed successfully",
        "filename": safe_filename,
        "chunks_created": len(chunks)
    }
