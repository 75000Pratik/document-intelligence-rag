import uuid
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Document Intelligence RAG Assistant API"
    }


def test_ask_short_question():
    response = client.post(
        "/ask",
        json={
            "question": "Hi",
            "document_id": "sample"
        }
    )

    assert response.status_code == 422


def test_ask_empty_document_id():
    response = client.post(
        "/ask",
        json={
            "question": "WHat is RAG?",
            "document_id": ""
        }
    )

    assert response.status_code == 422


def test_upload_non_pdf():
    response = client.post(
        "upload",
        files={
            "file": (
                "test.txt",
                b"hello",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are allowed."
    }


def test_upload_empty_pdf():
    response = client.post(
        "/upload",
        files={
            "file": (
                "empty.pdf",
                b"",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded file is empty."
    }


def test_upload_fake_pdf():
    response = client.post(
        "/upload",
        files={
            "file": (
                "fake.pdf",
                b"this is not a real pdf",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Invalid PDF file."
    }


def test_ask_missing_document():
    response = client.post(
        "/ask",
        json={
            "question": "What is this document about?",
            "document_id": "does not exist",
            "conversation_id": "missing_document test"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "I don't know based on the provided document."
    assert data["sources"] == []
    assert data["distances"] == []


def test_upload_duplicate_document():
    unique_id = uuid.uuid4().hex
    filename = f"duplicate_test_{unique_id}.pdf"

    with open("data/sample.pdf", "rb") as file:
        first_response = client.post(
            "/upload",
            files={
                "file": (
                    filename,
                    file,
                    "application/pdf"
                )
            }
        )

    assert first_response.status_code == 200

    with open("data/sample.pdf", "rb") as file:
        second_response = client.post(
            "/upload",
            files={
                "file": (
                    filename,
                    file,
                    "application/pdf"
                )
            }
        )

    assert second_response.status_code == 409


def test_upload_sanitizes_filename():
    with open("data/sample.pdf", "rb") as file:
        response = client.post(
            "/upload",
            files={
                "file": (
                    "../unsafe.pdf",
                    file,
                    "application/pdf"
                )
            }
        )

    assert response.status_code in (200, 409)

    data = response.json()

    if response.status_code == 200:
        assert data["filename"] == "unsafe.pdf"

    if response.status_code == 409:
        assert "already exists" in data["detail"]


def test_upload_internal_server_error():
    with patch(
        "app.main.ingest_document",
        side_effect=RuntimeError("Database failure")
    ):
        with open("data/sample.pdf", "rb") as file:
            response = client.post(
                "/upload",
                files={
                    "file": (
                        "server_error_test.pdf",
                        file,
                        "application/pdf"
                    )
                }
            )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error while processing the document."
    }


def test_conversation_history():
    conversation_id = "test_conversation_history"

    ask_response = client.post(
        "/ask",
        json={
            "question": "What does Retrieval-Augmented Generation combine?",
            "document_id": "sample",
            "conversation_id": conversation_id
        }
    )

    assert ask_response.status_code == 200

    history_response = client.get(
        f"/history/{conversation_id}"
    )

    assert history_response.status_code == 200

    data = history_response.json()

    assert data["conversation_id"] == conversation_id
    assert len(data["messages"]) >= 1
    assert data["messages"][-1]["question"] == (
        "What does Retrieval-Augmented Generation combine?"
    )
    assert data["messages"][-1]["answer"] == (
        "document retrieval with a language model"
    )

def test_conversation_history_persistence():
    conversation_id = "persistence_test_case"

    ask_response = client.post(
        "/ask",
        json={
            "question": "What does Retrieval-Augmented Generation combine?",
            "document_id": "sample",
            "conversation_id": conversation_id
        }
    )

    assert ask_response.status_code == 200

    history_response = client.get(
        f"/history/{conversation_id}"
    )

    assert history_response.status_code == 200

    data = history_response.json()

    assert data["conversation_id"] == conversation_id
    assert len(data["messages"]) >= 1