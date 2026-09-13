from fastapi.testclient import TestClient
from app.main import app

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