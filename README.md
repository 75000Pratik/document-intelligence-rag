# Document Intelligence RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask grounded questions with source citations.

The system combines semantic retrieval, vector search, and a lightweight language model to generate answers based only on the uploaded document context.

---

## Features

- PDF upload and validation
- PDF text extraction
- Word-based chunking with overlap
- SentenceTransformer embeddings
- Chroma vector database
- Semantic search
- Retrieval-Augmented Generation
- Source citations
- Hallucination fallback handling
- Conversation history
- FastAPI backend
- Browser-based frontend
- Docker support
- GitHub Actions CI
- API tests with pytest
- Retrieval evaluation
- Chunking benchmark
- Persistent vector storage
- Duplicate document detection
- Safe filename handling

---

## Tech Stack

- Python
- FastAPI
- SentenceTransformers
- Hugging Face Transformers
- FLAN-T5
- ChromaDB
- PyPDF
- HTML
- CSS
- JavaScript
- Docker
- Pytest
- GitHub Actions

---

## Architecture

```text
User
  |
  v
Frontend
  |
  v
FastAPI
  |
  +--------------------------+
  |                          |
  |                          |
  v                          v
PDF Upload                 Question
  |                          |
  v                          v
Validation               Query Embedding
  |                          |
  v                          v
Text Extraction          Semantic Search
  |                          |
  v                          v
Chunking                 ChromaDB
  |                          |
  v                          v
Embeddings               Relevant Chunks
  |                          |
  v                          v
ChromaDB                 Prompt Construction
                             |
                             v
                          FLAN-T5
                             |
                             v
                       Grounded Answer
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
          Source Citations       Conversation History
```

---

## RAG Workflow

The application follows this pipeline:

1. User uploads a PDF document.
2. The application validates the uploaded file.
3. Text is extracted from each PDF page.
4. Extracted text is split into overlapping chunks.
5. Each chunk is converted into a semantic embedding.
6. Embeddings and metadata are stored in ChromaDB.
7. The user asks a question.
8. The question is converted into an embedding.
9. ChromaDB retrieves the most semantically relevant chunks.
10. Retrieved context is passed to FLAN-T5.
11. The model generates a grounded answer.
12. Source citations and conversation history are returned to the user.

---

## Retrieval Configuration

The current production retrieval configuration uses:

- Embedding model: `all-MiniLM-L6-v2`
- Embedding dimension: `384`
- Chunk size: `20 words`
- Chunk overlap: `5 words`
- Top-k retrieval: `2`
- Retrieval distance threshold: `1.325`

The retrieval threshold was selected using measured benchmark distances rather than a manually chosen value.

---

## Retrieval Evaluation

The retrieval system was evaluated using a controlled benchmark consisting of:

- 3 relevant questions
- 3 irrelevant questions
- 6 total questions

### Evaluation Result

```text
Passed: 6 / 6
Accuracy: 1.0
```

### Distance Analysis

```text
Highest relevant distance: 0.8871
Lowest irrelevant distance: 1.7628
Separation gap: 0.8757
Recommended threshold: 1.32499
Production threshold: 1.325
```

The benchmark currently uses a small controlled dataset, so the result should be interpreted as a configuration benchmark rather than a universal accuracy score.

---

## Chunking Benchmark

Three chunking strategies were compared.

| Strategy                           | Average Retrieval Distance |
| ---------------------------------- | -------------------------: |
| Fixed word chunking                |                     0.8362 |
| Sentence-aware chunking - 20 words |                     1.1126 |
| Sentence-aware chunking - 60 words |                     1.1207 |

Lower retrieval distance indicates stronger semantic similarity.

The fixed word chunking strategy performed best on the current evaluation dataset and was therefore selected for the production pipeline.

---

## Hallucination Handling

The system uses a retrieval distance threshold to reject weak or unrelated matches.

If the retrieved document context is not sufficiently relevant, the application returns:

```text
I don't know based on the provided document.
```

This helps prevent the model from generating unsupported answers.

---

## Source Citations

Each stored chunk includes metadata such as:

- source document
- page number
- chunk ID
- document ID

The API returns source metadata together with the generated answer.

Example:

```text
Answer:
document retrieval with a language model

Source:
sample.pdf - Page 1
```

---

## Conversation History

Each question can be associated with a:

```text
conversation_id
```

The application stores question-answer pairs and allows the frontend to display previous messages.

Current conversation history is stored locally in:

```text
data/conversation_history.json
```

The current implementation stores and displays previous messages, but previous messages are not yet used as model context for follow-up reasoning.

---

## API Endpoints

### Root

```http
GET /
```

Example response:

```json
{
  "message": "Document Intelligence RAG Assistant API"
}
```

---

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

---

### Upload PDF

```http
POST /upload
```

Uploads, validates, extracts, chunks, embeds, and indexes a PDF document.

---

### Ask a Question

```http
POST /ask
```

Example request:

```json
{
  "question": "What does Retrieval-Augmented Generation combine?",
  "document_id": "sample",
  "conversation_id": "conversation_1"
}
```

Example response:

```json
{
  "answer": "document retrieval with a language model",
  "sources": [
    {
      "source": "sample.pdf",
      "page": 1
    }
  ],
  "distances": [0.7497]
}
```

---

### Conversation History

```http
GET /history/{conversation_id}
```

Example:

```text
/history/conversation_1
```

Returns previously stored question-answer pairs for that conversation.

---

### Frontend

```text
/app/
```

The frontend is served directly by FastAPI.

---

### Swagger API Documentation

```text
/docs
```

---

## File Validation and Security

The upload system includes several validation checks:

- PDF extension validation
- MIME-type validation
- PDF magic-byte validation
- Empty-file detection
- Maximum file-size validation
- Safe filename extraction
- Path traversal protection
- Duplicate document detection
- Controlled server-error handling

Example path sanitization:

```text
../unsafe.pdf
```

is stored safely as:

```text
unsafe.pdf
```

---

## Error Handling

The application handles several common failure scenarios.

### Missing document

Returns:

```text
I don't know based on the provided document.
```

### Duplicate document

Returns HTTP:

```text
409 Conflict
```

### Invalid PDF

Returns HTTP:

```text
400 Bad Request
```

### Unexpected ingestion failure

Returns HTTP:

```text
500 Internal Server Error
```

---

## Project Structure

```text
document-intelligence-rag/
|
├── app/
│   ├── main.py
│   ├── rag.py
│   ├── ingest.py
│   ├── chunker.py
│   └── conversation.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── evaluation/
│   ├── evaluate_retrieval.py
│   ├── compare_chunking.py
│   └── benchmark_report.md
│
├── tests/
│   └── test_api.py
│
├── data/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/75000Pratik/document-intelligence-rag.git
cd document-intelligence-rag
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Run Locally

Start FastAPI:

```powershell
uvicorn app.main:app --reload
```

Open the frontend:

```text
http://127.0.0.1:8000/app/
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

---

## Run Tests

Run the API test suite:

```powershell
python -m pytest tests/test_api.py -v
```

Current result:

```text
13 passed
```

---

## Docker

Build the image:

```powershell
docker build -t document-intelligence-rag .
```

Run the container:

```powershell
docker run `
  --name document-intelligence-rag `
  -p 8000:8000 `
  -v rag_chroma_data:/app/chroma_db `
  -v rag_uploads_data:/app/data/uploads `
  document-intelligence-rag
```

Then open:

```text
http://127.0.0.1:8000/app/
```

---

## Persistent Storage

Docker volumes are used for persistent application data.

Current volumes:

```text
rag_chroma_data
rag_uploads_data
```

These preserve:

- ChromaDB vector data
- uploaded PDF documents

between container recreations.

---

## CI/CD

GitHub Actions automatically:

- checks out the repository
- sets up Python
- installs dependencies
- runs the API tests
- builds the Docker image

Current status:

```text
13 tests passing
GitHub Actions passing
Docker build passing
```

---

## Evaluation Scripts

Run retrieval evaluation:

```powershell
python .\evaluation\evaluate_retrieval.py
```

Run chunking comparison:

```powershell
python .\evaluation\compare_chunking.py
```

Detailed results are documented in:

```text
evaluation/benchmark_report.md
```

---

## Testing Strategy

The API tests cover areas such as:

- health endpoint
- root endpoint
- request validation
- invalid file uploads
- empty PDFs
- fake PDFs
- missing documents
- duplicate documents
- safe filename handling
- internal server errors
- conversation history
- conversation persistence

Tests use mocking and unique IDs where appropriate so they do not depend on existing local application state.

---

## Current Limitations

- Conversation history is stored in a local JSON file.
- Conversation history is not yet included as context for follow-up questions.
- The evaluation dataset contains only six questions.
- The application currently uses a lightweight local generation model.
- No authentication or user accounts are implemented.
- Uploaded documents are not separated by authenticated users.
- Conversation history is not yet persisted using a database.
- The project has not yet been hardened for large-scale concurrent usage.

---

## Future Improvements

Possible future improvements include:

- PostgreSQL or MongoDB conversation storage
- Authentication and authorization
- User-specific document collections
- Context-aware multi-turn conversations
- Larger RAG evaluation dataset
- Precision@k
- Recall@k
- Mean Reciprocal Rank
- Automated RAG answer-quality evaluation
- Reranking models
- Hybrid keyword + semantic search
- Streaming responses
- Cloud deployment
- Monitoring and observability
- Rate limiting
- Background document processing
- Support for additional document formats

---

## What I Learned

This project provided practical experience with:

- designing an end-to-end RAG pipeline
- semantic embeddings
- vector databases
- retrieval threshold tuning
- chunking strategy evaluation
- FastAPI architecture
- API validation
- error handling
- frontend/backend integration
- persistent application storage
- automated testing
- test isolation
- Docker
- CI/CD
- empirical evaluation of AI systems

---

## Author

**Pratik Tekade**

GitHub: `75000Pratik`

Repository:

```text
https://github.com/75000Pratik/document-intelligence-rag
```
