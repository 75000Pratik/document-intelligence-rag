import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path
from app.chunker import chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )
    return pages


def ingest_document(pdf_path, document_id=None):

    if document_id is None:
        document_id = Path(pdf_path).stem
    pages = extract_pdf_text(pdf_path)

    print("Pages extracted:", len(pages))

    all_chunks = []

    for page in pages:
        print(f"\nPage {page['page']}:")

        chunks = chunk_text(page["text"])

        for index, chunk in enumerate(chunks, start=1):
            chunk_data = {
                "source": "sample.pdf",
                "page": page["page"],
                "chunk_id": f"{document_id}_page{page['page']}_chunk{index}",
                "document_id": document_id,
                "text": chunk
            }

            all_chunks.append(chunk_data)

    print("\nTotal chunks created:", len(all_chunks))

    if not all_chunks:
        raise ValueError(
            "No readable text could be extracted from the PDF."
        )

    chunk_texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = model.encode(chunk_texts)

    print("Embeddings shape:", embeddings.shape)

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    collection = client.get_or_create_collection(
        name="document_chunks"
    )

    existing = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    if existing["ids"]:
        raise ValueError(f"Document '{document_id}' already exists.")

    ids = [
        chunk["chunk_id"]
        for chunk in all_chunks
    ]

    documents = [
        chunk["text"]
        for chunk in all_chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"]
        }
        for chunk in all_chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print("Chunks stored in Chroma:", len(ids))

    return all_chunks, embeddings


if __name__ == "__main__":
    chunks, embeddings = ingest_document(
        "data/sample.pdf"
    )

    print("\nReturned chunks:")
    print(chunks)
