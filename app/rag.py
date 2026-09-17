import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)

generator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="document_chunks"
)


def answer_question(question, document_id):
    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2,
        where={
            "document_id": document_id
        },
        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    distances = results["distances"][0]

    if not distances:
        return {
            "answer": "I don't know based on the provided document.",
            "sources": [],
            "distances": []
        }

    best_distance = distances[0]

    threshold = 1.325

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are a document question-answering assistant.

Use ONLY the information in the context below.

If the context does not contain the answer, respond exactly with:
I don't know based on the provided document.

Do not guess.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""

    if best_distance > threshold:
        answer = "I don't know based on the provided document."

    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        outputs = generator_model.generate(
            **inputs,
            max_new_tokens=50
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

    print("\nGenerated answer:")
    print(answer)

    print("\nSources:")

    unique_sources = set()

    for metadata in retrieved_metadata:
        source_info = (
            metadata["source"],
            metadata["page"]
        )

        unique_sources.add(source_info)

    for source, page in unique_sources:
        print(f"- {source} - Page {page}")

    print("\nRetrieved chunks:")
    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):
        print(f"\nRank {index}:")
        print(chunk)

        print(
            "Distance:",
            distances[index - 1]
        )

    sources = []

    for source, page in unique_sources:
        sources.append(
            {
                "source": source,
                "page": page
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "distances": distances
    }


if __name__ == "__main__":
    result = answer_question(
        "What does Retrieval-Augmented Generation combine?",
        "sample"
    )

    print("\nReturned result:")
    print(result)
