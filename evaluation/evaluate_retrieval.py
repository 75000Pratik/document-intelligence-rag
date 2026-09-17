import chromadb
from sentence_transformers import SentenceTransformer


evaluation_cases = [
    {
        "question": "What does Retrieval-Augmented Generation combine?",
        "document_id": "sample",
        "expected_keyword": "document retrieval"
    },
    {
        "question": "What will this project answer questions using?",
        "document_id": "sample",
        "expected_keyword": "uploaded documents"
    },
    {
        "question": "Does the project provide source citations?",
        "document_id": "sample",
        "expected_keyword": "source citations"
    },
    {
        "question": "What is the capital of France?",
        "document_id": "sample",
        "expected_keyword": None
    },
    {
        "question": "Who is the CEO of Microsoft?",
        "document_id": "sample",
        "expected_keyword": None
    },
    {
        "question": "What is the weather today?",
        "document_id": "sample",
        "expected_keyword": None
    }
]


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="document_chunks"
)


passed_count = 0
total_count = len(evaluation_cases)

relevant_distances = []
irrelevant_distances = []


for case in evaluation_cases:
    question = case["question"]
    document_id = case["document_id"]
    expected_keyword = case["expected_keyword"]

    question_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2,
        where={"document_id": document_id},
        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    best_distance = distances[0] if distances else None

    print("\nQuestion:", question)
    print("Documents:", documents)
    print("Distances:", distances)
    print("Best distance:", best_distance)
    print("Expected keyword:", expected_keyword)

    if expected_keyword is None:
        if best_distance is not None:
            irrelevant_distances.append(best_distance)

        passed = (
            best_distance is not None
            and best_distance > 1.0
        )

    else:
        if best_distance is not None:
            relevant_distances.append(best_distance)

        passed = any(
            expected_keyword.lower() in document.lower()
            for document in documents
        )

    print("Passed:", passed)

    if passed:
        passed_count += 1


print("\nEvaluation Summary")
print("------------------")
print("Passed:", passed_count)
print("Total:", total_count)

accuracy = passed_count / total_count

print("Accuracy:", accuracy)


print("\nDistance Analysis")
print("-----------------")
print("Relevant distances:", relevant_distances)
print("Irrelevant distances:", irrelevant_distances)

if relevant_distances:
    print(
        "Highest relevant distance:",
        max(relevant_distances)
    )

if irrelevant_distances:
    print(
        "Lowest irrelevant distance:",
        min(irrelevant_distances)
    )

separation_gap = (
    min(irrelevant_distances)
    - max(relevant_distances)
)

print(
    "Separation gap:",
    separation_gap
)

highest_relevant = max(relevant_distances)
lowest_irrelevant = min(irrelevant_distances)

recommended_threshold = (
    highest_relevant + lowest_irrelevant
) / 2

print(
    "Recommended threshold:",
    recommended_threshold
)

print("\nSelected production threshold:", 1.325)
