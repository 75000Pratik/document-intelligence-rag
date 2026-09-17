import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="document_chunks"
)


questions = [
    "What does Retrieval-Augmented Generation combine?",
    "What will this project answer questions using?",
    "Does the project provide source citations?"
]


document_versions = {
    "old_chunking": "sample",
    "sentence_chunking_60": "sample_sentence",
    "sentence_chunking_20": "sample_sentence20"
}


results_summary = {}


for version_name, document_id in document_versions.items():

    print(f"\n=== {version_name} ===")

    distances_for_version = []

    for question in questions:

        question_embedding = model.encode(
            question
        ).tolist()

        results = collection.query(
            query_embeddings=[
                question_embedding
            ],
            n_results=2,
            where={
                "document_id": document_id
            },
            include=[
                "documents",
                "distances"
            ]
        )

        distances = results["distances"][0]

        if distances:

            best_distance = distances[0]

            distances_for_version.append(
                best_distance
            )

            print("\nQuestion:", question)

            print(
                "Best distance:",
                best_distance
            )

        else:

            print("\nQuestion:", question)

            print(
                "No retrieval result"
            )

    if distances_for_version:

        average_distance = (
            sum(distances_for_version)
            / len(distances_for_version)
        )

    else:

        average_distance = None

    results_summary[version_name] = {
        "distances": distances_for_version,
        "average_distance": average_distance
    }


print("\n\n=== Comparison Summary ===")

for version_name, data in results_summary.items():

    print(f"\n{version_name}")

    print(
        "Distances:",
        data["distances"]
    )

    print(
        "Average distance:",
        data["average_distance"]
    )


print("\n\n=== Final Ranking ===")

ranking = sorted(
    results_summary.items(),
    key=lambda item: item[1]["average_distance"]
)

for position, (
    version_name,
    data
) in enumerate(
    ranking,
    start=1
):

    print(
        f"{position}. "
        f"{version_name} "
        f"- Average distance: "
        f"{data['average_distance']}"
    )


best_strategy = ranking[0][0]

print(
    "\nBest strategy:",
    best_strategy
)