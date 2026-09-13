import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

collection = client.create_collection(
    name = "document_chunks"
)

chunks = [
    "Employees receive 20 paid leave days every year.",
    "The company provides health insurance to employees.",
    "Employees must change their passwords every 90 days."
]

embeddings = model.encode(chunks).tolist()

collection.add(
    ids = ["chunk1", "chunk2", "chunk3"],
    documents = chunks,
    embeddings = embeddings
)

question = "How many vacation days do employees get?"

question_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings = [question_embedding],
    n_results = 2
)

print("\nQuestion:")
print(question)

print("\nTop results:")

for index, document in enumerate(results["documents"][0], start=1):
    print(f"\nRank {index}")
    print(document)
