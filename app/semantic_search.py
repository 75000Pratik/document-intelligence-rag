from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Employees receive 20 paid leave days every year.",
    "The company provides health insurance to employees.",
    "Employees must change their passwords every 90 days."
]

chunk_embeddings = model.encode(chunks)

question = "How many vacation days do employee get?"

question_embedding = model.encode(question)

similarities = cosine_similarity(
    [question_embedding],
    chunk_embeddings
)

print("Question:")
print(question)

print("\nSimilarity scores:")

for index, score in enumerate(similarities[0], start=1):
    print(f"Chunk {index}: {score:.4f}")

scores = similarities[0]

top_indices = scores.argsort()[::-1][:2]

print("\nTop 2 relevant chunks:")

for rank, index in enumerate(top_indices, start=1):
    print(f"\nRank {rank}")
    print("Score:", round(scores[index], 4))
    print("Chunk:", chunks[index])