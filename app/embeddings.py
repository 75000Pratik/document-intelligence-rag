from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "Employees receive paid leave every year.",
    "Workers get annual vacation days.",
    "The weather is sunny today."
]

embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Embeddings shape:", embeddings.shape)

for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)
    print("First 5 embedding values:", embeddings[index - 1][:5])