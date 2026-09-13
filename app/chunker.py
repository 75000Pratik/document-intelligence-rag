def chunk_text(text, chunk_size=20, overlap=5):
    words = text.split()

    chunks=[]

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

sample_text = """
Retrieval-Augmented Generation combines information retrieval
with language models. Documents are first processed and divided
into smaller chunks. These chunks can later be converted into
embeddings and stored inside a vector database.
"""    

chunks = chunk_text(sample_text, chunk_size=20, overlap=5)

print("Number of chunks:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print(chunk)