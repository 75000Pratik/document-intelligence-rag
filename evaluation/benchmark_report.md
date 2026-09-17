# RAG Retrieval Benchmark Report

## Project

Document Intelligrnce RAG Assistant

## Embedding Model

all-MiniLM-L6-v2

## Retrieval Evaluation

Total evaluation questions: 6

- Relevant questions: 3
- Irrelevant questions: 3
- Evaluation accuracy: 1.0

## Distance Analysis

Highest relevant distances:

0.8871

Lowest irrelevant distance:

1.7628

Separation gap:

0.8757

## Retrieval Threshold

Calculated midpoint threshold:

1.32499

Selected production threshold:

1.325

The production threshold was selected using measured retrieval distances rather than a manually chosen value.

## Chunking Benchmark

Three chunking configurations were compared.

### 1. Fixed Word Chunking

Average distance:

0.8362

### 2. Sentence-Aware Chunking - 20 Words

Average distance:

1.1126

### 3. Sentence-Aware Chunking - 60 Words

Average distance:

1.1207

## Best Strategy

Fixed word chunking performed best on the current evaluation dataset.

Lower retrieval distance indicates stronger semantic similarity between the query and retrieved documnet chunk.

## Evaluation Limitation

This benchmark currently uses a small controlled evaluation set of six questions.

The results are useful for comparing the current retrieval configurations, but they should not be interpreted as a universal accuracy score for all documents or domains.

Future evaluation can be expanded with:

- More documents
- More question types
- Larger ground-truth datasets
- Precision@k and Recall@k
- MRR
- RAG answer-quality metrics

## Conclusion

The final RAG configuration uses:

- Fixed word chunking
- Chunk size: 20 words
- Overlap: 5 words
- Top-k retrieval: 2
- Retrieval threshold: 1.325
- Embedding model: all-MiniLM-L6-v2

These settings were selected using retrieval evaluation rather than assumptions.
