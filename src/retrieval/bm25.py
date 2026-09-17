import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

def load_chunks(chunks_path: Path) -> list[dict]:
    with chunks_path.open("r", encoding="utf-8") as chunks_file:
        chunks = json.load(chunks_file)

    return chunks

def build_bm25_index(
    chunks: list[dict],
) -> BM25Okapi:
    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25

def search_bm25(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    top_indexes = ranked_indexes[:top_k]

    results = []

    for rank, chunk_index in enumerate(top_indexes, start=1):
        chunk = chunks[chunk_index]

        results.append(
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "score": float(scores[chunk_index]),
                "text": chunk["text"],
            }
        )

    return results

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    chunks_path = (
        project_root
        / "data"
        / "processed"
        / "chunks_512.json"
    )

    chunks = load_chunks(chunks_path)

    bm25 = build_bm25_index(chunks)

    query = "What are the four AI RMF functions?"

    results = search_bm25(
        query=query,
        bm25=bm25,
        chunks=chunks,
        top_k=5,
    )

    print(f"Query: {query}")
    print()

    for result in results:
        print(
            f"Rank {result['rank']} | "
            f"Score: {result['score']:.4f} | "
            f"Chunk: {result['chunk_id']}"
        )

        print(result["text"][:300])
        print()


if __name__ == "__main__":
    main()