from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.config import settings


MODEL_NAME = settings.embedding_model
QDRANT_URL = settings.qdrant_url

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

COLLECTIONS = settings.collections

def encode_query(
    query: str,
    model: SentenceTransformer,
) -> list[float]:
    instructed_query = QUERY_INSTRUCTION + query

    query_embedding = model.encode(
        instructed_query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return query_embedding.tolist()

def search_dense(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 5,
) -> list[dict]:
    query_vector = encode_query(
        query=query,
        model=model,
    )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results = []

    for rank, point in enumerate(response.points, start=1):
        payload = point.payload or {}

        results.append(
            {
                "rank": rank,
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "score": float(point.score),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "text": payload.get("text", ""),
            }
        )

    return results


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(**settings.qdrant_kwargs())

    chunk_size = 256
    collection_name = COLLECTIONS[chunk_size]

    if not client.collection_exists(collection_name):
        raise RuntimeError(
            f"Collection does not exist: {collection_name}. "
            "Run ingest_embeddings.py first."
        )

    query = "What are the four AI RMF functions?"

    results = search_dense(
        query=query,
        model=model,
        client=client,
        collection_name=collection_name,
        top_k=settings.retrieval_top_k,
    )

    print(f"Query: {query}")
    print(f"Collection: {collection_name}")
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
