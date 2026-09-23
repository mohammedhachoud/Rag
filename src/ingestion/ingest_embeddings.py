import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import uuid

from qdrant_client import QdrantClient, models

MODEL_NAME = "BAAI/bge-small-en-v1.5"

QDRANT_URL = "http://localhost:6333"

COLLECTIONS = {
    256: "rag_dense_256",
    512: "rag_dense_512",
}

UPSERT_BATCH_SIZE = 100

client = QdrantClient(url=QDRANT_URL)

def load_chunks(chunks_path: Path) -> list[dict]:
    with chunks_path.open("r", encoding="utf-8") as chunks_file:
        chunks = json.load(chunks_file)

    return chunks

def generate_embeddings(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings

def create_collection_if_missing(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if client.collection_exists(collection_name):
        print(f"Collection already exists: {collection_name}")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_size",
        field_schema=models.PayloadSchemaType.INTEGER,
    )

    print(f"Created collection: {collection_name}")

def create_point_id(chunk_id: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            chunk_id,
        )
    )

def upsert_embeddings(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    batch_size: int = 100,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(
            "The number of chunks and embeddings must be equal."
        )

    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(
            batch_start + batch_size,
            len(chunks),
        )

        points = []

        for index in range(batch_start, batch_end):
            chunk = chunks[index]
            embedding = embeddings[index]

            point = models.PointStruct(
                id=create_point_id(chunk["chunk_id"]),
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "file_name": chunk["file_name"],
                    "chunk_size": chunk["chunk_size"],
                    "chunk_number": chunk["chunk_number"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "end_token": chunk["end_token"],
                    "token_count": chunk["token_count"],
                    "text": chunk["text"],
                },
            )

            points.append(point)

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        print(
            f"Upserted points {batch_start + 1}–{batch_end} "
            f"into {collection_name}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    model = SentenceTransformer(MODEL_NAME)
    vector_size = model.get_sentence_embedding_dimension()

    if vector_size is None:
        raise ValueError(
            "Could not determine the embedding dimension."
        )

    client = QdrantClient(url=QDRANT_URL)

    for chunk_size, collection_name in COLLECTIONS.items():
        chunks_path = (
            project_root
            / "data"
            / "processed"
            / f"chunks_{chunk_size}.json"
        )

        print()
        print(f"Processing {chunk_size}-token chunks...")

        chunks = load_chunks(chunks_path)

        print(f"Loaded {len(chunks)} chunks.")

        embeddings = generate_embeddings(
            chunks=chunks,
            model=model,
            batch_size=32,
        )

        print(f"Generated embeddings: {embeddings.shape}")

        create_collection_if_missing(
            client=client,
            collection_name=collection_name,
            vector_size=vector_size,
        )

        upsert_embeddings(
            client=client,
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
            batch_size=UPSERT_BATCH_SIZE,
        )

        stored_count = client.count(
            collection_name=collection_name,
            exact=True,
        ).count

        print(
            f"{collection_name} contains "
            f"{stored_count} points."
        )


if __name__ == "__main__":
    main()