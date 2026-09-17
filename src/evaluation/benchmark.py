import json
from pathlib import Path
from time import perf_counter

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.evaluation.metrics import calculate_query_metrics
from src.retrieval.bm25 import (
    build_bm25_index,
    load_chunks,
    search_bm25,
)
from src.retrieval.dense import (
    COLLECTIONS,
    MODEL_NAME,
    QDRANT_URL,
    search_dense,
)


TOP_K = 5
CHUNK_SIZES = (256, 512)


def load_evaluation_queries(
    evaluation_path: Path,
) -> list[dict]:
    """
    Load synthetic evaluation queries from JSON.
    """

    with evaluation_path.open(
        "r",
        encoding="utf-8",
    ) as evaluation_file:
        evaluation_queries = json.load(evaluation_file)

    if not evaluation_queries:
        raise ValueError(
            "The evaluation dataset is empty."
        )

    return evaluation_queries


def validate_evaluation_queries(
    evaluation_queries: list[dict],
) -> None:
    """
    Verify that every query contains the required fields.
    """

    required_fields = {
        "query_id",
        "query",
        "relevant_chunks_256",
        "relevant_chunks_512",
    }

    query_ids = set()

    for evaluation_item in evaluation_queries:
        missing_fields = (
            required_fields - evaluation_item.keys()
        )

        if missing_fields:
            raise ValueError(
                f"Query is missing fields: {missing_fields}"
            )

        query_id = evaluation_item["query_id"]

        if query_id in query_ids:
            raise ValueError(
                f"Duplicate query ID: {query_id}"
            )

        query_ids.add(query_id)

        if not evaluation_item["query"].strip():
            raise ValueError(
                f"Query {query_id} has empty text."
            )

        for chunk_size in CHUNK_SIZES:
            relevant_key = (
                f"relevant_chunks_{chunk_size}"
            )

            if not evaluation_item[relevant_key]:
                raise ValueError(
                    f"Query {query_id} has no relevant "
                    f"{chunk_size}-token chunks."
                )


def create_result_record(
    retrieval_results: list[dict],
    relevant_ids: list[str],
    latency_ms: float,
    k: int,
) -> dict:
    """
    Convert retrieval output into a benchmark record.
    """

    retrieved_ids = [
        result["chunk_id"]
        for result in retrieval_results
        if result["chunk_id"] is not None
    ]

    scores = [
        float(result["score"])
        for result in retrieval_results
    ]

    metrics = calculate_query_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=k,
    )

    return {
        "retrieved_chunk_ids": retrieved_ids,
        "scores": scores,
        "latency_ms": latency_ms,
        "metrics": metrics,
    }


def summarize_system(
    query_results: list[dict],
    k: int,
) -> dict:
    """
    Average metrics and calculate latency percentiles.
    """

    if not query_results:
        raise ValueError(
            "Cannot summarize an empty result list."
        )

    precision_key = f"precision_at_{k}"
    recall_key = f"recall_at_{k}"

    precision_values = [
        result["metrics"][precision_key]
        for result in query_results
    ]

    recall_values = [
        result["metrics"][recall_key]
        for result in query_results
    ]

    reciprocal_ranks = [
        result["metrics"]["reciprocal_rank"]
        for result in query_results
    ]

    latency_values = [
        result["latency_ms"]
        for result in query_results
    ]

    return {
        "number_of_queries": len(query_results),
        precision_key: float(
            np.mean(precision_values)
        ),
        recall_key: float(
            np.mean(recall_values)
        ),
        "mrr": float(
            np.mean(reciprocal_ranks)
        ),
        "p50_latency_ms": float(
            np.percentile(latency_values, 50)
        ),
        "p95_latency_ms": float(
            np.percentile(latency_values, 95)
        ),
        "mean_latency_ms": float(
            np.mean(latency_values)
        ),
    }


def print_summary(
    summary: dict,
    k: int,
) -> None:
    """
    Print a compact comparison table.
    """

    precision_key = f"precision_at_{k}"
    recall_key = f"recall_at_{k}"

    print()
    print("Benchmark summary")
    print()

    print(
        f"{'System':<12}"
        f"{f'P@{k}':>10}"
        f"{f'R@{k}':>10}"
        f"{'MRR':>10}"
        f"{'p50 ms':>12}"
        f"{'p95 ms':>12}"
    )

    print("-" * 66)

    for system_name, values in summary.items():
        print(
            f"{system_name:<12}"
            f"{values[precision_key]:>10.4f}"
            f"{values[recall_key]:>10.4f}"
            f"{values['mrr']:>10.4f}"
            f"{values['p50_latency_ms']:>12.2f}"
            f"{values['p95_latency_ms']:>12.2f}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    evaluation_path = (
        project_root
        / "data"
        / "evaluation"
        / "synthetic_eval.json"
    )

    results_path = (
        project_root
        / "results"
        / "benchmark_results.json"
    )

    # 1. Load and validate the evaluation dataset

    evaluation_queries = load_evaluation_queries(
        evaluation_path
    )

    validate_evaluation_queries(
        evaluation_queries
    )

    print(
        f"Loaded {len(evaluation_queries)} "
        "evaluation queries."
    )

    if len(evaluation_queries) != 100:
        print(
            "Warning: synthetic_eval.json does not "
            "contain exactly 100 queries."
        )

    # 2. Load both chunk corpora

    chunks_by_size = {}

    for chunk_size in CHUNK_SIZES:
        chunks_path = (
            project_root
            / "data"
            / "processed"
            / f"chunks_{chunk_size}.json"
        )

        chunks = load_chunks(chunks_path)

        chunks_by_size[chunk_size] = chunks

        print(
            f"Loaded {len(chunks)} chunks "
            f"for the {chunk_size}-token corpus."
        )

    # 3. Build BM25 indexes

    bm25_indexes = {}

    for chunk_size in CHUNK_SIZES:
        bm25_indexes[chunk_size] = (
            build_bm25_index(
                chunks_by_size[chunk_size]
            )
        )

        print(
            f"Built BM25 index for "
            f"{chunk_size}-token chunks."
        )

    # 4. Load dense retrieval dependencies

    print(f"Loading dense model: {MODEL_NAME}")

    dense_model = SentenceTransformer(
        MODEL_NAME
    )

    qdrant_client = QdrantClient(
        url=QDRANT_URL
    )

    for chunk_size in CHUNK_SIZES:
        collection_name = COLLECTIONS[chunk_size]

        if not qdrant_client.collection_exists(
            collection_name
        ):
            raise RuntimeError(
                f"Qdrant collection does not exist: "
                f"{collection_name}. Run "
                f"ingest_embeddings.py first."
            )

        print(
            f"Found Qdrant collection: "
            f"{collection_name}"
        )

    # 5. Warm up dense retrieval

    first_query = evaluation_queries[0]["query"]

    print("Warming up dense retrieval...")

    for chunk_size in CHUNK_SIZES:
        search_dense(
            query=first_query,
            model=dense_model,
            client=qdrant_client,
            collection_name=COLLECTIONS[
                chunk_size
            ],
            top_k=TOP_K,
        )

    # 6. Prepare result containers

    system_results = {
        "bm25_256": [],
        "bm25_512": [],
        "dense_256": [],
        "dense_512": [],
    }

    per_query_results = []

    # 7. Evaluate every query

    total_queries = len(evaluation_queries)

    for position, evaluation_item in enumerate(
        evaluation_queries,
        start=1,
    ):
        query_id = evaluation_item["query_id"]
        query = evaluation_item["query"]

        print(
            f"Evaluating {position}/{total_queries}: "
            f"{query_id}"
        )

        query_record = {
            "query_id": query_id,
            "query": query,
            "query_type": evaluation_item.get(
                "query_type"
            ),
            "document_id": evaluation_item.get(
                "document_id"
            ),
            "expected_answer": evaluation_item.get(
                "expected_answer"
            ),
            "relevant_chunks": {
                "256": evaluation_item[
                    "relevant_chunks_256"
                ],
                "512": evaluation_item[
                    "relevant_chunks_512"
                ],
            },
            "systems": {},
        }

        for chunk_size in CHUNK_SIZES:
            relevant_ids = evaluation_item[
                f"relevant_chunks_{chunk_size}"
            ]

            # BM25 retrieval

            bm25_system_name = (
                f"bm25_{chunk_size}"
            )

            bm25_start_time = perf_counter()

            bm25_results = search_bm25(
                query=query,
                bm25=bm25_indexes[chunk_size],
                chunks=chunks_by_size[chunk_size],
                top_k=TOP_K,
            )

            bm25_latency_ms = (
                perf_counter() - bm25_start_time
            ) * 1000

            bm25_record = create_result_record(
                retrieval_results=bm25_results,
                relevant_ids=relevant_ids,
                latency_ms=bm25_latency_ms,
                k=TOP_K,
            )

            query_record["systems"][
                bm25_system_name
            ] = bm25_record

            system_results[
                bm25_system_name
            ].append(bm25_record)

            # Dense retrieval

            dense_system_name = (
                f"dense_{chunk_size}"
            )

            dense_start_time = perf_counter()

            dense_results = search_dense(
                query=query,
                model=dense_model,
                client=qdrant_client,
                collection_name=COLLECTIONS[
                    chunk_size
                ],
                top_k=TOP_K,
            )

            dense_latency_ms = (
                perf_counter() - dense_start_time
            ) * 1000

            dense_record = create_result_record(
                retrieval_results=dense_results,
                relevant_ids=relevant_ids,
                latency_ms=dense_latency_ms,
                k=TOP_K,
            )

            query_record["systems"][
                dense_system_name
            ] = dense_record

            system_results[
                dense_system_name
            ].append(dense_record)

        per_query_results.append(
            query_record
        )

    # 8. Calculate global summaries

    summary = {}

    for system_name, results in system_results.items():
        summary[system_name] = summarize_system(
            query_results=results,
            k=TOP_K,
        )

    # 9. Create and save benchmark output

    benchmark_output = {
        "configuration": {
            "top_k": TOP_K,
            "embedding_model": MODEL_NAME,
            "number_of_queries": total_queries,
            "chunk_sizes": list(CHUNK_SIZES),
            "dense_collections": {
                str(chunk_size): COLLECTIONS[
                    chunk_size
                ]
                for chunk_size in CHUNK_SIZES
            },
        },
        "summary": summary,
        "queries": per_query_results,
    }

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as results_file:
        json.dump(
            benchmark_output,
            results_file,
            ensure_ascii=False,
            indent=2,
        )

    # 10. Display comparison

    print_summary(
        summary=summary,
        k=TOP_K,
    )

    print()
    print(
        f"Benchmark results saved to: "
        f"{results_path.resolve()}"
    )


if __name__ == "__main__":
    main()