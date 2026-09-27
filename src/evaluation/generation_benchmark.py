"""Compare Granite generation with three versus five retrieved chunks."""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.evaluation.benchmark import (
    load_evaluation_queries,
    validate_evaluation_queries,
)
from src.config import settings
from src.generation.ollama_client import OllamaGenerator
from src.generation.prompt_builder import build_citation, build_messages
from src.retrieval.dense import COLLECTIONS, MODEL_NAME, QDRANT_URL, search_dense


REFUSAL = "I cannot answer this question from the provided sources."
CITATION_PATTERN = re.compile(r"\[([^\[\]\r\n]+)\]")
WORD_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_CORRECTNESS_THRESHOLD = settings.correctness_threshold


@dataclass(frozen=True)
class GenerationConfiguration:
    name: str
    model: str
    context_chunks: int


DEFAULT_CONFIGURATIONS = (
    GenerationConfiguration("Granite 3B - Top 3 chunks", settings.ollama_model, 3),
    GenerationConfiguration("Granite 3B - Top 5 chunks", settings.ollama_model, 5),
    GenerationConfiguration("Granite 8B - Top 3 chunks", settings.ollama_model_8b, 3),
    GenerationConfiguration("Granite 8B - Top 5 chunks", settings.ollama_model_8b, 5),
)


def extract_citations(answer: str) -> list[str]:
    """Return citation text without surrounding square brackets."""
    return [match.strip() for match in CITATION_PATTERN.findall(answer)]


def content_word_recall(answer: str, expected_answer: str) -> float:
    """A small diagnostic for answer coverage; it is not an LLM judge."""
    expected_words = set(WORD_PATTERN.findall(expected_answer.lower()))
    answer_words = set(WORD_PATTERN.findall(answer.lower()))
    if not expected_words:
        return 0.0
    return len(expected_words & answer_words) / len(expected_words)


def semantic_answer_similarity(
    answer: str,
    expected_answer: str,
    model: SentenceTransformer,
) -> float:
    """Cosine similarity between the generated and gold answers."""
    answer_without_citations = CITATION_PATTERN.sub("", answer).strip()
    embeddings = model.encode(
        [answer_without_citations, expected_answer],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return float(np.dot(embeddings[0], embeddings[1]))


def score_answer(
    answer: str,
    context_chunks: list[dict[str, Any]],
    relevant_chunk_ids: list[str],
    expected_answer: str,
) -> dict[str, Any]:
    """Score source validity and whether citations point at gold evidence.

    A grounded answer is a non-refusal containing citations, where every citation
    is present in the supplied context and at least one points to a gold-relevant
    chunk. This deliberately measures observable grounding, not semantic
    entailment; the full records are exported for human review.
    """
    citations = extract_citations(answer)
    citation_to_ids: dict[str, set[str]] = {}
    for chunk in context_chunks:
        citation_to_ids.setdefault(build_citation(chunk), set()).add(chunk["chunk_id"])

    relevant = set(relevant_chunk_ids)
    valid_flags = [citation in citation_to_ids for citation in citations]
    correct_flags = [
        bool(citation_to_ids.get(citation, set()) & relevant)
        for citation in citations
    ]
    citation_count = len(citations)
    valid_count = sum(valid_flags)
    correct_count = sum(correct_flags)
    citation_validity = valid_count / citation_count if citation_count else 0.0
    citation_correctness = correct_count / citation_count if citation_count else 0.0
    refused = answer.strip() == REFUSAL
    grounded = bool(citations) and not refused and all(valid_flags) and any(correct_flags)

    return {
        "grounded": grounded,
        "refused": refused,
        "citations": citations,
        "citation_count": citation_count,
        "valid_citation_count": valid_count,
        "correct_citation_count": correct_count,
        "citation_validity": citation_validity,
        "citation_correctness": citation_correctness,
        "expected_answer_word_recall": content_word_recall(answer, expected_answer),
    }


def percentile(values: list[float], quantile: int) -> float:
    return float(np.percentile(values, quantile))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [record["response_time_seconds"] for record in records]
    cpu_memory = [
        record["resource_usage"]["cpu_memory_mb"] for record in records
        if record["resource_usage"]["cpu_memory_mb"] is not None
    ]
    vram = [
        record["resource_usage"]["vram_mb"] for record in records
        if record["resource_usage"]["vram_mb"] is not None
    ]
    return {
        "questions": len(records),
        "context_chunks": records[0]["context_chunks"] if records else 0,
        "grounded_answer_rate": mean(record["scores"]["grounded"] for record in records),
        "correct_citation_rate": mean(
            record["scores"]["citation_correctness"] for record in records
        ),
        "valid_citation_rate": mean(
            record["scores"]["citation_validity"] for record in records
        ),
        "expected_answer_word_recall": mean(
            record["scores"]["expected_answer_word_recall"] for record in records
        ),
        "answer_correctness_rate": mean(
            record["scores"]["answer_correct"] for record in records
        ),
        "mean_answer_similarity": mean(
            record["scores"]["answer_similarity"] for record in records
        ),
        "mean_response_time_seconds": mean(latencies),
        "median_response_time_seconds": median(latencies),
        "p95_response_time_seconds": percentile(latencies, 95),
        "mean_cpu_memory_mb": mean(cpu_memory) if cpu_memory else None,
        "peak_cpu_memory_mb": max(cpu_memory) if cpu_memory else None,
        "mean_vram_mb": mean(vram) if vram else None,
        "peak_vram_mb": max(vram) if vram else None,
    }


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    print("\nGeneration benchmark summary\n")
    print(
        f"{'Generator':<34}{'Context':>9}{'Correct':>10}{'Grounded':>11}"
        f"{'Citations':>11}{'Mean sec':>11}{'CPU MB':>10}{'VRAM MB':>10}"
    )
    print("-" * 106)
    for name, values in summary.items():
        print(
            f"{name:<34}{values['context_chunks']:>9}"
            f"{values['answer_correctness_rate']:>10.1%}"
            f"{values['grounded_answer_rate']:>11.1%}"
            f"{values['correct_citation_rate']:>11.1%}"
            f"{values['mean_response_time_seconds']:>11.2f}"
            f"{(values['peak_cpu_memory_mb'] or 0):>10.0f}"
            f"{(values['peak_vram_mb'] or 0):>10.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Run only the first N questions.")
    parser.add_argument("--chunk-size", type=int, choices=(256, 512), default=512)
    parser.add_argument("--three-b-model", default=settings.ollama_model)
    parser.add_argument("--eight-b-model", default=settings.ollama_model_8b)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--correctness-threshold", type=float, default=DEFAULT_CORRECTNESS_THRESHOLD,
        help="Minimum semantic similarity counted as correct (default: 0.75).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.correctness_threshold <= 1.0:
        raise ValueError("--correctness-threshold must be between 0 and 1.")
    project_root = Path(__file__).resolve().parents[2]
    evaluation_path = project_root / "data" / "evaluation" / "synthetic_eval.json"
    output_path = args.output or project_root / "results" / "generation_benchmark_results.json"

    queries = load_evaluation_queries(evaluation_path)
    validate_evaluation_queries(queries)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero.")
        queries = queries[: args.limit]

    configurations = (
        GenerationConfiguration("Granite 3B - Top 3 chunks", args.three_b_model, 3),
        GenerationConfiguration("Granite 3B - Top 5 chunks", args.three_b_model, 5),
        GenerationConfiguration("Granite 8B - Top 3 chunks", args.eight_b_model, 3),
        GenerationConfiguration("Granite 8B - Top 5 chunks", args.eight_b_model, 5),
    )

    qdrant_client = QdrantClient(**settings.qdrant_kwargs())
    collection = COLLECTIONS[args.chunk_size]
    try:
        collection_exists = qdrant_client.collection_exists(collection)
    except Exception as error:
        raise RuntimeError(
            f"Cannot connect to Qdrant at {QDRANT_URL}; start Qdrant before "
            "running the generation benchmark."
        ) from error
    if not collection_exists:
        raise RuntimeError(f"Missing Qdrant collection {collection}; run ingestion first.")
    embedding_model = SentenceTransformer(MODEL_NAME)

    records_by_configuration = {configuration.name: [] for configuration in configurations}

    # Retrieve once, then benchmark one model at a time. This prevents Ollama
    # from keeping both Granite models resident in CPU/GPU memory together.
    retrieved_queries = []
    for position, item in enumerate(queries, start=1):
        print(f"Retrieving {position}/{len(queries)}: {item['query_id']}")
        retrieved = search_dense(
            query=item["query"], model=embedding_model, client=qdrant_client,
            collection_name=collection, top_k=5,
        )
        retrieved_queries.append((item, retrieved))

    models = dict.fromkeys(configuration.model for configuration in configurations)
    for model in models:
        generator = OllamaGenerator(model=model)
        model_configurations = [
            configuration for configuration in configurations
            if configuration.model == model
        ]
        print(f"\nRunning {model}")
        for configuration in model_configurations:
            print(f"  {configuration.name}")
            for position, (item, retrieved) in enumerate(retrieved_queries, start=1):
                print(f"    Generating {position}/{len(retrieved_queries)}: {item['query_id']}")
                context = retrieved[: configuration.context_chunks]
                started = perf_counter()
                generation = generator.generate(
                    build_messages(question=item["query"], chunks=context)
                )
                elapsed = perf_counter() - started
                scores = score_answer(
                    answer=generation["answer"],
                    context_chunks=context,
                    relevant_chunk_ids=item[f"relevant_chunks_{args.chunk_size}"],
                    expected_answer=item["expected_answer"],
                )
                answer_similarity = semantic_answer_similarity(
                    generation["answer"], item["expected_answer"], embedding_model
                )
                scores["answer_similarity"] = answer_similarity
                scores["answer_correct"] = (
                    answer_similarity >= args.correctness_threshold
                )
                resource_usage = generator.resource_usage()
                records_by_configuration[configuration.name].append({
                    "query_id": item["query_id"],
                    "question": item["query"],
                    "expected_answer": item["expected_answer"],
                    "model": generation["model"],
                    "context_chunks": configuration.context_chunks,
                    "context_chunk_ids": [chunk["chunk_id"] for chunk in context],
                    "answer": generation["answer"],
                    "response_time_seconds": elapsed,
                    "resource_usage": resource_usage,
                    "generation_stats": {key: value for key, value in generation.items()
                                         if key not in {"answer", "model"}},
                    "scores": scores,
                })
        print(f"Unloading {model} from CPU/GPU memory")
        generator.unload()

    summary = {
        name: summarize(records) for name, records in records_by_configuration.items()
    }
    output = {
        "configuration": {
            "chunk_size": args.chunk_size,
            "retriever": "dense",
            "embedding_model": MODEL_NAME,
            "answer_correctness_threshold": args.correctness_threshold,
            "systems": [asdict(configuration) for configuration in configurations],
            "scoring_note": (
                "Grounded means all cited sources were supplied and at least one citation "
                "maps to a gold-relevant chunk. Citation correctness is the fraction of "
                "citations mapping to gold-relevant chunks."
            ),
        },
        "summary": summary,
        "queries": records_by_configuration,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(summary)
    print(f"\nResults saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
