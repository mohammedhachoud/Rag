import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def build_chunk_map(
    chunks: list[dict],
) -> dict[str, dict]:
    """
    Create a mapping from chunk ID to chunk metadata.
    """

    return {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }


def find_first_relevant_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> int | None:
    """
    Return the rank of the first relevant chunk.

    Return None if no relevant chunk was retrieved.
    """

    relevant_set = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            return rank

    return None


def classify_system_result(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    recall: float,
    reciprocal_rank: float,
) -> list[str]:
    """
    Assign one or more retrieval failure tags.
    """

    tags = []

    if recall == 0:
        tags.append("complete_miss")

    elif recall < 1:
        tags.append("partial_recall")

    first_relevant_rank = find_first_relevant_rank(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
    )

    if (
        first_relevant_rank is not None
        and first_relevant_rank > 1
    ):
        tags.append(
            "distractors_before_relevant"
        )

    if not tags:
        tags.append("success")

    return tags


def get_recall(
    query_record: dict,
    system_name: str,
    recall_key: str,
) -> float:
    return float(
        query_record["systems"][system_name][
            "metrics"
        ][recall_key]
    )


def detect_cross_system_signals(
    query_record: dict,
    recall_key: str,
) -> list[str]:
    """
    Compare systems for the same query.

    These are diagnostic signals, not definitive causes.
    """

    signals = []

    recalls = {
        system_name: get_recall(
            query_record,
            system_name,
            recall_key,
        )
        for system_name in (
            "bm25_256",
            "bm25_512",
            "dense_256",
            "dense_512",
        )
    }

    for chunk_size in (256, 512):
        bm25_recall = recalls[
            f"bm25_{chunk_size}"
        ]

        dense_recall = recalls[
            f"dense_{chunk_size}"
        ]

        if dense_recall > bm25_recall:
            signals.append(
                f"dense_better_{chunk_size}"
                "_possible_lexical_gap"
            )

        elif bm25_recall > dense_recall:
            signals.append(
                f"bm25_better_{chunk_size}"
                "_possible_semantic_drift"
            )

    best_256_recall = max(
        recalls["bm25_256"],
        recalls["dense_256"],
    )

    best_512_recall = max(
        recalls["bm25_512"],
        recalls["dense_512"],
    )

    if best_256_recall > best_512_recall:
        signals.append(
            "possible_large_chunk_noise"
        )

    elif best_512_recall > best_256_recall:
        signals.append(
            "possible_small_chunk_context_loss"
        )

    if all(
        recall == 0
        for recall in recalls.values()
    ):
        signals.append("all_systems_missed")

    return signals


def create_chunk_previews(
    chunk_ids: list[str],
    chunk_map: dict[str, dict],
    maximum_characters: int = 300,
) -> list[dict]:
    """
    Return short text previews for manual inspection.
    """

    previews = []

    for chunk_id in chunk_ids:
        chunk = chunk_map.get(chunk_id)

        if chunk is None:
            previews.append(
                {
                    "chunk_id": chunk_id,
                    "text": "[Chunk ID not found]",
                }
            )

            continue

        text = chunk.get("text", "")

        previews.append(
            {
                "chunk_id": chunk_id,
                "text": text[:maximum_characters],
            }
        )

    return previews


def analyze_failures(
    benchmark_data: dict,
    chunk_maps: dict[int, dict[str, dict]],
) -> tuple[list[dict], dict]:
    """
    Analyze every query-system pair.
    """

    top_k = benchmark_data[
        "configuration"
    ]["top_k"]

    precision_key = f"precision_at_{top_k}"
    recall_key = f"recall_at_{top_k}"

    cases = []

    taxonomy_counts = defaultdict(Counter)
    cross_signal_counts = Counter()

    for query_record in benchmark_data["queries"]:
        cross_system_signals = (
            detect_cross_system_signals(
                query_record=query_record,
                recall_key=recall_key,
            )
        )

        cross_signal_counts.update(
            cross_system_signals
        )

        for system_name, system_result in (
            query_record["systems"].items()
        ):
            chunk_size = int(
                system_name.rsplit("_", 1)[1]
            )

            relevant_ids = query_record[
                "relevant_chunks"
            ][str(chunk_size)]

            retrieved_ids = system_result[
                "retrieved_chunk_ids"
            ]

            metrics = system_result["metrics"]

            precision = float(
                metrics[precision_key]
            )

            recall = float(
                metrics[recall_key]
            )

            reciprocal_rank = float(
                metrics["reciprocal_rank"]
            )

            first_relevant_rank = (
                find_first_relevant_rank(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                )
            )

            failure_tags = classify_system_result(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                recall=recall,
                reciprocal_rank=reciprocal_rank,
            )

            taxonomy_counts[
                system_name
            ].update(failure_tags)

            case = {
                "query_id": query_record[
                    "query_id"
                ],
                "query": query_record["query"],
                "query_type": query_record.get(
                    "query_type"
                ),
                "document_id": query_record.get(
                    "document_id"
                ),
                "expected_answer": query_record.get(
                    "expected_answer"
                ),
                "system": system_name,
                "chunk_size": chunk_size,
                "precision": precision,
                "recall": recall,
                "reciprocal_rank": (
                    reciprocal_rank
                ),
                "first_relevant_rank": (
                    first_relevant_rank
                ),
                "latency_ms": system_result[
                    "latency_ms"
                ],
                "failure_tags": failure_tags,
                "cross_system_signals": (
                    cross_system_signals
                ),
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": (
                    retrieved_ids
                ),
                "relevant_chunk_previews": (
                    create_chunk_previews(
                        relevant_ids,
                        chunk_maps[chunk_size],
                    )
                ),
                "retrieved_chunk_previews": (
                    create_chunk_previews(
                        retrieved_ids,
                        chunk_maps[chunk_size],
                    )
                ),
                "manual_failure_label": "",
                "manual_notes": "",
            }

            cases.append(case)

    summary = {
        "top_k": top_k,
        "total_query_system_pairs": len(cases),
        "taxonomy_by_system": {
            system_name: dict(counts)
            for system_name, counts
            in taxonomy_counts.items()
        },
        "cross_system_signal_counts": dict(
            cross_signal_counts
        ),
    }

    return cases, summary


def save_json_analysis(
    output_path: Path,
    summary: dict,
    cases: list[dict],
) -> None:
    """
    Save detailed cases for programmatic analysis.
    """

    failure_cases = [
        case
        for case in cases
        if case["failure_tags"] != ["success"]
    ]

    output = {
        "summary": summary,
        "number_of_failure_cases": len(
            failure_cases
        ),
        "failure_cases": failure_cases,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            output,
            output_file,
            ensure_ascii=False,
            indent=2,
        )


def save_csv_analysis(
    output_path: Path,
    cases: list[dict],
) -> None:
    """
    Save failure cases in a format suitable for
    manual annotation.
    """

    failure_cases = [
        case
        for case in cases
        if case["failure_tags"] != ["success"]
    ]

    fieldnames = [
        "query_id",
        "query",
        "query_type",
        "document_id",
        "expected_answer",
        "system",
        "chunk_size",
        "precision",
        "recall",
        "reciprocal_rank",
        "first_relevant_rank",
        "latency_ms",
        "failure_tags",
        "cross_system_signals",
        "relevant_chunk_ids",
        "retrieved_chunk_ids",
        "relevant_chunk_previews",
        "retrieved_chunk_previews",
        "manual_failure_label",
        "manual_notes",
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for case in failure_cases:
            csv_case = case.copy()

            list_fields = [
                "failure_tags",
                "cross_system_signals",
                "relevant_chunk_ids",
                "retrieved_chunk_ids",
                "relevant_chunk_previews",
                "retrieved_chunk_previews",
            ]

            for field_name in list_fields:
                csv_case[field_name] = json.dumps(
                    csv_case[field_name],
                    ensure_ascii=False,
                )

            writer.writerow(csv_case)


def print_summary(summary: dict) -> None:
    print()
    print("Failure taxonomy summary")
    print()

    for system_name, counts in summary[
        "taxonomy_by_system"
    ].items():
        print(system_name)

        for category, count in counts.items():
            print(f"  {category}: {count}")

        print()

    print("Cross-system diagnostic signals")

    for signal, count in summary[
        "cross_system_signal_counts"
    ].items():
        print(f"  {signal}: {count}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    benchmark_path = (
        project_root
        / "results"
        / "benchmark_results.json"
    )

    chunks_256_path = (
        project_root
        / "data"
        / "processed"
        / "chunks_256.json"
    )

    chunks_512_path = (
        project_root
        / "data"
        / "processed"
        / "chunks_512.json"
    )

    output_json_path = (
        project_root
        / "results"
        / "failure_analysis.json"
    )

    output_csv_path = (
        project_root
        / "results"
        / "failure_analysis.csv"
    )

    benchmark_data = load_json(
        benchmark_path
    )

    chunks_256 = load_json(
        chunks_256_path
    )

    chunks_512 = load_json(
        chunks_512_path
    )

    chunk_maps = {
        256: build_chunk_map(chunks_256),
        512: build_chunk_map(chunks_512),
    }

    cases, summary = analyze_failures(
        benchmark_data=benchmark_data,
        chunk_maps=chunk_maps,
    )

    save_json_analysis(
        output_path=output_json_path,
        summary=summary,
        cases=cases,
    )

    save_csv_analysis(
        output_path=output_csv_path,
        cases=cases,
    )

    print_summary(summary)

    print(
        f"JSON saved to: "
        f"{output_json_path.resolve()}"
    )

    print(
        f"CSV saved to: "
        f"{output_csv_path.resolve()}"
    )


if __name__ == "__main__":
    main()