import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "evaluation" / "out_of_domain_eval.json"
EXPECTED_TYPE_COUNTS = {
    "direct_lexical": 20,
    "paraphrased_semantic": 30,
    "definitions_concepts": 15,
    "lists_numeric_facts": 15,
    "comparison_reasoning": 10,
    "multi_chunk": 10,
}
REQUIRED_FIELDS = {
    "query_id",
    "query",
    "query_type",
    "document_id",
    "expected_answer",
    "relevant_chunks_256",
    "relevant_chunks_512",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def validate(dataset_path: Path) -> list[str]:
    errors = []
    records = load_json(dataset_path)

    if not isinstance(records, list):
        return ["The dataset root must be a JSON array."]

    chunks_by_size = {
        size: load_json(ROOT / "data" / "processed" / f"chunks_{size}.json")
        for size in (256, 512)
    }
    chunk_indexes = {
        size: {chunk["chunk_id"]: chunk for chunk in chunks}
        for size, chunks in chunks_by_size.items()
    }

    if len(records) != 100:
        errors.append(f"Expected 100 queries; found {len(records)}.")

    query_ids = [record.get("query_id") for record in records]
    queries = [record.get("query") for record in records]
    if len(query_ids) != len(set(query_ids)):
        errors.append("query_id values must be unique.")
    if len(queries) != len(set(queries)):
        errors.append("query values must be unique.")

    type_counts = Counter(record.get("query_type") for record in records)
    if type_counts != Counter(EXPECTED_TYPE_COUNTS):
        errors.append(
            "Unexpected query type distribution: "
            f"{dict(sorted(type_counts.items(), key=lambda item: str(item[0])))}"
        )

    for index, record in enumerate(records, start=1):
        label = record.get("query_id", f"record {index}")
        missing = REQUIRED_FIELDS - set(record)
        extra = set(record) - REQUIRED_FIELDS
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}.")
        if extra:
            errors.append(f"{label}: unexpected fields {sorted(extra)}.")

        if not isinstance(record.get("query"), str) or not record.get("query", "").strip():
            errors.append(f"{label}: query must be a non-empty string.")
        if not isinstance(record.get("expected_answer"), str) or not record.get("expected_answer", "").strip():
            errors.append(f"{label}: expected_answer must be a non-empty string.")

        for size in (256, 512):
            field = f"relevant_chunks_{size}"
            relevant_ids = record.get(field)
            if not isinstance(relevant_ids, list):
                errors.append(f"{label}: {field} must be an array.")
                continue
            if not 1 <= len(relevant_ids) <= 5:
                errors.append(
                    f"{label}: {field} must contain 1-5 IDs; "
                    f"found {len(relevant_ids)}."
                )
            if len(relevant_ids) != len(set(relevant_ids)):
                errors.append(f"{label}: {field} contains duplicate IDs.")

            for chunk_id in relevant_ids:
                chunk = chunk_indexes[size].get(chunk_id)
                if chunk is None:
                    errors.append(f"{label}: unknown {size}-token chunk {chunk_id!r}.")
                elif chunk["document_id"] != record.get("document_id"):
                    errors.append(
                        f"{label}: {chunk_id!r} belongs to "
                        f"{chunk['document_id']!r}, not {record.get('document_id')!r}."
                    )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the retrieval evaluation dataset and its canonical chunk IDs."
    )
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    errors = validate(args.dataset)
    if errors:
        print(f"Validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    records = load_json(args.dataset)
    counts = Counter(record["query_type"] for record in records)
    print(f"Validated {len(records)} queries in {args.dataset.resolve()}")
    for query_type, count in EXPECTED_TYPE_COUNTS.items():
        print(f"- {query_type}: {counts[query_type]}")
    for size in (256, 512):
        cardinalities = Counter(len(record[f"relevant_chunks_{size}"]) for record in records)
        print(f"- {size}-token relevance cardinalities: {dict(sorted(cardinalities.items()))}")


if __name__ == "__main__":
    main()
