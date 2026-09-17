def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    top_k_ids = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    relevant_retrieved = len(
        set(top_k_ids) & relevant_set
    )

    return relevant_retrieved / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        raise ValueError(
            "Recall cannot be calculated without relevant chunks."
        )

    top_k_ids = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    relevant_retrieved = len(
        set(top_k_ids) & relevant_set
    )

    return relevant_retrieved / len(relevant_set)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    relevant_set = set(relevant_ids)

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            return 1.0 / rank

    return 0.0


def calculate_query_metrics(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5,
) -> dict:
    return {
        f"precision_at_{k}": precision_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        f"recall_at_{k}": recall_at_k(
            retrieved_ids,
            relevant_ids,
            k,
        ),
        "reciprocal_rank": reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        ),
    }


def main() -> None:
    relevant_ids = [
        "chunk_2",
        "chunk_4",
    ]

    retrieved_ids = [
        "chunk_1",
        "chunk_2",
        "chunk_3",
        "chunk_4",
        "chunk_5",
    ]

    metrics = calculate_query_metrics(
        retrieved_ids=retrieved_ids,
        relevant_ids=relevant_ids,
        k=5,
    )

    print(metrics)


if __name__ == "__main__":
    main()