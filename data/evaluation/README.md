# Retrieval evaluation data

`out_of_domain_eval.json` is the gold dataset for comparing retrieval methods over the two processed NIST documents. It contains 100 questions across lexical, semantic, definition, factual, reasoning, and multi-chunk query types.

The file is a closed-corpus retrieval set: every question is answerable from the indexed NIST documents. The `out_of_domain` name should therefore mean that the NIST subject matter is outside the application's main domain, not that the answers come from documents absent from the retrieval corpus.

## Relevance policy

The `relevant_chunks_256` and `relevant_chunks_512` fields are unordered binary-relevance sets for their respective chunk collections.

A chunk is labeled relevant only when it directly supplies evidence needed to answer the query. A merely topical mention is not sufficient. Labels include overlapping chunks when either chunk independently retains meaningful answer evidence, as well as separate chunks when the expected answer requires multiple facts.

Each relevance set contains between one and five canonical chunk IDs. Five is a cap, not a target: a query has five labels only when five chunks are meaningfully relevant. When more than five passages contain similar evidence, the labels prioritize complete answer support, coverage of distinct answer facets, and the clearest source wording.

## Recommended retrieval metrics

Evaluate the 256-token and 512-token collections separately. For a retrieved ID list `R_k` and gold relevance set `G`:

- `Hit@k`: `1` when `R_k` contains at least one member of `G`, otherwise `0`.
- `Recall@k`: `|R_k intersect G| / |G|`.
- `Precision@k`: `|R_k intersect G| / k`.
- `MRR@k`: reciprocal rank of the first retrieved relevant chunk, or `0` when none is retrieved.
- `nDCG@k`: ranking quality with each gold chunk assigned binary relevance.

Report macro averages overall and by `query_type`. `Recall@5` is the primary coverage metric because the gold lists are capped at five labels and the retrieval implementations default to `top_k=5`. `MRR@5` complements it by measuring how early the first useful passage appears.

## Validation

Run the structural validator after changing the dataset or regenerating chunks:

```powershell
python tools/validate_evaluation_dataset.py
```

The validator checks query counts and categories, required fields, unique records, the one-to-five relevance constraint, canonical chunk existence, duplicate labels, and document consistency. Evidence quality still requires human review against the chunk text.
