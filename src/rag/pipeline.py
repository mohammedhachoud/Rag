from typing import Any

from src.generation.ollama_client import OllamaGenerator
from src.generation.prompt_builder import build_messages
from src.retrieval.dense import search_dense

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class RAGPipeline:
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        qdrant_client: QdrantClient,
        collection_name: str,
        generator: OllamaGenerator | None = None,
        top_k: int = 5,
    ) -> None:
        self.embedding_model = embedding_model
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.generator = generator or OllamaGenerator()
        self.top_k = top_k

    def answer(
        self,
        question: str,
    ) -> dict[str, Any]:
        # 1. Retrieve the relevant chunks
        chunks = search_dense(
            query=question,
            model=self.embedding_model,
            client=self.qdrant_client,
            collection_name=self.collection_name,
            top_k=self.top_k,
        )

        # 2. Construct the LLM messages
        messages = build_messages(
            question=question,
            chunks=chunks,
        )

        # 3. Generate the answer
        generation = self.generator.generate(messages)

        # 4. Keep retrieval information for inspection
        return {
            "question": question,
            "answer": generation["answer"],
            "model": generation["model"],
            "retrieved_chunks": chunks,
            "generation_stats": {
                "prompt_tokens": generation[
                    "prompt_tokens"
                ],
                "generated_tokens": generation[
                    "generated_tokens"
                ],
                "total_duration_seconds": generation[
                    "total_duration_seconds"
                ],
                "load_duration_seconds": generation[
                    "load_duration_seconds"
                ],
                "prompt_evaluation_seconds": generation[
                    "prompt_evaluation_seconds"
                ],
                "generation_duration_seconds": generation[
                    "generation_duration_seconds"
                ],
                "generation_tokens_per_second": generation[
                    "generation_tokens_per_second"
                ],
            },
        }


if __name__ == "__main__":
    import json
    import argparse
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        type=str,
        help="Question to answer",
    )
    args = parser.parse_args()

    embedding_model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

    qdrant_client = QdrantClient(
        url="http://localhost:6333"
    )

    generator = OllamaGenerator(
        model="granite4.1:8b-q4_K_M",
        temperature=0.0,
        max_output_tokens=512,
        context_length=8192,
    )

    pipeline = RAGPipeline(
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        collection_name="rag_dense_512",
        generator=generator,
        top_k=5,
    )

    result = pipeline.answer(question=args.question)


    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )