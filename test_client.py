from src.generation.prompt_builder import build_messages
from src.generation.ollama_client import OllamaGenerator

chunks = [
    {
        "chunk_id": "ai_rmf_512_0041",
        "document": "NIST AI RMF",
        "page": 20,
        "text": (
            "The Framework is organized around four "
            "functions: GOVERN, MAP, MEASURE and MANAGE."
        ),
    }
]

messages = build_messages(
    question="What are the four AI RMF core functions?",
    chunks=chunks,
)

generator = OllamaGenerator()

result = generator.generate(messages)

messages2 = build_messages(
    question="How do I configure PostgreSQL replication?",
    chunks=chunks,
)

result2 = generator.generate(messages2)

print(result2["answer"])
print(result2)