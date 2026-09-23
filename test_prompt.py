from src.generation.prompt_builder import build_messages

chunks = [
    {
        "chunk_id": "nist-ai-100-1-pdf_512_0001",
        "document_id": "NIST.AI.100-1.pdf",
        "file_name": "NIST.AI.100-1.pdf.pdf",
        "chunk_size": 512,
        "chunk_overlap": 32,
        "chunk_number": 1,
        "start_token": 0,
        "end_token": 510,
        "token_count": 510,
        "text": "The NIST AI Risk Management Framework defines four core functions: GOVERN, MAP, MEASURE, and MANAGE."
    }
]

messages = build_messages(
    "What are the AI RMF core functions?",
    chunks,
)

for message in messages:
    print(message["role"].upper())
    print(message["content"])
    print()