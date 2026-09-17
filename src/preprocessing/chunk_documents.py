import json
import re
from pathlib import Path

# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZES = [256, 512]
CHUNK_OVERLAP = 32


def create_document_slug(document_id: str) -> str:
    slug = document_id.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    special_token_count = tokenizer.num_special_tokens_to_add(
        pair=False
    )

    content_limit = chunk_size - special_token_count

    if overlap >= content_limit:
        raise ValueError(
            "The overlap must be smaller than the chunk content limit."
        )

    step = content_limit - overlap
    chunks = []

    for start_token in range(0, len(token_ids), step):
        end_token = min(
            start_token + content_limit,
            len(token_ids),
        )

        chunk_token_ids = token_ids[start_token:end_token]

        chunk_text_value = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        if chunk_text_value:
            chunks.append(
                {
                    "text": chunk_text_value,
                    "start_token": start_token,
                    "end_token": end_token,
                    "token_count": len(chunk_token_ids),
                }
            )

        if end_token == len(token_ids):
            break

    return chunks


def chunk_documents(
    documents: list[dict],
    tokenizer,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    all_chunks = []

    for document in documents:
        document_id = document["document_id"]
        document_slug = create_document_slug(document_id)

        document_chunks = chunk_text(
            text=document["text"],
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_number, chunk in enumerate(
            document_chunks,
            start=1,
        ):
            chunk_id = (
                f"{document_slug}_"
                f"{chunk_size}_"
                f"{chunk_number:04d}"
            )

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "file_name": document["file_name"],
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "chunk_number": chunk_number,
                    "start_token": chunk["start_token"],
                    "end_token": chunk["end_token"],
                    "token_count": chunk["token_count"],
                    "text": chunk["text"],
                }
            )

    return all_chunks


def save_chunks(
    chunks: list[dict],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            chunks,
            output_file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "documents.json"
    )

    with input_path.open("r", encoding="utf-8") as input_file:
        documents = json.load(input_file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for chunk_size in CHUNK_SIZES:
        chunks = chunk_documents(
            documents=documents,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=CHUNK_OVERLAP,
        )

        output_path = (
            project_root
            / "data"
            / "processed"
            / f"chunks_{chunk_size}.json"
        )

        save_chunks(chunks, output_path)

        print(
            f"Created {len(chunks)} chunks "
            f"for the {chunk_size}-token configuration."
        )
        print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()