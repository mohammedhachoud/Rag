import argparse
import json
import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:

    # Remove null characters.
    text = text.replace("\x00", "")

    # Join words separated by line-break hyphenation.
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Replace line breaks inside paragraphs with spaces.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize paragraph spacing.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf(pdf_path: Path) -> dict:

    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = clean_text(raw_text)

        pages.append(
            {
                "page_number": page_number,
                "text": cleaned_text,
            }
        )

    document_text = "\n\n".join(
        page["text"] for page in pages if page["text"]
    )

    return {
        "document_id": pdf_path.stem,
        "file_name": pdf_path.name,
        "page_count": len(reader.pages),
        "text": document_text,
        "pages": pages,
    }


def extract_all_pdfs(raw_directory: Path) -> list[dict]:

    pdf_files = sorted(raw_directory.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files were found in: {raw_directory.resolve()}"
        )

    documents = []

    for pdf_path in pdf_files:
        print(f"Extracting: {pdf_path.name}")

        try:
            document = extract_pdf(pdf_path)
            documents.append(document)

            character_count = len(document["text"])

            print(
                f"  Extracted {document['page_count']} pages "
                f"and {character_count:,} characters."
            )

            if character_count == 0:
                print(
                    "  Warning: no text was extracted. "
                    "This PDF may require OCR."
                )

        except Exception as error:
            print(f"  Failed to extract {pdf_path.name}: {error}")

    return documents


def save_documents(documents: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            documents,
            output_file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Extract text from PDF documents."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "raw",
        help="Directory containing the PDF files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "processed" / "documents.json",
        help="Path of the generated JSON file.",
    )

    args = parser.parse_args()

    documents = extract_all_pdfs(args.input)
    save_documents(documents, args.output)

    print()
    print(f"Extracted {len(documents)} document(s).")
    print(f"Output saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()