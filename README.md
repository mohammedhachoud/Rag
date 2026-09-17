# RAG Retrieval Benchmark

A benchmark project for preparing documents and evaluating retrieval strategies for retrieval-augmented generation (RAG) systems.

The current implementation provides the preprocessing pipeline: it extracts and cleans text from PDF files, preserves page-level metadata, and creates overlapping token-based chunks for later indexing and retrieval experiments.

## Current features

- Extract text from every PDF in `data/raw/` with `pypdf`.
- Clean null characters, line-break hyphenation, whitespace, and paragraph spacing.
- Preserve document metadata and text for each page.
- Create chunks using the `BAAI/bge-small-en-v1.5` tokenizer.
- Generate separate 256-token and 512-token chunk datasets.
- Apply a 32-token overlap between consecutive chunks.
- Assign stable chunk IDs and retain token offsets and source metadata.

## Project structure

```text
rag-retrieval-benchmark/
|-- data/
|   |-- raw/                    # Source PDF files
|   `-- processed/
|       |-- documents.json      # Extracted documents and pages
|       |-- chunks_256.json     # Generated after chunking
|       `-- chunks_512.json     # Generated after chunking
|-- src/
|   `-- preprocessing/
|       |-- extract_text.py
|       `-- chunk_documents.py
|-- requirements.txt
`-- README.md
```

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first chunking run downloads the tokenizer for `BAAI/bge-small-en-v1.5` from Hugging Face, so an internet connection is required unless it is already cached.

## Usage

### 1. Add source documents

Place PDF files in:

```text
data/raw/
```

### 2. Extract PDF text

```powershell
python src/preprocessing/extract_text.py
```

By default, the script reads `data/raw/*.pdf` and writes `data/processed/documents.json`.

Custom paths can also be supplied:

```powershell
python src/preprocessing/extract_text.py --input path/to/pdfs --output path/to/documents.json
```

Each extracted document contains:

- `document_id`
- `file_name`
- `page_count`
- complete cleaned `text`
- page-level text and page numbers

### 3. Create token chunks

```powershell
python src/preprocessing/chunk_documents.py
```

This reads `data/processed/documents.json` and produces:

- `data/processed/chunks_256.json`
- `data/processed/chunks_512.json`

Each chunk includes its ID, source document, file name, chunk configuration, sequence number, token range, token count, and text.

## Configuration

Chunking settings are currently defined in `src/preprocessing/chunk_documents.py`:

```python
MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZES = [256, 512]
CHUNK_OVERLAP = 32
```

## Notes and limitations

- PDFs must contain an extractable text layer. Scanned/image-only PDFs require OCR, which is not yet included.
- Generated chunk files are created only after running the chunking script.
- Retrieval indexing, search strategies, evaluation datasets, and benchmark metrics are planned future stages.

