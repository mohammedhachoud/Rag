"""Central, validated application configuration loaded from the project .env."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding process variables."""
    if not path.is_file():
        return
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env entry on line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"Empty .env key on line {line_number}")
        os.environ.setdefault(key, value)


_load_env(PROJECT_ROOT / ".env")


def _integer(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _floating(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _chunk_sizes() -> tuple[int, ...]:
    raw = os.getenv("CHUNK_SIZES", "256,512")
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("CHUNK_SIZES must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("CHUNK_SIZES must contain positive integers")
    return values


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    chunk_sizes: tuple[int, ...] = _chunk_sizes()
    chunk_overlap: int = _integer("CHUNK_OVERLAP", 32, 0)
    embedding_batch_size: int = _integer("EMBEDDING_BATCH_SIZE", 32)
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection_256: str = os.getenv("QDRANT_COLLECTION_256", "rag_dense_256")
    qdrant_collection_512: str = os.getenv("QDRANT_COLLECTION_512", "rag_dense_512")
    qdrant_upsert_batch_size: int = _integer("QDRANT_UPSERT_BATCH_SIZE", 100)
    retrieval_top_k: int = _integer("RETRIEVAL_TOP_K", 5)
    generation_top_k: int = _integer("GENERATION_TOP_K", 3)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "granite4.1:3b")
    ollama_model_8b: str = os.getenv("OLLAMA_MODEL_8B", "granite4.1:8b-q4_K_M")
    ollama_temperature: float = _floating("OLLAMA_TEMPERATURE", 0.0)
    ollama_max_output_tokens: int = _integer("OLLAMA_MAX_OUTPUT_TOKENS", 512)
    ollama_context_length: int = _integer("OLLAMA_CONTEXT_LENGTH", 8192)
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "30m")
    correctness_threshold: float = _floating("CORRECTNESS_THRESHOLD", 0.75)

    @property
    def collections(self) -> dict[int, str]:
        configured = {256: self.qdrant_collection_256, 512: self.qdrant_collection_512}
        missing = set(self.chunk_sizes) - configured.keys()
        if missing:
            raise ValueError(f"No Qdrant collection configured for: {sorted(missing)}")
        return {size: configured[size] for size in self.chunk_sizes}

    def qdrant_kwargs(self) -> dict[str, str]:
        values = {"url": self.qdrant_url}
        if self.qdrant_api_key:
            values["api_key"] = self.qdrant_api_key
        return values


settings = Settings()
