import os
from typing import Any

from ollama import Client


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
#DEFAULT_MODEL = "granite4.1:8b-q4_K_M"
DEFAULT_MODEL = "granite4.1:3b"

def nanoseconds_to_seconds(
    value: int | None,
) -> float | None:
    if value is None:
        return None

    return round(value / 1_000_000_000, 3)

class OllamaGenerator:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        context_length: int = 4096,
    ) -> None:
        self.model = (
            model
            or os.getenv("OLLAMA_MODEL")
            or DEFAULT_MODEL
        )

        self.host = (
            host
            or os.getenv("OLLAMA_HOST")
            or DEFAULT_OLLAMA_HOST
        )

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.context_length = context_length

        self.client = Client(host=self.host)

    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Send messages to the LLM and return the response.

        This is a blocking call.
        """
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            keep_alive="30m",
            options={
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
                "num_ctx": self.context_length,
            },
        )

        answer = response.message.content.strip()

        generation_seconds = nanoseconds_to_seconds(
            response.eval_duration
        )

        generated_tokens = response.eval_count or 0

        tokens_per_second = None

        if generation_seconds and generation_seconds > 0:
            tokens_per_second = round(
                generated_tokens / generation_seconds,
                2,
            )

        return {
            "answer": answer,
            "model": response.model,
            "prompt_tokens": (
                response.prompt_eval_count or 0
            ),
            "generated_tokens": generated_tokens,
            "total_duration_seconds": (
                nanoseconds_to_seconds(
                    response.total_duration
                )
            ),
            "load_duration_seconds": (
                nanoseconds_to_seconds(
                    response.load_duration
                )
            ),
            "prompt_evaluation_seconds": (
                nanoseconds_to_seconds(
                    response.prompt_eval_duration
                )
            ),
            "generation_duration_seconds": (
                generation_seconds
            ),
            "generation_tokens_per_second": (
                tokens_per_second
            ),
        }

    def unload(self) -> None:
        """Release this model from Ollama's CPU/GPU memory."""
        self.client.generate(
            model=self.model,
            prompt="",
            keep_alive=0,
        )

    def resource_usage(self) -> dict[str, float | None]:
        """Return Ollama's current allocation for this model in MiB."""
        for process in self.client.ps().models:
            if process.model == self.model or process.name == self.model:
                total_bytes = int(process.size or 0)
                vram_bytes = int(process.size_vram or 0)
                mib = 1024 * 1024
                return {
                    "total_memory_mb": round(total_bytes / mib, 2),
                    "vram_mb": round(vram_bytes / mib, 2),
                    "cpu_memory_mb": round(max(total_bytes - vram_bytes, 0) / mib, 2),
                }
        return {
            "total_memory_mb": None,
            "vram_mb": None,
            "cpu_memory_mb": None,
        }
