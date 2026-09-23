import os
from typing import Any

from ollama import Client


DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "granite4.1:8b-q4_K_M"

class OllamaGenerator:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        context_length: int = 8192,
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
            options={
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
                "num_ctx": self.context_length,
            },
        )

        answer = response.message.content.strip()

        return {
            "answer": answer,
            "model": response.model,
            "prompt_tokens": response.prompt_eval_count,
            "generated_tokens": response.eval_count,
            "total_duration_ns": response.total_duration,
            "load_duration_ns": response.load_duration,
            "generation_duration_ns": response.eval_duration,
        }