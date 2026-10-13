"""OpenRouter client (OpenAI-compatible API)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-5.5")

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost: float | None  # USD as reported by OpenRouter


def complete(messages: list[dict], model: str = MODEL) -> LLMResponse:
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        extra_body={"usage": {"include": True}},  # asks OpenRouter to report the cost
    )
    usage = response.usage
    return LLMResponse(
        text=response.choices[0].message.content or "",
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost=(usage.model_extra or {}).get("cost"),
    )
