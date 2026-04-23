"""Configuration loaded from environment, with sensible defaults.

Supports two providers: 'anthropic' and 'openai'. The provider is selected via
ORCHESTRATOR_PROVIDER (or --provider on the CLI). The relevant API key and
default model are looked up per-provider.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}


@dataclass
class Config:
    provider: str
    api_key: str
    model: str
    max_iterations: int
    auto_apply_threshold: float
    repo_root: Path

    @classmethod
    def load(
        cls,
        repo_root: Path | None = None,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> "Config":
        provider = (
            provider_override
            or os.environ.get("ORCHESTRATOR_PROVIDER", "anthropic")
        ).lower()

        if provider == "anthropic":
            key_env = "ANTHROPIC_API_KEY"
        elif provider == "openai":
            key_env = "OPENAI_API_KEY"
        else:
            raise RuntimeError(
                f"Unknown provider {provider!r}. Set ORCHESTRATOR_PROVIDER or --provider to 'anthropic' or 'openai'."
            )

        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"{key_env} is not set. Copy .env.example to .env and fill it in, "
                f"or choose a different provider with --provider."
            )

        model = model_override or os.environ.get("ORCHESTRATOR_MODEL") or DEFAULT_MODELS[provider]

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            max_iterations=int(os.environ.get("ORCHESTRATOR_MAX_ITERATIONS", "30")),
            auto_apply_threshold=float(os.environ.get("ORCHESTRATOR_AUTO_APPLY_THRESHOLD", "0.85")),
            repo_root=(repo_root or Path.cwd()).resolve(),
        )
