"""AI layer: manages Foundry Local models and exposes embedding/chat helpers.

Uses the Foundry Local Python SDK (v1.2+): a singleton FoundryLocalManager
gives access to the model catalog; each model object can be downloaded,
loaded into memory, and queried through OpenAI-compatible clients.
"""

import sys

from foundry_local_sdk import Configuration, FoundryLocalManager

import config

_embed_client = None
_chat_client = None


def _progress(percent: float) -> None:
    print(f"\r  downloading... {percent:5.1f}%", end="", flush=True)
    if percent >= 100:
        print()


def _get_manager() -> FoundryLocalManager:
    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="local-rag-assistant"))
        manager = FoundryLocalManager.instance
        # Execution providers (CPU/GPU backends) must be registered once
        # before any model can run.
        eps = manager.discover_eps()
        if not any(ep.is_registered for ep in eps):
            print("First run: downloading execution providers "
                  "(one-time setup, may take a few minutes)...")
            result = manager.download_and_register_eps(
                progress_callback=lambda name, pct: print(
                    f"\r  {name}: {pct:5.1f}%", end="", flush=True)
            )
            print()
            if not result.success and not result.registered_eps:
                sys.exit(f"Could not register execution providers: {result.status}")
    return FoundryLocalManager.instance


def _prepare_model(alias: str):
    manager = _get_manager()
    model = manager.catalog.get_model(alias)
    if model is None:
        sys.exit(f"Model alias '{alias}' not found in the Foundry Local catalog.")
    if not model.is_cached:
        print(f"Downloading model '{alias}' (one-time)...")
        model.download(progress_callback=_progress)
    if not model.is_loaded:
        model.load()
    return model


def init_embedding_model() -> None:
    """Download (if needed) and load the local embedding model."""
    global _embed_client
    if _embed_client is None:
        model = _prepare_model(config.EMBEDDING_MODEL_ALIAS)
        _embed_client = model.get_embedding_client()


def init_chat_model() -> None:
    """Download (if needed) and load the local chat model."""
    global _chat_client
    if _chat_client is None:
        model = _prepare_model(config.CHAT_MODEL_ALIAS)
        _chat_client = model.get_chat_client()
        _chat_client.settings.temperature = 0.2
        _chat_client.settings.max_tokens = 600


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input text."""
    init_embedding_model()
    response = _embed_client.generate_embeddings(texts)
    # The API may return items out of order; sort by index to be safe.
    items = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in items]


def chat(system_prompt: str, user_prompt: str) -> str:
    """One-turn chat completion against the local model."""
    init_chat_model()
    completion = _chat_client.complete_chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return completion.choices[0].message.content.strip()
