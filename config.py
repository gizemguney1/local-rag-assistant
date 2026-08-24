"""Central configuration for the local RAG assistant."""

from pathlib import Path

PROJECT_DIR = Path(__file__).parent
DOCUMENTS_DIR = PROJECT_DIR / "documents"
DB_PATH = PROJECT_DIR / "rag.db"

# Foundry Local model aliases. Foundry resolves an alias to the best
# variant for the local hardware (CPU/GPU/NPU).
CHAT_MODEL_ALIAS = "phi-3.5-mini"
EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

# Retrieval settings
TOP_K = 3                 # number of chunks handed to the LLM as context
MIN_SIMILARITY = 0.30     # below this the chunk is considered irrelevant

# Chunking settings
MAX_CHUNK_CHARS = 1200    # merge paragraphs up to roughly this size

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context. Rules:\n"
    "1. Base your answer strictly on the context below. Do not use outside "
    "knowledge.\n"
    "2. If the context does not contain the answer, reply exactly: "
    "\"I don't have that information in my documents.\"\n"
    "3. Mention which source document your answer comes from, e.g. "
    "\"(source: <name>)\".\n"
    "4. Keep answers concise and polite."
)
