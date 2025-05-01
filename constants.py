from pathlib import Path

# File paths
ROOT = Path(__file__).parent.resolve()
DOC_PATH = Path(f"{ROOT}/data-scraper/data")
INDEX_CACHE = ROOT / "semantic_faiss_index"

# Chunking constants
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40
SEMANTIC_CHUNKING_BLOCK_SIZE = 256

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_COMPLETION_URL = "http://localhost:11434/api/generate"
EMBEDING_MODEL_NAME = "nomic-embed-text"
CHUNKING_MODEL_NAME = "phi4-mini"
CHUNKER = "semantic"
