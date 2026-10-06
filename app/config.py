from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / os.getenv("DOCS_DIR", "docs")
CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "renata_leaflets_v4")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieve more candidates than we finally expose. This is important for
# cross-document questions where several products share the same section.
TOP_K = int(os.getenv("TOP_K", "12"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "30"))
MAX_EVIDENCE_DOCUMENTS = int(os.getenv("MAX_EVIDENCE_DOCUMENTS", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))

DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.70"))
LEXICAL_WEIGHT = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.20"))
SECTION_WEIGHT = float(os.getenv("HYBRID_SECTION_WEIGHT", "0.10"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

ABSTAIN = "I can't answer that from the provided Renata leaflets."
