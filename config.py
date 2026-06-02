from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_index"

GRAPHRAG_ROOT = PROJECT_ROOT / "GraphRAG" / "tourist_graphrag"
GRAPHRAG_OUTPUT_DIR = GRAPHRAG_ROOT / "output"

VLM_RESULT_MARKDOWN_DIR = PROJECT_ROOT / "vlm" / "result_markdown"
PURE_OCR_DIR = PROJECT_ROOT / "pure_ocr"
PURE_OCR_IMAGE_DIR = PURE_OCR_DIR / "image_path"
PURE_OCR_RESULT_MARKDOWN_DIR = PURE_OCR_DIR / "result_markdown"


def load_project_env() -> None:
    """Load project-level environment variables once from .env."""
    load_dotenv(PROJECT_ROOT / ".env")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {name}. "
            f"Create {PROJECT_ROOT / '.env'} from .env.example and fill it in."
        )
    return value


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path



def get_env(name: str, default=None):
    load_project_env()
    return os.getenv(name, default)


def require_env(name: str) -> str:
    load_project_env()
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
