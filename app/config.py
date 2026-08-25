"""
Application configuration — loads environment variables and defines paths.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"
DATA_DIR = PROJECT_ROOT / "data"
ORDERS_FILE = DATA_DIR / "orders.json"
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
CHROMA_COLLECTION_NAME = "aster_row_kb"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

if NVIDIA_API_KEY:
    OPENROUTER_BASE_URL = "https://integrate.api.nvidia.com/v1"
    OPENROUTER_API_KEY = NVIDIA_API_KEY   # agent.py uses OPENROUTER_API_KEY
    LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
else:
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
LLM_TEMPERATURE = 0.0

RETRIEVAL_TOP_K = 8  
