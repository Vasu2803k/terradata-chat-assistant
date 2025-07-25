# Terradata_Assignment/backend/core/vectorstore.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Initialize embeddings only once
embeddings = HuggingFaceEmbeddings(
    model_name="all-mpnet-base-v2",
    cache_folder=str(PROJECT_ROOT / "hf_cache")
)

# Initialize Chroma vectorstore only once
vectorstore = Chroma(
    collection_name="text_embeddings",
    embedding_function=embeddings,
    persist_directory=str(PROJECT_ROOT / "vector_db")
)
