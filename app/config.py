from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Groq
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-8b-8192"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "erp_docs"

    # RAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    DOCS_DIR: str = "./data/sample_docs"

    class Config:
        env_file = ".env"

settings = Settings()
