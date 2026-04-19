"""
Standalone ingestion script.
Run this once to load documents into ChromaDB before starting the API.

Usage:
    python -m app.ingest
"""

from app.rag_pipeline import RAGPipeline

def main():
    print("Starting document ingestion...")
    rag = RAGPipeline()
    count = rag.ingest_documents()
    print(f"Done. {count} chunks stored in ChromaDB.")

if __name__ == "__main__":
    main()
