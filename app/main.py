from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline
from app.config import settings
import uvicorn


# lifespan replaces deprecated @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ingest documents on startup if vector store is empty."""
    rag_instance.ingest_if_empty()
    yield

rag_instance = RAGPipeline()

app = FastAPI(
    title="ERP RAG Demo API",
    description="RAG pipeline demo — mirrors the architecture used in production JD Edwards ERP chatbot",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_retrieved: int


@app.get("/")
def root():
    return {
        "message": "ERP RAG Demo API is running",
        "docs": "/docs",
        "endpoints": ["/query", "/ingest", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": settings.GROQ_MODEL,
        "vectorstore": "ChromaDB"
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Ask a natural language question against the ingested ERP documents.
    Returns an answer with source references.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = rag_instance.query(request.question, top_k=request.top_k)
    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks_retrieved=result["chunks_retrieved"]
    )

@app.post("/ingest")
def ingest():
    """Re-ingest all documents from data/sample_docs/ into the vector store."""
    count = rag_instance.ingest_documents()
    return {"message": f"Ingested {count} document chunks into ChromaDB"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
