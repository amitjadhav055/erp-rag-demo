# ERP RAG Chatbot Demo

A production-style **Retrieval-Augmented Generation (RAG)** pipeline that answers natural language questions against enterprise ERP documents — invoices, purchase orders, error guides, and financial policies.

Built with **LangChain + Groq (Llama 3) + ChromaDB + FastAPI**.

> This project mirrors the architecture I built and deployed in production at Prodware Solutions, where the RAG system serves ~1,000 users across JD Edwards ERP environments. That system uses OpenAI and Oracle 23AI Vector DB; this demo uses Groq and ChromaDB to keep it fully open and free to run.

---

## Architecture

```
User Question (HTTP POST /query)
        │
        ▼
  FastAPI Backend
        │
        ▼
  LangChain Orchestration
        │
   ┌────┴────┐
   │         │
   ▼         ▼
ChromaDB   Groq API
(retrieval) (Llama 3)
   │         │
   └────┬────┘
        │
        ▼
 Answer + Sources (JSON)
```

**Retrieval flow:**
1. User question is embedded using `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free)
2. ChromaDB performs similarity search and returns the top-k relevant document chunks
3. Retrieved chunks are injected into the prompt as context
4. Groq's Llama 3 generates a grounded answer using only the provided context
5. Response includes the answer, source document names, and number of chunks retrieved

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| LLM | Llama 3 via Groq | Free, fast, OpenAI-compatible API |
| LLM Framework | LangChain | Chain orchestration, retriever abstraction |
| Vector Store | ChromaDB | Local persistent vector DB, no setup needed |
| Embeddings | sentence-transformers (local) | No API key required, runs on CPU |
| API | FastAPI | Same stack used in production |
| Document Loading | LangChain DirectoryLoader | Handles .txt, extensible to PDF |

---

## Project Structure

```
erp-rag-demo/
├── app/
│   ├── main.py          # FastAPI app — endpoints: /query, /ingest, /health
│   ├── rag_pipeline.py  # Core RAG logic — ingest, embed, retrieve, generate
│   ├── ingest.py        # Standalone ingestion script
│   └── config.py        # Settings via pydantic-settings + .env
├── data/
│   └── sample_docs/     # ERP knowledge base documents (.txt)
│       ├── invoice_processing_policy.txt
│       ├── purchase_order_guide.txt
│       ├── voucher_payment_guide.txt
│       └── erp_error_resolution_guide.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/amitjadhav055/erp-rag-demo.git
cd erp-rag-demo

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key (free at [console.groq.com](https://console.groq.com)):

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama3-8b-8192
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

Documents are ingested automatically on first startup. The API will be live at `http://localhost:8000`.

### 4. Query the chatbot

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens when an invoice fails three-way matching?"}'
```

Or open the interactive docs at `http://localhost:8000/docs` and use the built-in Swagger UI.

---

## Example Queries

Try these against the sample knowledge base:

```
"What are the approval levels for purchase orders?"
"How do I resolve error ERR-003 on an invoice?"
"What is the payment process for amounts above ₹2,00,000?"
"What happens if a vendor is on hold when I raise a PO?"
"How long are voucher records retained in the system?"
"What causes a duplicate invoice error and how do I fix it?"
```

---

## Adding Your Own Documents

Drop `.txt` files into `data/sample_docs/` and call the ingest endpoint:

```bash
curl -X POST http://localhost:8000/ingest
```

Or run the standalone script:

```bash
python -m app.ingest
```

The pipeline also supports PDF ingestion — extend `DirectoryLoader` in `rag_pipeline.py` with `PyMuPDFLoader` for PDF support.

---

## Key Design Decisions

**Semantic chunking over fixed-size splitting**
`RecursiveCharacterTextSplitter` with `separators=["\n\n", "\n", ".", " "]` respects natural document boundaries (paragraphs, sentences) rather than cutting at arbitrary character counts. This significantly improves retrieval quality for multi-step procedural documents like error resolution guides.

**Low temperature (0.1) for factual domains**
ERP questions require precise, factual answers. High temperature introduces hallucination risk. 0.1 keeps the model grounded in the retrieved context.

**Local embeddings, no API dependency**
`sentence-transformers/all-MiniLM-L6-v2` runs on CPU with no API key. This eliminates a dependency and keeps ingestion costs at zero regardless of document volume.

**Source attribution in every response**
Every answer includes the source document names. In enterprise contexts, auditability matters — users need to know where the answer came from.

---

## Production Differences

In the production system at Prodware Solutions:

| This demo | Production system |
|---|---|
| Groq (Llama 3) | OpenAI GPT-4o |
| ChromaDB (local) | Oracle 23AI Vector DB |
| .txt files | Live JDE API connector + internal docs |
| Single pipeline | Dual-bot system (data query + error resolution) |
| localhost | Oracle Cloud Infrastructure (OCI) |

The RAG logic, chunking strategy, retrieval pattern, and prompt structure are the same.

---

## Author

**Amit Jadhav** — AI Engineer  
[LinkedIn](https://linkedin.com/in/amitjadhav01) · [Portfolio](https://amitjadhav01-portfolio.vercel.app) · [GitHub](https://github.com/amitjadhav055)
