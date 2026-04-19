from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.config import settings
import os


ERP_SYSTEM_PROMPT = """You are an expert ERP assistant with deep knowledge of enterprise 
resource planning systems, procurement processes, invoice management, and financial workflows.

Use ONLY the context provided below to answer the question. If the answer is not in the 
context, say "I don't have enough information in the knowledge base to answer that."

Be specific, accurate, and concise. When referencing processes or policies, cite the 
source document name.

Context:
{context}
"""


class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        self.vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )

        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.1,
            max_tokens=1024,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ERP_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "]
        )

    def _format_docs(self, docs) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )

    def ingest_documents(self) -> int:
        loader = DirectoryLoader(
            settings.DOCS_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()
        chunks = self.splitter.split_documents(docs)

        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(
                chunk.metadata.get("source", "unknown")
            )

        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=settings.CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR
        )
        self.vectorstore.add_documents(chunks)

        print(f"[RAG] Ingested {len(chunks)} chunks from {len(docs)} documents")
        return len(chunks)

    def ingest_if_empty(self):
        count = self.vectorstore._collection.count()
        if count == 0:
            print("[RAG] Vector store empty — ingesting documents...")
            self.ingest_documents()
        else:
            print(f"[RAG] Vector store has {count} chunks — skipping ingestion")

    def query(self, question: str, top_k: int = 4) -> dict:
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )

        retrieved_docs = retriever.invoke(question)
        context = self._format_docs(retrieved_docs)

        chain = (
            {
                "context": lambda _: context,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        sources = list(set(
            doc.metadata.get("source", "unknown")
            for doc in retrieved_docs
        ))

        return {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(retrieved_docs)
        }
