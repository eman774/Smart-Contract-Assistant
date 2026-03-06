"""
server.py
=========
FastAPI + LangServe backend for the Smart Contract Assistant.

- FastAPI: custom endpoints (/ingest, /chat, /summarize, /evaluate)
- LangServe: exposes the RAG chain at /qa/invoke (standard LangChain interface)

Run with: python server.py
API docs: http://localhost:8000/docs
LangServe playground: http://localhost:8000/qa/playground
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import tempfile
import os

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.ingestion import ingest
from src.retrieval import answer, summarize, reload_vectorstore, get_vectorstore, get_llm
from src.evaluation import evaluate_retrieval_only
from src.config import TOP_K

app = FastAPI(
    title="Smart Contract Assistant API",
    description="RAG pipeline for contract Q&A — FastAPI + LangServe",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

last_file = {"name": None}


# ── LangServe Chain Setup ────────────────────────────────────────────────────

def build_rag_chain():
    """
    Builds a LangChain LCEL chain and exposes it via LangServe.
    Chain flow: question → retriever → prompt → LLM → answer
    """
    SYSTEM = """You are a professional contract analysis assistant.
Answer questions strictly based on the contract excerpts below.
If the answer is not in the contract, say: "This information was not found in the contract."
Always cite where you found the information.

Contract excerpts:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    def retrieve_context(inputs: dict) -> dict:
        vs       = get_vectorstore()
        docs     = vs.similarity_search(inputs["question"], k=TOP_K)
        context  = "\n\n---\n\n".join([d.page_content for d in docs])
        return {"context": context, "question": inputs["question"]}

    chain = (
        RunnableLambda(retrieve_context)
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain


# Register LangServe route
try:
    from langserve import add_routes
    rag_chain = build_rag_chain()
    add_routes(
        app,
        rag_chain,
        path="/qa",
        input_type=dict,
    )
    print("✅ LangServe route registered at /qa/invoke and /qa/playground")
except Exception as e:
    print(f"⚠️  LangServe not available: {e}")


# ── Request/Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    history: list = []

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

class IngestResponse(BaseModel):
    file_name: str
    characters: int
    chunks: int
    message: str


# ── FastAPI Endpoints ────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Smart Contract Assistant API",
        "endpoints": {
            "docs":       "/docs",
            "ingest":     "POST /ingest",
            "chat":       "POST /chat",
            "summarize":  "POST /summarize",
            "evaluate":   "GET /evaluate",
            "langserve":  "POST /qa/invoke  |  GET /qa/playground",
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy", "document": last_file["name"] or "none"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and process a PDF or DOCX contract."""
    if not file.filename.endswith((".pdf", ".docx", ".doc")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        import src.retrieval as _ret
        import gc
        _ret._vectorstore = None
        gc.collect()

        stats = ingest(tmp_path)
        last_file["name"] = stats["file_name"]
        reload_vectorstore()

        return IngestResponse(
            file_name=stats["file_name"],
            characters=stats["characters"],
            chunks=stats["chunks"],
            message="Document ingested successfully.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question about the uploaded contract."""
    if not last_file["name"]:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Call /ingest first.")
    try:
        response, source_docs = answer(request.question, request.history)
        sources = list({d.metadata.get("source", "?") for d in source_docs})
        return ChatResponse(answer=response, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
def summarize_document():
    """Generate a structured summary of the uploaded contract."""
    if not last_file["name"]:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Call /ingest first.")
    try:
        summary = summarize(last_file["name"])
        return {"summary": summary, "file": last_file["name"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evaluate")
def evaluate():
    """Run retrieval evaluation metrics on the current document."""
    if not last_file["name"]:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Call /ingest first.")
    try:
        report = evaluate_retrieval_only()
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
