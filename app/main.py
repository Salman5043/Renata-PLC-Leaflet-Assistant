from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from .config import ABSTAIN, CHROMA_DIR
from .rag_graph import ask, health_check
from .schemas import AskBatchRequest, AskBatchResponse, AskRequest, AskResponse, Citation, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Renata PLC Grounded RAG Assistant", version="1.0.0", docs_url="/docs", redoc_url="/redoc")

def to_response(result: dict) -> AskResponse:
    citations = [Citation(**c) for c in result.get("citations", [])]
    return AskResponse(answer=result.get("answer", ABSTAIN), citations=citations, metrics=result.get("metrics"))

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health():
    data = health_check()
    return HealthResponse(**data)

@app.post("/ask", response_model=AskResponse)
@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        return to_response(ask(request.question))
    except Exception as exc:
        logger.exception("RAG request failed")
        raise HTTPException(status_code=500, detail="The RAG service could not process the request.") from exc

@app.post("/api/ask/batch", response_model=AskBatchResponse)
async def ask_batch(request: AskBatchRequest):
    results = []
    for question in request.questions:
        try:
            results.append(to_response(ask(question)))
        except Exception as exc:
            logger.exception("Batch item failed")
            results.append(AskResponse(answer=ABSTAIN, citations=[], error=str(exc)))
    return AskBatchResponse(results=results)

@app.get("/api/info")
async def info():
    return {"version": app.version, "vector_store": str(CHROMA_DIR), "products": health_check()["products"]}
