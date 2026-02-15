"""
FastAPI backend for Jurisbusca React frontend.
Exposes chat and file upload endpoints, wiring into existing backend.py orchestration.
"""
import os
import json
import uuid
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import backend as be

app = FastAPI(title="Jurisbusca API", version="1.0.0")

@app.on_event("startup")
async def log_routes():
    print("\n🚀 Registered routes:")
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        path = getattr(route, 'path', getattr(route, 'path_regex', '?'))
        print(f"   {methods or 'MOUNT'} {path}")
    print()

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory conversation store (per session) ──────────────────────────────
conversations: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    model: str = "gemini"
    conversation_id: Optional[str] = None
    agent_prompt: Optional[str] = None
    ocr_engine: str = "gemini_flash"
    uploaded_text: Optional[str] = None  # pre-extracted text from uploaded file
    style_dossier: Optional[str] = None  # cloning prompt from style report


# ── Model mapping ───────────────────────────────────────────────────────────
MODEL_MAP = {
    "gemini": {"provider": "google", "model": "gemini-2.5-flash", "key_env": "GOOGLE_API_KEY"},
    "gpt": {"provider": "openai", "model": "gpt-4o", "key_env": "OPENAI_API_KEY"},
    "claude": {"provider": "anthropic", "model": "claude-sonnet-4-20250514", "key_env": "ANTHROPIC_API_KEY"},
    "deepseek": {"provider": "openai", "model": "deepseek-reasoner", "key_env": "DEEPSEEK_API_KEY"},
}


def _get_api_key(model_id: str) -> str:
    cfg = MODEL_MAP.get(model_id, MODEL_MAP["gemini"])
    key = os.getenv(cfg["key_env"], "")
    if not key:
        raise HTTPException(status_code=400, detail=f"API key não configurada: {cfg['key_env']}")
    return key


def _build_prompt(message: str, agent_prompt: Optional[str], uploaded_text: Optional[str]) -> str:
    """Combines agent system prompt + uploaded file text + user message into a single prompt."""
    parts = []
    if agent_prompt:
        parts.append(agent_prompt)
    if uploaded_text:
        parts.append(f"\n\n---\n📄 **DOCUMENTO ANEXADO (PEÇA PROCESSUAL):**\n\n{uploaded_text}\n---\n")
    parts.append(f"\n\n**SOLICITAÇÃO DO USUÁRIO:**\n{message}")
    return "\n".join(parts)


@app.get("/api/health")
async def health():
    return {"status": "ok", "routes": len(app.routes)}


@app.get("/api/debug-routes")
async def debug_routes():
    """Diagnostic: list all registered routes."""
    routes = []
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        path = getattr(route, 'path', str(getattr(route, 'path_regex', '?')))
        routes.append({"methods": list(methods) if methods else ["MOUNT"], "path": str(path)})
    return {"routes": routes}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Process a chat message and return LLM response."""
    try:
        api_key = _get_api_key(req.model)
        cfg = MODEL_MAP.get(req.model, MODEL_MAP["gemini"])

        # Build the LLM instance
        llm = be.get_llm(
            provider=cfg["provider"],
            model_name=cfg["model"],
            api_key=api_key,
            temperature=0.3,
        )

        # Conversation history
        conv_id = req.conversation_id or str(uuid.uuid4())
        if conv_id not in conversations:
            conversations[conv_id] = []

        # ── Build LangChain message list with full history ──────────────
        messages = []

        # System prompt: agent instructions + document context
        system_parts = []
        if req.agent_prompt:
            system_parts.append(req.agent_prompt)
        if req.uploaded_text:
            system_parts.append(
                f"\n\n---\n📄 **DOCUMENTO ANEXADO (PEÇA PROCESSUAL):**\n\n{req.uploaded_text}\n---"
            )
        if req.style_dossier:
            system_parts.append(
                f"\n\n---\n🧬 **SYSTEM PROMPT DE CLONAGEM ESTILÍSTICA (DOSSIÊ DO MAGISTRADO):**\n"
                f"⚠️ INSTRUÇÃO PRIMÁRIA: Replique rigorosamente o estilo descrito abaixo ao redigir qualquer decisão.\n\n"
                f"{req.style_dossier}\n---"
            )
        if system_parts:
            messages.append(SystemMessage(content="\n".join(system_parts)))

        # Append previous conversation turns
        for turn in conversations[conv_id]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        # Append current user message
        messages.append(HumanMessage(content=req.message))

        # Persist user turn in history
        conversations[conv_id].append({"role": "user", "content": req.message})

        # Call LLM with full conversation
        response = llm.invoke(messages)
        response_text = be.safe_content(response)

        # Persist assistant turn in history
        conversations[conv_id].append({"role": "assistant", "content": response_text})

        return {
            "conversation_id": conv_id,
            "response": response_text,
            "model": cfg["model"],
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    ocr_engine: str = Form("gemini_flash"),
):
    """Upload and extract text from a file (PDF/DOCX/TXT)."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY", "")

        # Save to temp file
        suffix = os.path.splitext(file.filename or "doc.pdf")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text using backend
        try:
            full_text, _ = be.process_uploaded_file(
                open(tmp_path, "rb"),
                file.filename or "documento",
                api_key=api_key,
                ocr_engine_choice=ocr_engine,
            )
        finally:
            os.unlink(tmp_path)

        if not full_text or not full_text.strip():
            raise HTTPException(status_code=422, detail="Não foi possível extrair texto do arquivo.")

        return {
            "filename": file.filename,
            "text": full_text,
            "char_count": len(full_text),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no upload: {str(e)}")


@app.post("/api/xray")
async def batch_xray(files: list[UploadFile] = File(...)):
    """
    Batch X-Ray: upload multiple process files, run MAP-REDUCE clustering.
    Returns clustered report JSON.
    """
    import io

    try:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY não configurada")

        # Convert UploadFiles into file-like objects with .name attribute
        file_objects = []
        for f in files:
            content = await f.read()
            buf = io.BytesIO(content)
            buf.name = f.filename or "documento.pdf"
            buf.seek(0)
            file_objects.append(buf)

        # Call the existing MAP-REDUCE pipeline
        report, text_cache = be.generate_batch_xray(file_objects, api_key)

        if "error" in report:
            raise HTTPException(status_code=422, detail=report.get("error", "Erro desconhecido"))

        return {
            "report": report,
            "file_count": len(file_objects),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no Raio-X: {str(e)}")


@app.post("/api/style-report")
async def style_report(files: list[UploadFile] = File(...)):
    """
    Generate a Style Dossier (Dossiê de Identidade Decisional) from template files.
    Uses the existing generate_style_dossier pipeline in backend.py.
    """
    import io

    try:
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY não configurada")

        # Convert UploadFiles into file-like objects with .name attribute
        file_objects = []
        for f in files:
            content = await f.read()
            buf = io.BytesIO(content)
            buf.name = f.filename or "template.pdf"
            buf.seek(0)
            file_objects.append(buf)

        # Call the existing style dossier pipeline
        result = be.generate_style_dossier(file_objects, api_key)

        if not result or result.get("error"):
            error_detail = (result or {}).get("error", "Não foi possível gerar o dossiê de estilo.")
            raise HTTPException(status_code=422, detail=error_detail)

        return {
            "dossier": result.get("dossier", ""),
            "glossary": result.get("glossary", ""),
            "cloning_prompt": result.get("cloning_prompt", ""),
            "full_response": result.get("full_response", ""),
            "file_count": len(file_objects),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao gerar relatório de estilo: {str(e)}")


# ── Serve React frontend (production) ────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"
if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str):
        """Catch-all: serve static files or fallback to index.html (SPA routing).
        IMPORTANT: never serve HTML for /api/ paths."""
        # Never intercept API routes
        if full_path.startswith("api/") or full_path.startswith("api"):
            raise HTTPException(status_code=404, detail=f"API endpoint not found: /{full_path}")
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
