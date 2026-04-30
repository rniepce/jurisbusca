"""
FastAPI backend for Jurisbusca React frontend.
Exposes chat and file upload endpoints, wiring into existing backend.py orchestration.
"""
import os
import json
import uuid
import hashlib
import logging
import tempfile
import traceback
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt  # PyJWT
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import backend as be
import history_db

# ── Structured Logging (COARF §XI — logs as event stream to stdout) ───────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("jurisbusca")

# ── Local SLM Engine (Apple Silicon only) ─────────────────────────────────────
try:
    import slm_engine
    HAS_SLM = slm_engine.is_available()
    if HAS_SLM:
        logger.info("SLM Engine disponível (MLX). Opção 'SLMs Locais' ativa.")
    else:
        logger.warning("MLX não instalado. SLM Engine desabilitado.")
except ImportError:
    HAS_SLM = False
    slm_engine = None
    logger.warning("slm_engine.py não encontrado. SLM Engine desabilitado.")

# ── SLM Orchestrator (Pipeline multi-modelo) ───────────────────────────────
try:
    from slm_orchestrator import SLMOrchestrator
    HAS_SLM_ORCHESTRATOR = HAS_SLM  # só funciona se MLX estiver disponível
    _slm_orchestrator = None  # lazy init
    if HAS_SLM_ORCHESTRATOR:
        logger.info("SLM Orchestrator disponível. Engine 'V3 Local' ativa.")
except ImportError:
    HAS_SLM_ORCHESTRATOR = False
    _slm_orchestrator = None
    logger.warning("slm_orchestrator.py não encontrado.")

# ── Jurisprudência Search ──────────────────────────────────────────────────
try:
    from jurisprudence_search import search_jurisprudencia  # noqa: F401
    HAS_JURISPRUDENCIA = True
    logger.info("Módulo jurisprudence_search disponível.")
except ImportError:
    HAS_JURISPRUDENCIA = False

# ── Remote SLM Server (Hybrid: Railway → MacBook via Tunnel) ───────────────
import httpx

SLM_SERVER_URL = os.environ.get("SLM_SERVER_URL", "").rstrip("/")  # ex: https://slm.meudominio.com
SLM_SERVER_KEY = os.environ.get("SLM_SERVER_KEY", "")
HAS_REMOTE_SLM = bool(SLM_SERVER_URL)
if HAS_REMOTE_SLM:
    logger.info("SLM Server remoto configurado: %s", SLM_SERVER_URL)


def _call_remote_slm(processo_text: str, estilo: str = "") -> dict:
    """Proxy: envia requisição para o SLM server remoto (MacBook via tunnel)."""
    headers = {"Content-Type": "application/json"}
    if SLM_SERVER_KEY:
        headers["X-SLM-Key"] = SLM_SERVER_KEY

    resp = httpx.post(
        f"{SLM_SERVER_URL}/slm/pipeline",
        json={"processo_text": processo_text, "estilo": estilo},
        headers=headers,
        timeout=300.0,  # 5 min timeout (pipeline pode demorar)
    )
    resp.raise_for_status()
    return resp.json()


def _check_remote_slm_health() -> dict:
    """Checa se o SLM server remoto está online."""
    try:
        headers = {}
        if SLM_SERVER_KEY:
            headers["X-SLM-Key"] = SLM_SERVER_KEY
        resp = httpx.get(
            f"{SLM_SERVER_URL}/slm/health",
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        data["remote"] = True
        data["url"] = SLM_SERVER_URL
        return data
    except Exception as e:
        return {"status": "offline", "remote": True, "url": SLM_SERVER_URL, "error": str(e)[:200]}


@asynccontextmanager
async def lifespan(app):
    logger.info("🚀 Registered routes:")
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        path = getattr(route, 'path', getattr(route, 'path_regex', '?'))
        logger.info("   %s %s", methods or 'MOUNT', path)
    yield

app = FastAPI(title="Jurisbusca API", version="1.0.0", lifespan=lifespan)

# ── Rate Limiting (CESEC §2.2 — rate limiting após tentativas excessivas) ─────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning("Rate limit exceeded: ip=%s path=%s", get_remote_address(request), request.url.path)
    return JSONResponse(
        status_code=429,
        content={"detail": "Limite de requisições excedido. Tente novamente em breve."},
    )

# ── CORS — dynamic origins (CESEC §5.2) ──────────────────────────────────────
_cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = os.getenv("CORS_ORIGINS", "")
if _extra_origins:
    for o in _extra_origins.split(","):
        o = o.strip()
        if not o:
            continue
        # Auto-add https:// if no protocol specified (common Railway config mistake)
        if o and not o.startswith("http://") and not o.startswith("https://"):
            _cors_origins.append(f"https://{o}")
            _cors_origins.append(f"http://{o}")  # Also allow http for flexibility
        else:
            _cors_origins.append(o)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Azure-Key", "Accept"],
)
logger.info("CORS origins: %s", _cors_origins)


# ── Global catch-all: convert Starlette text/plain 500 into JSON with diagnostics ──
from starlette.responses import JSONResponse as _JSONResponse

@app.exception_handler(Exception)
async def _catch_all_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions that would otherwise produce text/plain 500."""
    tb = traceback.format_exc()
    logger.error("‼️ Unhandled exception on %s %s:\n%s", request.method, request.url.path, tb)
    return _JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )


# ── JWT Authentication (CESEC §2.2 — verificação de assinatura) ──────────────
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()
_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip() or os.getenv("VITE_SUPABASE_URL", "").strip()
_security_scheme = HTTPBearer(auto_error=False)

# JWKS client for RS256 tokens (modern Supabase projects)
_jwks_client = None
if _SUPABASE_URL:
    try:
        _jwks_url = f"{_SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
        _jwks_client = jwt.PyJWKClient(_jwks_url, cache_keys=True, lifespan=3600)
        logger.info("JWKS client ready: %s", _jwks_url)
    except Exception as e:
        logger.warning("JWKS client init failed: %s", e)


def verify_jwt(token: str) -> dict:
    """Verify Supabase JWT. Reads alg from header, then dispatches to correct strategy."""
    # 1. Read algorithm from token header (safe — no signature check yet)
    try:
        header = jwt.get_unverified_header(token)
    except Exception as e:
        logger.error("JWT header unreadable: %s", e)
        raise jwt.InvalidTokenError(f"JWT header inválido: {e}")

    alg = header.get("alg", "HS256")
    logger.info("JWT alg=%s, jwks_client=%s, secret_len=%d", alg, bool(_jwks_client), len(SUPABASE_JWT_SECRET))

    # 2. HS256 → verify with shared secret
    if alg == "HS256":
        if not SUPABASE_JWT_SECRET:
            raise jwt.InvalidTokenError("HS256 token mas SUPABASE_JWT_SECRET não configurada")
        return jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"], options={"verify_aud": False})

    # 3. RS256/RS384/RS512 → verify with JWKS public key from Supabase
    if alg.startswith("RS") or alg.startswith("ES") or alg.startswith("PS"):
        if not _jwks_client:
            raise jwt.InvalidTokenError(f"Token usa {alg} mas JWKS client não disponível (SUPABASE_URL={bool(_SUPABASE_URL)})")
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(token, signing_key.key, algorithms=[alg], options={"verify_aud": False})

    # 4. Unknown algorithm
    raise jwt.InvalidTokenError(f"Algoritmo JWT não suportado: {alg}")


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(_security_scheme),
) -> dict:
    """FastAPI dependency: requires valid JWT. Returns decoded payload."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Token não fornecido. Faça login.")
    try:
        payload = verify_jwt(credentials.credentials)
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado. Faça login novamente.")
    except Exception as e:
        # Catch ALL JWT errors (InvalidAlgorithmError is NOT a subclass of InvalidTokenError in PyJWT)
        logger.warning("JWT auth failed for %s: %s: %s", get_remote_address(request), type(e).__name__, str(e)[:200])
        raise HTTPException(status_code=401, detail=f"Token inválido: {str(e)}")



# ── In-memory conversation store (fallback) ──────────────────────────────
import time as _time
from collections import OrderedDict

_CONVERSATIONS_FALLBACK_MAXSIZE = 500
conversations_fallback: OrderedDict[str, list[dict]] = OrderedDict()

def _conversations_set(conv_id: str, value: list):
    """Bounded setter — evicts oldest entry when limit is reached."""
    if conv_id in conversations_fallback:
        conversations_fallback.move_to_end(conv_id)
    else:
        if len(conversations_fallback) >= _CONVERSATIONS_FALLBACK_MAXSIZE:
            conversations_fallback.popitem(last=False)
    conversations_fallback[conv_id] = value

# ── In-memory process RAG cache (LRU, max 50 entries) ────────────────────
_RAG_CACHE_MAXSIZE = 50
_process_rag_cache: OrderedDict[str, tuple] = OrderedDict()

def _rag_cache_set(key: str, value: tuple):
    if key in _process_rag_cache:
        _process_rag_cache.move_to_end(key)
    else:
        if len(_process_rag_cache) >= _RAG_CACHE_MAXSIZE:
            _process_rag_cache.popitem(last=False)
    _process_rag_cache[key] = value

# ── In-memory background task store (shared: upload + xray) ───────────────
# Key: task_id, Value: {"status": str, "result": dict|None, "error": str|None, "progress": str, "created_at": float}
_bg_tasks: dict[str, dict] = {}
_BG_TASK_TTL = 3600  # 1 hour

def _bg_tasks_cleanup():
    """Remove tasks older than TTL that were never polled."""
    now = _time.time()
    expired = [k for k, v in _bg_tasks.items() if now - v.get("created_at", now) > _BG_TASK_TTL]
    for k in expired:
        del _bg_tasks[k]


class ChatRequest(BaseModel):
    message: str
    model: str = "v1"
    llm: Optional[str] = None  # LLM deployment name (e.g. 'gpt-5.3-chat', 'DeepSeek-V3.2-Speciale')
    conversation_id: Optional[str] = None
    agent_prompt: Optional[str] = None
    ocr_engine: str = "paddle"
    uploaded_text: Optional[str] = None  # pre-extracted text from uploaded file
    style_dossier: Optional[str] = None  # cloning prompt from style report
    use_rag: bool = False  # if True, use RAG retrieval instead of full text
    jurisprudence_context: Optional[str] = None  # V0.5: imported jurisprudence for minuta fundamentação


# ── Model mapping (Azure AI Foundry) ──────────────────────────────────────────
MODEL_MAP = {
    "gemini": "gpt-5.3-chat",
    "claude": "gpt-5.3-chat",
    "gpt":    "gpt-5.3-chat",
    "v0":     "gpt-5.3-chat",
    "v0.5":   "gpt-5.3-chat",
    "v1":     "gpt-5.3-chat",
    "v2":     "gpt-5.3-chat",
    "v3-local": "local-slm",
    "local-slm": "local-slm",
}


# ── Context-safe text truncation ───────────────────────────────────────────
# Token limits per model family (conservative — leaves room for system prompt,
# agent prompt, style dossier, RAG templates, and conversation history).
_MODEL_CHAR_LIMITS: dict[str, int] = {
    "v0.5":     750_000,   # GPT-5.3 Azure (272k token limit — reserva ~50k para prompts/histórico)
    "v1":       600_000,   # Gemini (large context, but keep reasonable)
    "v2":       600_000,   # GPT/Claude (~150k tokens)
    "v3":       600_000,   # Multi-model pipeline
    "v3-local": 200_000,   # SLMs (small context)
}
_DEFAULT_CHAR_LIMIT = 600_000  # ~150k tokens


def _truncate_for_context(text: str, model: str = "v1") -> str:
    """Truncate uploaded text to fit within the model's context window.

    Preserves the beginning (cabeçalho/relatório) and end (dispositivo/conclusão)
    of the document, which are the most important parts in Brazilian legal docs.
    """
    limit = _MODEL_CHAR_LIMITS.get(model, _DEFAULT_CHAR_LIMIT)
    if not text or len(text) <= limit:
        return text

    # Keep 80% from the start (relatório, fundamentação) and 20% from the end (dispositivo)
    head_size = int(limit * 0.8)
    tail_size = limit - head_size
    original_len = len(text)
    truncated_chars = original_len - limit

    head = text[:head_size]
    tail = text[-tail_size:]

    marker = (
        f"\n\n[... ⚠️ DOCUMENTO TRUNCADO: {truncated_chars:,} caracteres omitidos "
        f"({original_len:,} → {limit:,} chars) para caber no contexto do modelo. "
        f"Use o modo RAG para busca semântica no texto completo. ...]\n\n"
    )

    logger.warning(
        "Texto truncado para contexto: %s chars → %s chars (modelo=%s)",
        f"{original_len:,}", f"{limit:,}", model,
    )
    return head + marker + tail


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
    import subprocess as _sp, datetime as _dt
    try:
        git_hash = _sp.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=_sp.DEVNULL).decode().strip()
    except Exception:
        git_hash = "unknown"
    return {
        "status": "ok",
        "routes": len(app.routes),
        "commit": git_hash,
        "server_time": _dt.datetime.utcnow().isoformat(),
    }


@app.post("/api/validate-key")
@limiter.limit("5/minute")
async def validate_key(request: Request):
    """Test if an Azure OpenAI API key is valid by making a minimal LLM call."""
    key = request.headers.get("X-Azure-Key", "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Chave não fornecida. Envie no header X-Azure-Key.")

    try:
        from langchain_core.messages import HumanMessage as HM
        llm = be.get_llm(temperature=0.0, api_key=key, max_tokens=10)
        response = llm.invoke([HM(content="Diga OK")])
        return {"valid": True, "message": "Chave válida!"}
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "Unauthorized" in error_str or "invalid" in error_str.lower():
            return {"valid": False, "message": "Chave inválida ou sem permissão."}
        return {"valid": False, "message": f"Erro ao validar: {error_str[:200]}"}


@app.get("/api/debug-routes")
async def debug_routes(_auth: dict = Depends(require_auth)):
    """Diagnostic: list all registered routes."""
    routes = []
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        path = getattr(route, 'path', str(getattr(route, 'path_regex', '?')))
        routes.append({"methods": list(methods) if methods else ["MOUNT"], "path": str(path)})
    return {"routes": routes}


# ── User Memory (Preferences) ─────────────────────────────────────────────────

class MemoryRequest(BaseModel):
    content: str = ""
    enabled: bool = True


@app.get("/api/memory")
async def get_memory_endpoint(_auth: dict = Depends(require_auth)):
    """Get user memory/preferences."""
    user_key = _auth.get("email") or _auth.get("sub", "default")
    return history_db.get_memory(user_key)


@app.put("/api/memory")
async def save_memory_endpoint(req: MemoryRequest, _auth: dict = Depends(require_auth)):
    """Save user memory/preferences (max 2000 chars)."""
    user_key = _auth.get("email") or _auth.get("sub", "default")
    return history_db.save_memory(user_key, req.content, req.enabled)


@app.post("/api/chat")
@limiter.limit("30/minute")
async def chat(req: ChatRequest, request: Request, _auth: dict = Depends(require_auth)):
    """Process a chat message and return LLM response."""
    try:
        # Resolve LLM deployment: prefer explicit llm field, fallback to MODEL_MAP
        model_name = req.llm or MODEL_MAP.get(req.model, "gpt-5.3-chat")

        # Read Azure key from header (frontend sends it), fallback to env var
        azure_key = request.headers.get("X-Azure-Key", "").strip() or None

        # Build the LLM instance (skip for local SLM — handled separately below)
        original_model = model_name
        llm = None
        if model_name != "local-slm":
            try:
                llm = be.get_llm(model_name=model_name, temperature=0.3, api_key=azure_key)
                logger.info("LLM instanciado: %s", model_name)
            except Exception as llm_err:
                logger.warning("Falha ao instanciar %s: %s. Fallback para gpt-5.3-chat.", model_name, llm_err)
                model_name = "gpt-5.3-chat"
                llm = be.get_llm(model_name="gpt-5.3-chat", temperature=0.3, api_key=azure_key)

        conv_id = req.conversation_id or str(uuid.uuid4())

        # ── Truncate uploaded text to fit model context window ──
        if req.uploaded_text:
            req.uploaded_text = _truncate_for_context(req.uploaded_text, req.model)

        # Fetch history (in-memory fallback, no auth)
        past_messages = []
        if conv_id not in conversations_fallback:
            _conversations_set(conv_id, [])
        past_messages = conversations_fallback[conv_id]

        # ── Build LangChain message list with full history ──────────────
        messages = []

        # System prompt: agent instructions + document context
        system_parts = []

        # ── Inject user memory/preferences as first context block ──────
        try:
            _user_key = _auth.get("email") or _auth.get("sub", "default")
            user_mem = history_db.get_memory(_user_key)
            if user_mem["enabled"] and user_mem["content"].strip():
                system_parts.append(
                    f"🧠 **MEMÓRIA/PREFERÊNCIAS DO USUÁRIO (considere sempre):**\n"
                    f"{user_mem['content'].strip()}\n---"
                )
        except Exception as mem_err:
            logger.warning("Erro ao carregar memória do usuário: %s", mem_err)

        if req.agent_prompt:
            system_parts.append(req.agent_prompt)

        # ── Process RAG: retrieve relevant chunks instead of full text ──
        if req.use_rag and req.uploaded_text and _process_rag_cache:
            try:
                # Find the cached retriever (use first available)
                retriever = None
                for cache_key, (cached_ret, _, _) in _process_rag_cache.items():
                    if cached_ret:
                        retriever = cached_ret
                        break

                if retriever:
                    relevant_chunks = retriever.invoke(req.message)
                    if relevant_chunks:
                        chunks_text = "\n\n".join([
                            f"[Pág. {doc.metadata.get('page', '?')} — {doc.metadata.get('extraction', 'text')}]\n{doc.page_content}"
                            for doc in relevant_chunks
                        ])
                        system_parts.append(
                            f"\n\n---\n🎯 **TRECHOS RELEVANTES DO PROCESSO (RAG — {len(relevant_chunks)} chunks):**\n\n"
                            f"{chunks_text}\n---"
                        )
                        total_tokens_approx = sum(len(d.page_content) for d in relevant_chunks) // 4
                        logger.info("RAG Processo: %d chunks recuperados (~%d tokens)", len(relevant_chunks), total_tokens_approx)
                    else:
                        # Fallback to full text if no chunks found
                        system_parts.append(
                            f"\n\n---\n📄 **DOCUMENTO ANEXADO (PEÇA PROCESSUAL):**\n\n{req.uploaded_text}\n---"
                        )
                else:
                    # No retriever cached, use full text
                    system_parts.append(
                        f"\n\n---\n📄 **DOCUMENTO ANEXADO (PEÇA PROCESSUAL):**\n\n{req.uploaded_text}\n---"
                    )
            except Exception as e:
                logger.warning("Process RAG falhou, usando texto completo: %s", e)
                system_parts.append(
                    f"\n\n---\n📄 **DOCUMENTO ANEXADO (PEÇA PROCESSUAL):**\n\n{req.uploaded_text}\n---"
                )
        elif req.uploaded_text:
            system_parts.append(
                f"\n\n---\n📄 **DOCUMENTO ANEXADO (PEÇA PROCESSUAL — TEXTO INTEGRAL EXTRAÍDO VIA OCR):**\n"
                f"⚠️ ATENÇÃO: O texto abaixo foi extraído automaticamente de um PDF via OCR. "
                f"Este É o conteúdo completo do documento/processo judicial. NÃO solicite o envio do arquivo — "
                f"ele já está aqui em formato texto. Analise este conteúdo diretamente.\n\n"
                f"{req.uploaded_text}\n---"
            )
        if req.style_dossier:
            system_parts.append(
                f"\n\n---\n🧬 **SYSTEM PROMPT DE CLONAGEM ESTILÍSTICA (DOSSIÊ DO MAGISTRADO):**\n"
                f"⚠️ INSTRUÇÃO PRIMÁRIA: Replique rigorosamente o estilo descrito abaixo ao redigir qualquer decisão.\n\n"
                f"{req.style_dossier}\n---"
            )

        # V0.5: Inject imported jurisprudence into the LLM context
        if req.jurisprudence_context:
            system_parts.append(
                f"\n\n---\n📚 **JURISPRUDÊNCIA SELECIONADA PELO MAGISTRADO (INCLUIR NA FUNDAMENTAÇÃO DA MINUTA):**\n"
                f"⚠️ INSTRUÇÃO: O magistrado selecionou a(s) jurisprudência(s) abaixo como relevante(s) para este caso. "
                f"INCLUA obrigatoriamente na fundamentação da minuta, citando o número do processo e o entendimento.\n\n"
                f"{req.jurisprudence_context}\n---"
            )

        # ── Auto RAG: retrieve mirror context from persisted templates ──
        if req.uploaded_text:
            try:
                if True:  # Templates use local storage, no API key needed
                    collection_name = "rag_templates_persistent"
                    rag_retriever = be.load_persistent_rag(collection_name=collection_name)
                    if rag_retriever:
                        # Search for similar cases
                        relevant_docs = rag_retriever.invoke(req.uploaded_text[:6000])
                        if relevant_docs:
                            mirror_doc = relevant_docs[0]
                            rag_block = (
                                f"\n\n---\n💎 **CASO ESPELHO (GOLDEN SAMPLE - MODELO MAIS SIMILAR):**\n"
                                f"⚠️ O caso abaixo ({mirror_doc.metadata.get('source', '?')}) é o seu GABARITO ESTRUTURAL.\n"
                                f"1. Copie a estrutura de tópicos (titulação, numeração).\n"
                                f"2. Se for o mesmo assunto, adapte apenas os fatos e nomes, mantendo a fundamentação jurídica.\n\n"
                                f"--- INÍCIO DO CASO ESPELHO ---\n{mirror_doc.page_content}\n--- FIM DO CASO ESPELHO ---\n"
                            )
                            # Add secondary references
                            for i, doc in enumerate(relevant_docs[1:3]):
                                rag_block += f"\n[MODELO SECUNDÁRIO {i+2} - {doc.metadata.get('source', '?')}]:\n{doc.page_content[:3000]}\n"
                            rag_block += "---"
                            system_parts.append(rag_block)

                    # Also inject cached style dossier if not already provided
                    if not req.style_dossier and be._style_dossier_cache:
                        # Use the first (and usually only) cached dossier
                        for cached in be._style_dossier_cache.values():
                            cloning = cached.get('cloning_prompt', '')
                            glossary = cached.get('glossary', '')
                            if cloning:
                                style_block = (
                                    f"\n\n---\n🧬 **SYSTEM PROMPT DE CLONAGEM (ESTILO DO MAGISTRADO):**\n"
                                    f"⚠️ INSTRUÇÃO PRIMÁRIA: Replique rigorosamente o estilo descrito abaixo.\n\n"
                                    f"{cloning}\n"
                                )
                                if glossary:
                                    style_block += f"\n📝 **GLOSSÁRIO DO MAGISTRADO:**\n{glossary}\n"
                                style_block += "---"
                                system_parts.append(style_block)
                            break
            except Exception as e:
                logger.warning("RAG auto-retrieval failed (non-blocking): %s", e)

        if system_parts:
            messages.append(SystemMessage(content="\n".join(system_parts)))

        # Append previous conversation turns
        for turn in past_messages:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))

        # Append current user message — ensure non-empty content for Gemini API
        user_content = req.message.strip() if req.message else ""
        if not user_content:
            # Provide a default prompt when user sends empty message but context exists
            if req.uploaded_text and req.agent_prompt:
                user_content = "Analise o documento anexado conforme as instruções do agente."
            elif req.uploaded_text:
                user_content = "Analise o documento anexado."
            elif req.agent_prompt:
                user_content = "Proceda conforme as instruções do agente."
            else:
                raise HTTPException(status_code=400, detail="Mensagem não pode estar vazia.")
        messages.append(HumanMessage(content=user_content))

        # Persist user turn in history (in-memory)
        conversations_fallback[conv_id].append({"role": "user", "content": user_content})

        # Call LLM or specific Engine workflow
        if req.model == "v2":
            # Call Orchestrator V2 (Hybrid/Linear 3-Stage Workflow)
            if getattr(be, "run_hybrid_orchestration", None):
                # keys dict is no longer needed — get_llm() reads Azure env vars directly
                keys = {}
                
                context_str = req.uploaded_text or ""
                user_msg = req.message or ""
                full_text = f"DADOS DO PROCESSO:\n{context_str}\n\nPEDIDO:\n{user_msg}"
                style_guide = req.style_dossier or ""
                
                v2_result = be.run_hybrid_orchestration(full_text, keys, style_guide)
                
                # Return structured sections so frontend can render collapsible cards
                triage_text = v2_result.get('final_report', '')
                draft_text = v2_result.get('final_output', '')
                audit_text = v2_result.get('audit_report', '')
                
                # Build the main response (the minuta) as primary content
                response_text = draft_text if draft_text.strip() else "(Nenhuma minuta gerada)"
                
                # Store V2 sections for structured frontend rendering
                v2_sections = {
                    "triage": triage_text,
                    "draft": draft_text,
                    "audit": audit_text,
                }
            else:
                response_text = "Erro: V2 Engine (run_hybrid_orchestration) não importada ou indisponível"
                
        elif req.model == "v3":
            # Call Orchestrator V3 (Autonomous Magistrate with LangGraph)
            # V3 uses 3 models sequentially (Kimi→DeepSeek→GPT), which can be slow.
            # We add a timeout + single-model fallback to prevent Railway proxy timeouts.
            import concurrent.futures
            
            context_str = req.uploaded_text or ""
            user_msg = req.message or ""
            full_text = f"{context_str}\n\nPEDIDO DO USUÁRIO:\n{user_msg}" if user_msg.strip() else context_str
            keys = {}  # Azure env vars are read internally

            V3_TIMEOUT = 240  # 4 minutes max (Railway has ~5 min proxy timeout)

            if getattr(be, "run_autonomous_magistrate", None):
                try:
                    # Run V3 MoE pipeline with timeout
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(be.run_autonomous_magistrate, full_text, keys, model_name)
                        v3_json, v3_logs = future.result(timeout=V3_TIMEOUT)

                    # Format the response
                    logs_text = "\n".join(v3_logs) if v3_logs else ""
                    if v3_json.get("error"):
                        response_text = f"⚠️ **V3 finalizou com erro:** {v3_json['error']}\n\n**Logs:**\n{logs_text}\n\n**Raw:**\n{v3_json.get('raw', '')}"
                    else:
                        minuta = v3_json.get("minuta_final", v3_json.get("raw", ""))
                        if isinstance(minuta, dict):
                            minuta = minuta.get("texto", json.dumps(minuta, ensure_ascii=False, indent=2))
                        response_text = f"{minuta}\n\n---\n**🔍 Logs V3:**\n{logs_text}"

                except concurrent.futures.TimeoutError:
                    print(f"⚠️ V3 MoE pipeline timed out after {V3_TIMEOUT}s. Falling back to single-model GPT-5.3.")
                    # Fallback: single GPT-5.3 call with the full document
                    try:
                        fallback_llm = be.get_llm(model_name="gpt-5.3-chat")
                        fb_messages = [
                            SystemMessage(content=(
                                "Você é um Magistrado Autônomo de alto nível. "
                                "Leia os autos integrais do processo abaixo e produza:\n"
                                "1. Um resumo dos fatos e pedidos\n"
                                "2. A fundamentação jurídica\n"
                                "3. O dispositivo (decisão final)\n\n"
                                "Formate a resposta como uma minuta de decisão/sentença completa, "
                                "pronta para assinatura."
                            )),
                            HumanMessage(content=f"AUTOS DO PROCESSO:\n{full_text}")
                        ]
                        fb_response = fallback_llm.invoke(fb_messages)
                        response_text = f"{be.safe_content(fb_response)}\n\n---\n⚠️ *V3 MoE excedeu o tempo limite. Resultado gerado por GPT-5.3 (modelo único).*"
                    except Exception as fb_err:
                        response_text = f"⚠️ **V3 timeout ({V3_TIMEOUT}s) e fallback GPT-5.3 também falhou:** {str(fb_err)}"

                except Exception as e:
                    traceback.print_exc()
                    # Fallback: single GPT-5.3 call
                    try:
                        fallback_llm = be.get_llm(model_name="gpt-5.3-chat")
                        fb_messages = [
                            SystemMessage(content=(
                                "Você é um Magistrado Autônomo de alto nível. "
                                "Leia os autos integrais do processo abaixo e produza:\n"
                                "1. Um resumo dos fatos e pedidos\n"
                                "2. A fundamentação jurídica\n"
                                "3. O dispositivo (decisão final)\n\n"
                                "Formate a resposta como uma minuta de decisão/sentença completa."
                            )),
                            HumanMessage(content=f"AUTOS DO PROCESSO:\n{full_text}")
                        ]
                        fb_response = fallback_llm.invoke(fb_messages)
                        response_text = f"{be.safe_content(fb_response)}\n\n---\n⚠️ *V3 MoE falhou ({str(e)[:100]}). Resultado gerado por GPT-5.3 (modelo único).*"
                    except Exception as fb_err:
                        response_text = f"⚠️ **Erro no V3 Engine:** {str(e)}\n\n**Fallback GPT-5.3 também falhou:** {str(fb_err)}"
            else:
                response_text = "Erro: V3 Engine (run_autonomous_magistrate) não foi importada. Verifique se langgraph está instalado."
        elif req.model == "v3-local" and (HAS_REMOTE_SLM or HAS_SLM_ORCHESTRATOR):
            # ── V3 Local: Pipeline de 5 SLMs (remoto via tunnel OU local) ──
            try:
                processo_text = req.uploaded_text or req.message
                estilo = req.style_dossier or ""

                if HAS_REMOTE_SLM:
                    # ── Proxy para SLM Server remoto (MacBook via tunnel) ──
                    result = _call_remote_slm(processo_text, estilo)
                else:
                    # ── Execução local (dev no MacBook) ──
                    global _slm_orchestrator
                    if _slm_orchestrator is None:
                        _slm_orchestrator = SLMOrchestrator(load_all=False)
                    result = _slm_orchestrator.run_pipeline(
                        processo_text=processo_text,
                        estilo=estilo,
                    )

                # Formatar resposta como a V2 (minuta + cards colapsáveis)
                minuta = result.get("minuta", "")
                timing = result.get("timing", {})
                audit = result.get("audit", {})
                fatos = result.get("fatos", {})
                rota = result.get("rota", {})

                timing_str = (f"Router: {timing.get('router', '?')}s | "
                              f"Extrator: {timing.get('extrator', '?')}s | "
                              f"Jurista: {timing.get('jurista', '?')}s | "
                              f"Redator: {timing.get('redator', '?')}s | "
                              f"Auditor: {timing.get('auditor', '?')}s | "
                              f"**Total: {timing.get('total', '?')}s**")

                source = "remoto" if HAS_REMOTE_SLM else "local"
                response_text = minuta
                model_name = f"SLM Pipeline ({timing.get('total', '?')}s, {source})"

                # V2-style sections for collapsible cards
                v2_sections = {
                    "draft": minuta,
                    "triage": (f"**Tipo:** {rota.get('tipo', '?')}\n"
                               f"**Matéria:** {rota.get('materia', '?')}\n"
                               f"**Confiança:** {rota.get('confianca', '?')}\n\n"
                               f"**Fatos extraídos:**\n"
                               f"- Autor: {fatos.get('autor', '?')}\n"
                               f"- Réu: {fatos.get('reu', '?')}\n"
                               f"- Ação: {fatos.get('acao', '?')}\n"
                               f"- Valor: {fatos.get('valor_causa', '?')}\n\n"
                               f"⏱️ {timing_str}"),
                    "audit": (f"**Aprovado:** {'✅ Sim' if audit.get('aprovado') else '❌ Não'}\n"
                              f"**Score:** {audit.get('score', '?')}/100\n"
                              f"**Resumo:** {audit.get('resumo', 'N/A')}\n\n"
                              f"**Modelos usados:**\n"
                              f"- Router: {result.get('model_info', {}).get('router', '?')}\n"
                              f"- Extrator: {result.get('model_info', {}).get('extrator', '?')}\n"
                              f"- Jurista: {result.get('model_info', {}).get('jurista', '?')}\n"
                              f"- Redator: {result.get('model_info', {}).get('redator', '?')}\n"
                              f"- Auditor: {result.get('model_info', {}).get('auditor', '?')}"),
                }

            except Exception as e:
                traceback.print_exc()
                response_text = f"⚠️ **Erro no pipeline V3 Local:** {str(e)[:300]}"
                model_name = "SLM Pipeline (erro)"
        elif model_name == "local-slm" and HAS_SLM:
            # ── Local SLM inference via MLX ──────────────────────────────
            try:
                # Convert LangChain messages to plain dicts for slm_engine
                slm_messages = []
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        slm_messages.append({"role": "system", "content": msg.content})
                    elif isinstance(msg, HumanMessage):
                        slm_messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        slm_messages.append({"role": "assistant", "content": msg.content})

                response_text = slm_engine.generate_chat(
                    slm_messages,
                    model_key="main",
                    max_tokens=4096,
                    temperature=0.3,
                )
                model_name = slm_engine.MODELS["main"]["name"]
            except Exception as slm_err:
                print(f"⚠️ SLM local falhou: {slm_err}. Fallback para GPT-5.3.")
                traceback.print_exc()
                try:
                    fallback_llm = be.get_llm(model_name="gpt-5.3-chat", temperature=0.3, api_key=azure_key)
                    response = fallback_llm.invoke(messages)
                    response_text = f"{be.safe_content(response)}\n\n---\n⚠️ *SLM local falhou ({str(slm_err)[:100]}). Resultado gerado por GPT-5.3.*"
                    model_name = "gpt-5.3-chat"
                except Exception as fb_err:
                    response_text = f"⚠️ **Erro:** SLM local falhou ({str(slm_err)[:150]}). Fallback GPT-5.3 também falhou: {str(fb_err)[:150]}"
        else:
            # Default V0/V1 - just invoke LLM (Chat-based logic)
            try:
                response = llm.invoke(messages)
                response_text = be.safe_content(response)
            except Exception as invoke_err:
                # If model fails at invocation, retry with GPT-5.3
                if original_model != "gpt-5.3-chat":
                    print(f"⚠️ {original_model} invoke falhou: {invoke_err}. Retrying com GPT-5.3.")
                    try:
                        fallback_llm = be.get_llm(model_name="gpt-5.3-chat", temperature=0.3, api_key=azure_key)
                        response = fallback_llm.invoke(messages)
                        response_text = f"{be.safe_content(response)}\n\n---\n⚠️ *Modelo {original_model} indisponível. Resultado gerado por GPT-5.3.*"
                        model_name = "gpt-5.3-chat"
                    except Exception as fb_err:
                        response_text = f"⚠️ **Erro:** {original_model} falhou ({str(invoke_err)[:150]}). Fallback GPT-5.3 também falhou: {str(fb_err)[:150]}"
                else:
                    raise invoke_err

        # Persist assistant turn in history (in-memory)
        conversations_fallback[conv_id].append({"role": "assistant", "content": response_text})

        # Build response payload
        result_payload = {
            "conversation_id": conv_id,
            "response": response_text,
            "model": model_name,
        }
        
        # Include V2 structured sections if available
        if req.model in ("v2", "v3-local") and 'v2_sections' in locals():
            result_payload["v2_sections"] = v2_sections

        # ── V0.5: trigger background jurisprudence research ──────────────
        # Only on the SECOND interaction (after triagem is done), when there's history
        if req.model == "v0.5" and req.uploaded_text and HAS_JURISPRUDENCIA and len(past_messages) >= 2:
            juris_task_id = str(uuid.uuid4())
            _bg_tasks[juris_task_id] = {
                "status": "pending",
                "result": None,
                "error": None,
                "progress": "Extraindo temas jurídicos do processo...",
                "created_at": _time.time(),
            }
            thread = threading.Thread(
                target=_run_jurisprudence_research_background,
                args=(juris_task_id, req.uploaded_text, response_text),
                daemon=True,
            )
            thread.start()
            result_payload["jurisprudence_task_id"] = juris_task_id

        return result_payload

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")


@app.get("/api/history")
async def get_history(_auth: dict = Depends(require_auth)):
    """Fetch conversation history (currently returns empty — no auth)."""
    return {"conversations": []}


def _run_upload_background(task_id: str, file_data: bytes, filename: str, ocr_engine: str, compress: bool, vectorize: bool = True):
    """Background worker: extracts text from uploaded file and stores result."""
    try:
        _bg_tasks[task_id]["status"] = "running"
        _bg_tasks[task_id]["progress"] = f"Extraindo texto de {filename}..."

        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                full_text, retriever = be.process_uploaded_file(
                    f,
                    filename,
                    ocr_engine_choice=ocr_engine,
                    compress=compress,
                    vectorize=vectorize,
                )
        finally:
            os.unlink(tmp_path)

        if not full_text or not full_text.strip():
            _bg_tasks[task_id]["status"] = "error"
            _bg_tasks[task_id]["error"] = "Não foi possível extrair texto do arquivo."
            return

        # Cache the retriever for RAG mode
        rag_available = False
        if retriever:
            cache_key = hashlib.md5(f"{filename}:{len(full_text)}".encode()).hexdigest()
            _rag_cache_set(cache_key, (retriever, full_text, len(full_text)))
            rag_available = True
            logger.info("Process RAG cached: key=%s", cache_key[:8])

        _bg_tasks[task_id]["status"] = "done"
        _bg_tasks[task_id]["result"] = {
            "filename": filename,
            "text": full_text,
            "char_count": len(full_text),
            "rag_available": rag_available,
        }

    except Exception as e:
        traceback.print_exc()
        _bg_tasks[task_id]["status"] = "error"
        err_type = type(e).__name__
        err_msg = str(e)
        if "AZURE_OPENAI" in err_msg:
            _bg_tasks[task_id]["error"] = f"Erro de configuração: variáveis Azure OpenAI não configuradas. O texto foi extraído mas a vetorização falhou."
        elif "TimeoutError" in err_type or "timeout" in err_msg.lower():
            _bg_tasks[task_id]["error"] = f"Timeout: o processamento do arquivo excedeu o tempo limite. Tente um arquivo menor ou desative a vetorização."
        else:
            _bg_tasks[task_id]["error"] = f"Erro no upload ({err_type}): {err_msg}"


@app.post("/api/upload")
@limiter.limit("10/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    ocr_engine: str = Form("paddle"),
    compress: bool = Form(True),
    vectorize: bool = Form(True),
    _auth: dict = Depends(require_auth),
):
    """Upload and extract text from a file. Returns task_id for polling."""
    try:
        content = await file.read()
        filename = file.filename or "documento.pdf"

        task_id = str(uuid.uuid4())
        _bg_tasks_cleanup()
        _bg_tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "progress": f"Processando {filename}...",
            "created_at": _time.time(),
        }

        thread = threading.Thread(
            target=_run_upload_background,
            args=(task_id, content, filename, ocr_engine, compress, vectorize),
            daemon=True,
        )
        thread.start()

        return {"task_id": task_id, "status": "pending"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no upload: {str(e)}")


@app.get("/api/upload/{task_id}")
async def upload_status(task_id: str, _auth: dict = Depends(require_auth)):
    """Poll the status of a background upload task."""
    task = _bg_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada.")

    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", ""),
    }

    if task["status"] == "done":
        response["result"] = task["result"]
        del _bg_tasks[task_id]
    elif task["status"] == "error":
        response["error"] = task.get("error", "Erro desconhecido")
        del _bg_tasks[task_id]

    return response


def _run_xray_background(task_id: str, file_data: list[tuple[str, bytes]]):
    """Background worker: runs the MAP-REDUCE xray pipeline and stores result."""
    import io

    try:
        _bg_tasks[task_id]["status"] = "running"
        _bg_tasks[task_id]["progress"] = "Extraindo texto dos arquivos..."

        file_objects = []
        for fname, content_bytes in file_data:
            buf = io.BytesIO(content_bytes)
            buf.name = fname
            buf.seek(0)
            file_objects.append(buf)

        _bg_tasks[task_id]["progress"] = f"Analisando {len(file_objects)} processos (MAP-REDUCE)..."

        report, text_cache = be.generate_batch_xray(file_objects, None)

        if isinstance(report, dict) and "error" in report:
            _bg_tasks[task_id]["status"] = "error"
            _bg_tasks[task_id]["error"] = report.get("error", "Erro desconhecido")
            _bg_tasks[task_id]["result"] = {"report": report, "text_cache": text_cache}
        else:
            _bg_tasks[task_id]["status"] = "done"
            _bg_tasks[task_id]["result"] = {
                "report": report,
                "file_count": len(file_objects),
                "text_cache": text_cache,
            }

    except Exception as e:
        traceback.print_exc()
        _bg_tasks[task_id]["status"] = "error"
        _bg_tasks[task_id]["error"] = f"Erro no Raio-X: {str(e)}"


@app.post("/api/xray")
async def batch_xray(files: list[UploadFile] = File(...), _auth: dict = Depends(require_auth)):
    """
    Batch X-Ray: upload files and start background MAP-REDUCE clustering.
    Returns a task_id immediately — poll GET /api/xray/{task_id} for results.
    """
    try:
        # Read all file content upfront (before the request lifecycle ends)
        file_data: list[tuple[str, bytes]] = []
        for f in files:
            content = await f.read()
            file_data.append((f.filename or "documento.pdf", content))

        task_id = str(uuid.uuid4())
        _bg_tasks_cleanup()
        _bg_tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "progress": "Tarefa criada, aguardando início...",
            "created_at": _time.time(),
        }

        # Spawn background thread
        thread = threading.Thread(
            target=_run_xray_background,
            args=(task_id, file_data),
            daemon=True,
        )
        thread.start()

        return {"task_id": task_id, "status": "pending"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao iniciar Raio-X: {str(e)}")


@app.get("/api/xray/{task_id}")
async def xray_status(task_id: str, _auth: dict = Depends(require_auth)):
    """
    Poll the status of a background X-Ray task.
    Returns status ('pending', 'running', 'done', 'error') and result when done.
    """
    task = _bg_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada.")

    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", ""),
    }

    if task["status"] == "done":
        response["result"] = task["result"]
        # Clean up after delivery
        del _bg_tasks[task_id]
    elif task["status"] == "error":
        response["error"] = task.get("error", "Erro desconhecido")
        del _bg_tasks[task_id]

    return response


@app.post("/api/style-report")
async def style_report(files: list[UploadFile] = File(...), _auth: dict = Depends(require_auth)):
    """
    Generate a Style Dossier (Dossiê de Identidade Decisional) from template files.
    Uses the existing generate_style_dossier pipeline in backend.py.
    """
    import io

    try:
        # Convert UploadFiles into file-like objects with .name attribute
        file_objects = []
        for f in files:
            content = await f.read()
            buf = io.BytesIO(content)
            buf.name = f.filename or "template.pdf"
            buf.seek(0)
            file_objects.append(buf)

        # Call the existing style dossier pipeline (uses Azure OpenAI internally)
        result = be.generate_style_dossier(file_objects, None)

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


# ── Cluster Batch Analysis (individual parallel processing) ───────────────────

class ClusterAnalyzeRequest(BaseModel):
    processes: list[dict]  # [{"filename": str, "text": str}]
    agent_prompt: Optional[str] = None
    model: str = "gemini"
    llm: Optional[str] = None


def _analyze_single_process(filename: str, text: str, agent_prompt: str, model_name: str, collection_name: str = "rag_templates_persistent"):
    """Analyze one process. Runs in a thread pool. Retries on 429 rate limit."""
    import time as _time

    MAX_RETRIES = 5
    BASE_DELAY = 30  # seconds (Azure S0 asks for 60s, so 30→60→120 covers it)

    for attempt in range(MAX_RETRIES + 1):
        try:
            llm = be.get_llm(model_name=model_name, temperature=0.3)

            messages = []
            system_parts = []

            if agent_prompt:
                system_parts.append(agent_prompt)

            system_parts.append(
                f"\n\n---\n📄 **DOCUMENTO ANEXADO (PEÇA PROCESSUAL):**\n\n{text}\n---"
            )

            # Auto-RAG: inject golden sample from persistent templates
            op_key = os.getenv("AZURE_OPENAI_API_KEY", "")
            if op_key:
                try:
                    rag_retriever = be.load_persistent_rag(collection_name=collection_name)
                    if rag_retriever:
                        relevant_docs = rag_retriever.invoke(text[:6000])
                        if relevant_docs:
                            mirror_doc = relevant_docs[0]
                            rag_block = (
                                f"\n\n---\n💎 **CASO ESPELHO (GOLDEN SAMPLE):**\n"
                                f"O caso abaixo ({mirror_doc.metadata.get('source', '?')}) é o GABARITO ESTRUTURAL.\n"
                                f"Copie a estrutura, adapte fatos e nomes.\n\n"
                                f"--- INÍCIO ---\n{mirror_doc.page_content}\n--- FIM ---\n---"
                            )
                            system_parts.append(rag_block)

                    # Inject cached style dossier
                    if be._style_dossier_cache:
                        for cached in be._style_dossier_cache.values():
                            cloning = cached.get('cloning_prompt', '')
                            if cloning:
                                system_parts.append(
                                    f"\n\n---\n🧬 **CLONAGEM ESTILÍSTICA:**\n{cloning}\n---"
                                )
                            break
                except Exception as e:
                    print(f"⚠️ RAG failed for {filename}: {e}")

            messages.append(SystemMessage(content="\n".join(system_parts)))
            messages.append(HumanMessage(content="Analise o documento anexado e gere a minuta de decisão conforme as instruções."))

            response = llm.invoke(messages)
            response_text = be.safe_content(response)

            return {
                "filename": filename,
                "status": "ok",
                "response": response_text,
                "model": model_name,
            }
        except Exception as e:
            error_str = str(e)
            # Retry on 429 rate limit errors
            if "429" in error_str and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** attempt)  # 15s, 30s, 60s
                print(f"⚠️ Rate limit (429) for {filename}, retry {attempt + 1}/{MAX_RETRIES} after {delay}s...")
                _time.sleep(delay)
                continue
            return {
                "filename": filename,
                "status": "error",
                "response": f"Erro: {error_str}",
                "model": model_name,
            }


@app.post("/api/cluster-analyze")
async def cluster_analyze(req: ClusterAnalyzeRequest, _auth: dict = Depends(require_auth)):
    """
    Process all files in a cluster individually, in parallel.
    Returns a list of individual analysis results.
    """
    import concurrent.futures

    try:
        # Resolve LLM deployment: prefer explicit llm field, fallback to MODEL_MAP
        model_name = req.llm or MODEL_MAP.get(req.model, "gpt-5.3-chat")

        if not req.processes:
            raise HTTPException(status_code=400, detail="Nenhum processo para analisar.")

        collection_name = "rag_templates_persistent"

        results = []
        # Serial execution (max_workers=1) to respect Azure S0 token-per-minute limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {}
            for i, p in enumerate(req.processes):
                if not p.get("text"):
                    continue
                # Stagger submissions to avoid rate limit bursts (Azure S0)
                if i > 0:
                    import time as _time
                    _time.sleep(5)
                future = executor.submit(
                    _analyze_single_process,
                    p["filename"],
                    p["text"],
                    req.agent_prompt or "",
                    model_name,
                    collection_name
                )
                futures[future] = p["filename"]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # Sort by filename for consistent ordering
        results.sort(key=lambda r: r["filename"])

        return {
            "results": results,
            "total": len(results),
            "ok_count": sum(1 for r in results if r["status"] == "ok"),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na análise em lote: {str(e)}")


# ── Batch Pilot Replication (pilot-guided batch analysis) ─────────────────────

class BatchPilotReplicateRequest(BaseModel):
    pilot_instructions: str          # Consolidated judge instructions from pilot conversation
    pilot_minuta: str                # Draft decision generated during pilot (template/gabarito)
    pilot_summary: Optional[str] = None  # Summary of pilot case facts for comparison
    processes: list[dict]            # [{filename: str, text: str}] — remaining processes
    model: str = "gpt53"
    llm: Optional[str] = None
    agent_prompt: Optional[str] = None


def _replicate_single_process(
    filename: str,
    text: str,
    pilot_instructions: str,
    pilot_minuta: str,
    pilot_summary: str,
    agent_prompt: str,
    model_name: str,
    collection_name: str = "rag_templates_persistent",
):
    """
    Replicate a pilot case's instructions + draft to a new process.
    Highlights differences with ⚠️ alerts in the generated draft.
    Runs in a thread pool. Retries on 429 rate limit.
    """
    import time as _time

    MAX_RETRIES = 5
    BASE_DELAY = 30

    for attempt in range(MAX_RETRIES + 1):
        try:
            llm = be.get_llm(model_name=model_name, temperature=0.3)

            system_parts = []

            # 1. Agent base prompt (if any)
            if agent_prompt:
                system_parts.append(agent_prompt)

            # 2. Pilot instructions block
            system_parts.append(
                f"\n\n---\n🧑‍⚖️ **INSTRUÇÕES DO MAGISTRADO (CASO PILOTO):**\n"
                f"As instruções abaixo foram estabelecidas pelo magistrado durante a análise interativa "
                f"de um caso piloto. SIGA RIGOROSAMENTE estas diretrizes ao elaborar a minuta.\n\n"
                f"{pilot_instructions}\n---"
            )

            # 3. Pilot minuta as structural template
            system_parts.append(
                f"\n\n---\n📋 **MINUTA-GABARITO (CASO PILOTO):**\n"
                f"A minuta abaixo foi gerada e APROVADA pelo magistrado para o caso piloto. "
                f"Use-a como MODELO ESTRUTURAL e de conteúdo jurídico.\n"
                f"- Se o novo processo for IDÊNTICO em matéria/fatos, replique a estrutura e "
                f"fundamentação, adaptando apenas nomes, datas e dados específicos.\n"
                f"- Se houver DIFERENÇAS relevantes (fatos, pedidos, partes, matéria), "
                f"DESTAQUE cada diferença com:\n"
                f"  ⚠️ **ALERTA DE DIFERENÇA:** [descrição da divergência e como impacta a decisão]\n\n"
                f"--- INÍCIO DA MINUTA-GABARITO ---\n{pilot_minuta}\n--- FIM DA MINUTA-GABARITO ---\n---"
            )

            # 4. Pilot summary for comparison
            if pilot_summary:
                system_parts.append(
                    f"\n\n---\n📝 **RESUMO DOS FATOS DO CASO PILOTO (para comparação):**\n"
                    f"{pilot_summary}\n---"
                )

            # 5. New process text
            system_parts.append(
                f"\n\n---\n📄 **NOVO PROCESSO PARA ANÁLISE:**\n\n{text}\n---"
            )

            # 6. Auto-RAG: inject golden sample from persistent templates
            op_key = os.getenv("AZURE_OPENAI_API_KEY", "")
            if op_key:
                try:
                    rag_retriever = be.load_persistent_rag(collection_name=collection_name)
                    if rag_retriever:
                        relevant_docs = rag_retriever.invoke(text[:6000])
                        if relevant_docs:
                            mirror_doc = relevant_docs[0]
                            rag_block = (
                                f"\n\n---\n💎 **CASO ESPELHO (GOLDEN SAMPLE):**\n"
                                f"O caso abaixo ({mirror_doc.metadata.get('source', '?')}) é referência adicional.\n\n"
                                f"--- INÍCIO ---\n{mirror_doc.page_content}\n--- FIM ---\n---"
                            )
                            system_parts.append(rag_block)

                    # Inject cached style dossier
                    if be._style_dossier_cache:
                        for cached in be._style_dossier_cache.values():
                            cloning = cached.get('cloning_prompt', '')
                            if cloning:
                                system_parts.append(
                                    f"\n\n---\n🧬 **CLONAGEM ESTILÍSTICA:**\n{cloning}\n---"
                                )
                            break
                except Exception as e:
                    print(f"⚠️ RAG failed for {filename}: {e}")

            messages = [
                SystemMessage(content="\n".join(system_parts)),
                HumanMessage(content=(
                    "Elabore a minuta de decisão para este NOVO processo, usando a minuta-gabarito "
                    "como modelo estrutural e seguindo rigorosamente as instruções do magistrado.\n\n"
                    "REGRAS CRÍTICAS:\n"
                    "1. Compare os fatos deste processo com o caso piloto.\n"
                    "2. Se for similar, replique a estrutura adaptando dados.\n"
                    "3. Se houver qualquer diferença relevante (fatos, partes, pedidos, matéria, "
                    "valores, provas), insira ⚠️ **ALERTA DE DIFERENÇA** explicando a divergência "
                    "e suas implicações.\n"
                    "4. Mantenha a fundamentação jurídica do gabarito quando aplicável.\n"
                    "5. Gere a minuta COMPLETA, não apenas as diferenças."
                )),
            ]

            response = llm.invoke(messages)
            response_text = be.safe_content(response)

            # Count alerts in the response
            alerts_count = response_text.count("ALERTA DE DIFERENÇA")

            return {
                "filename": filename,
                "status": "ok",
                "response": response_text,
                "model": model_name,
                "alerts_count": alerts_count,
                "is_pilot_replica": True,
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"⚠️ Rate limit (429) for {filename}, retry {attempt + 1}/{MAX_RETRIES} after {delay}s...")
                _time.sleep(delay)
                continue
            return {
                "filename": filename,
                "status": "error",
                "response": f"Erro: {error_str}",
                "model": model_name,
                "alerts_count": 0,
                "is_pilot_replica": True,
            }


@app.post("/api/batch-pilot/replicate")
async def batch_pilot_replicate(req: BatchPilotReplicateRequest, _auth: dict = Depends(require_auth)):
    """
    Replicate pilot case instructions + draft to remaining processes.
    Each process gets its own isolated LLM call with the pilot context.
    Returns individual results with difference alerts.
    """
    import concurrent.futures

    try:
        model_name = req.llm or MODEL_MAP.get(req.model, "gpt-5.3-chat")

        if not req.processes:
            raise HTTPException(status_code=400, detail="Nenhum processo para replicar.")

        if not req.pilot_minuta:
            raise HTTPException(status_code=400, detail="Minuta do caso piloto é obrigatória.")

        collection_name = "rag_templates_persistent"

        results = []
        # Serial execution to respect rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = {}
            for i, p in enumerate(req.processes):
                if not p.get("text"):
                    continue
                if i > 0:
                    import time as _time
                    _time.sleep(5)
                future = executor.submit(
                    _replicate_single_process,
                    p["filename"],
                    p["text"],
                    req.pilot_instructions,
                    req.pilot_minuta,
                    req.pilot_summary or "",
                    req.agent_prompt or "",
                    model_name,
                    collection_name,
                )
                futures[future] = p["filename"]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: r["filename"])

        total_alerts = sum(r.get("alerts_count", 0) for r in results)

        return {
            "results": results,
            "total": len(results),
            "ok_count": sum(1 for r in results if r["status"] == "ok"),
            "total_alerts": total_alerts,
            "pilot_mode": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na replicação piloto: {str(e)}")


# ── Template Management (Persistent RAG) ─────────────────────────────────────

@app.post("/api/templates")
async def upload_templates(
    files: list[UploadFile] = File(...),
    request: Request = None,
    _auth: dict = Depends(require_auth),
):
    """
    Upload and index template files in ChromaDB for persistent RAG.
    Also auto-generates the style dossier.
    """
    import io

    try:
        import time as _time
        import asyncio

        # Extract user identity from JWT for Supabase persistence
        user_id = _extract_user_id(request) or "default"
        token = _extract_token(request)
        logger.info("upload_templates: user=%s files=%d", user_id[:8] if user_id else "?", len(files))

        # Convert UploadFiles into file-like objects
        file_objects = []
        for f in files:
            content = await f.read()
            buf = io.BytesIO(content)
            buf.name = f.filename or "template.pdf"
            buf.seek(0)
            file_objects.append(buf)
            logger.info("  arquivo: %s (%d bytes)", f.filename, len(content))

        # 1. Index templates (100% local — no ChromaDB, no embeddings)
        t0 = _time.time()
        collection_name = "rag_templates_persistent"
        logger.info("Iniciando process_templates para user %s...", user_id[:8] if user_id else "?")
        retriever, docs = await asyncio.to_thread(be.process_templates, file_objects, None, collection_name=collection_name, user_id=user_id, token=token)
        indexed_count = len(docs) if docs else 0
        logger.info("⏱️ Indexação: %.1fs (%d chunks, user %s)", _time.time()-t0, indexed_count, user_id[:8] if user_id else "?")

        # 2. Auto-generate style dossier in BACKGROUND (don't block response)
        # The dossier calls GPT-5.3 which adds 10-30s; run it async instead.
        # Make copies of the byte buffers so background task has its own data.
        dossier_buffers = []
        for f in file_objects:
            f.seek(0)
            buf_copy = io.BytesIO(f.read())
            buf_copy.name = getattr(f, 'name', 'template.pdf')
            buf_copy.seek(0)
            dossier_buffers.append(buf_copy)

        async def _gen_dossier_bg(fobjs):
            try:
                result = await asyncio.to_thread(be.generate_style_dossier, fobjs, None)
                print(f"✅ Dossiê de estilo gerado em background: {bool(result and not result.get('error'))}")
            except Exception as e:
                print(f"⚠️ Erro ao gerar dossiê em background: {e}")

        try:
            _bg_task = asyncio.create_task(_gen_dossier_bg(dossier_buffers))
            _bg_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        except RuntimeError as bg_err:
            # No running event loop in some deployment contexts — skip background task
            logger.warning("Background dossier task skipped: %s", bg_err)

        return {
            "indexed_chunks": indexed_count,
            "file_count": len(file_objects),
            "has_dossier": False,  # Will be true on next status check after background completes
            "dossier_preview": "",
            "cloning_prompt": "",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ upload_templates falhou: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erro ao indexar modelos: {type(e).__name__}: {str(e)}")


@app.get("/api/templates/status")
async def templates_status(request: Request = None):
    """Check how many templates are indexed."""
    try:
        user_id = _extract_user_id(request) or "default"
        token = _extract_token(request)
        if user_id not in be._template_store or not be._template_store.get(user_id):
            be._load_template_store(user_id, token)
        count = len(be._template_store.get(user_id, []))
        has_dossier = len(be._style_dossier_cache) > 0
        return {"indexed_chunks": count, "has_dossier": has_dossier}
    except Exception:
        return {"indexed_chunks": 0, "has_dossier": False}


@app.delete("/api/templates")
async def clear_templates(request: Request = None, _auth: dict = Depends(require_auth)):
    """Clear all indexed templates."""
    try:
        user_id = _extract_user_id(request) or "default"
        token = _extract_token(request)
        # Clear in-memory store for this user
        be._template_store[user_id] = []
        # Persist the empty state (clears Supabase or local JSON)
        be._save_template_store(user_id, token)
        # Also delete from Supabase directly
        be._delete_templates_supabase(user_id, "", token)
        # Clear style dossier cache
        be._style_dossier_cache.clear()
        return {"status": "ok", "message": "Modelos e cache limpos."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao limpar modelos: {str(e)}")


@app.get("/api/templates/list")
async def list_templates_endpoint(request: Request = None, _auth: dict = Depends(require_auth)):
    """List all indexed templates grouped by source filename."""
    try:
        user_id = _extract_user_id(request) or "default"
        token = _extract_token(request)
        templates = be.list_templates(user_id=user_id, token=token)
        return {"templates": templates}
    except Exception as e:
        traceback.print_exc()
        return {"templates": []}


@app.delete("/api/templates/{filename:path}")
async def delete_template_endpoint(filename: str, request: Request = None, _auth: dict = Depends(require_auth)):
    """Delete all chunks from a specific source file."""
    try:
        user_id = _extract_user_id(request) or "default"
        token = _extract_token(request)
        removed = be.delete_template_by_source(user_id=user_id, source_name=filename, token=token)
        return {"status": "ok", "removed_chunks": removed, "filename": filename}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao remover modelo: {str(e)}")


class AskTemplatesRequest(BaseModel):
    query: str


@app.post("/api/templates/ask")
async def ask_templates_endpoint(req: AskTemplatesRequest, request: Request = None, _auth: dict = Depends(require_auth)):
    """RAG query: search templates + generate LLM summary."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query é obrigatória.")
    try:
        user_id = _extract_user_id(request) or "default"
        token = _extract_token(request)
        # 1. Search templates using TF-IDF
        results = be.search_templates(user_id=user_id, query=req.query, k=5, token=token)
        if not results:
            return {"summary": "Nenhum trecho relevante encontrado nos modelos indexados.", "results": []}

        # 2. Build context for LLM summary
        context = "\n\n---\n\n".join([
            f"[Fonte: {r['source']}]\n{r['text']}"
            for r in results
        ])

        # 3. LLM summary
        summary_llm = be.get_llm("gpt-5.4-mini", temperature=0.3, max_tokens=800)
        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM
        summary_response = summary_llm.invoke([
            SM(content=(
                "Você é um analista jurídico. Com base nos trechos de modelos de decisão abaixo, "
                "responda à pergunta do usuário de forma objetiva e fundamentada. "
                "Cite os modelos-fonte quando relevante."
            )),
            HM(content=f"**Pergunta:** {req.query}\n\n**Trechos dos Modelos:**\n\n{context}"),
        ])
        summary = be.safe_content(summary_response)

        return {"summary": summary, "results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na consulta: {str(e)}")


@app.post("/api/templates/themes")
@app.post("/api/templates/extract-themes")
async def extract_themes_endpoint(request: Request = None, _auth: dict = Depends(require_auth)):
    """Extract legal themes from indexed templates for verification."""
    try:
        user_id = _extract_user_id(request) or "default"
        token = _extract_token(request)
        # Load templates
        if user_id not in be._template_store or not be._template_store[user_id]:
            be._load_template_store(user_id, token)
        user_store = be._template_store.get(user_id, [])
        if not user_store:
            return {"themes": []}

        # Build a sample from templates (first 6000 chars from each source)
        sources = {}
        for chunk in user_store:
            src = chunk.get("metadata", {}).get("source", "?")
            if src not in sources:
                sources[src] = ""
            if len(sources[src]) < 6000:
                sources[src] += chunk.get("text", "")[:3000] + "\n"

        sample = "\n\n---\n\n".join([
            f"[Modelo: {src}]\n{text[:6000]}"
            for src, text in list(sources.items())[:10]
        ])

        # LLM: extract themes
        llm = be.get_llm("gpt-5.4-mini", temperature=0.1, max_tokens=1000)
        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM
        response = llm.invoke([
            SM(content=(
                "Você é um classificador jurídico especializado em decisões judiciais cíveis. "
                "Analise os modelos de decisão abaixo e identifique os TEMAS JURÍDICOS distintos "
                "abordados neles.\n\n"
                "Para cada tema, forneça:\n"
                "- title: título conciso do tema (ex: 'Dano Moral por Negativação Indevida')\n"
                "- description: breve descrição (1 linha)\n\n"
                "Retorne APENAS um JSON array com objetos {\"id\": N, \"title\": \"...\", \"description\": \"...\"}.\n"
                "Identifique entre 5 e 15 temas. Não invente temas que não estão nos modelos."
            )),
            HM(content=sample),
        ])
        raw = be.safe_content(response).strip()

        # Parse JSON from response
        import re
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            themes = json.loads(json_match.group())
        else:
            themes = []

        return {"themes": themes}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao extrair temas: {str(e)}")


class VerifyThemeRequest(BaseModel):
    theme: str


@app.post("/api/templates/verify-theme")
async def verify_theme_endpoint(req: VerifyThemeRequest, request: Request = None, _auth: dict = Depends(require_auth)):
    """Verify a legal theme against TJMG jurisprudence."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Banco de jurisprudência não disponível.")
    if not req.theme.strip():
        raise HTTPException(status_code=400, detail="Tema é obrigatório.")

    try:
        user_id = _extract_user_id(request) or "default"
        token = _extract_token(request)

        # 1. Search jurisprudence for this theme
        search_result = jsearch.semantic_search(query=req.theme, page=1, page_size=5)
        results = search_result.get("results", [])

        # Fallback to keyword
        if not results:
            search_result = jsearch.search(query=req.theme, page=1, page_size=5)
            results = search_result.get("results", [])

        if not results:
            return {
                "status": "no_data",
                "theme": req.theme,
                "majority_understanding": None,
                "model_approach": None,
                "alert": None,
                "acordaos": [],
            }

        # 2. Get model approach from templates
        model_results = be.search_templates(user_id=user_id, query=req.theme, k=3, token=token)
        model_context = "\n".join([r["text"][:1500] for r in model_results]) if model_results else "Nenhum trecho relevante nos modelos."

        # 3. Build jurisprudence context
        juris_context = "\n\n---\n\n".join([
            f"[{r.get('tipo_recurso', 'Acórdão')}] {r.get('numero_processo', '?')} ({r.get('data_publicacao', '?')})\n"
            f"Ementa: {r.get('ementa', '')[:600]}"
            for r in results[:5]
        ])

        # 4. LLM comparison
        llm = be.get_llm("gpt-5.4-mini", temperature=0.2, max_tokens=1200)
        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM
        comparison_response = llm.invoke([
            SM(content=(
                "Você é um analista jurídico comparando modelos de decisão com a jurisprudência do TJMG.\n\n"
                "Analise o tema abaixo, compare o entendimento dos modelos do magistrado com os acórdãos do TJMG, "
                "e determine se estão ALINHADOS ou DIVERGENTES.\n\n"
                "Retorne um JSON com:\n"
                "{\n"
                '  "status": "aligned" ou "divergent",\n'
                '  "majority_understanding": "resumo do entendimento majoritário do TJMG (2-3 parágrafos)",\n'
                '  "model_approach": "como os modelos do magistrado tratam o tema (1-2 parágrafos)",\n'
                '  "comparison": "análise comparativa detalhada",\n'
                '  "alert": null ou {"title": "título do alerta", "detail": "explicação da divergência"}\n'
                "}\n\n"
                "IMPORTANTE: se os modelos estão alinhados com a jurisprudência, alert deve ser null."
            )),
            HM(content=(
                f"**Tema:** {req.theme}\n\n"
                f"**Acórdãos do TJMG:**\n{juris_context}\n\n"
                f"**Trechos dos Modelos do Magistrado:**\n{model_context}"
            )),
        ])
        raw = be.safe_content(comparison_response).strip()

        # Parse JSON
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {
                "status": "aligned",
                "majority_understanding": raw,
                "model_approach": "",
                "comparison": "",
                "alert": None,
            }

        # Add acordaos data for frontend
        result["theme"] = req.theme
        result["acordaos"] = [
            {
                "id": r.get("id"),
                "tipo_recurso": r.get("tipo_recurso", "Acórdão"),
                "numero_processo": r.get("numero_processo", "?"),
                "data_publicacao": r.get("data_publicacao", "?"),
                "comarca": r.get("comarca", ""),
                "relator": r.get("relator", ""),
                "ementa": r.get("ementa", "")[:300],
                "similarity": r.get("similarity", 0),
            }
            for r in results[:5]
        ]

        return result

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao verificar tema: {str(e)}")


# ── Admin: Upload Jurisprudência DB ───────────────────────────────────────────

@app.post("/api/admin/upload-jurisprudencia")
async def admin_upload_jurisprudencia(
    file: UploadFile = File(...),
    request: Request = None,
):
    """
    Upload do banco SQLite de jurisprudência (gzip compressed).
    Protegido por X-Admin-Key header.
    """
    # Auth check
    admin_key = os.getenv("ADMIN_KEY", "")
    if not admin_key:
        raise HTTPException(status_code=503, detail="ADMIN_KEY não configurada no servidor.")

    provided_key = ""
    if request:
        provided_key = request.headers.get("X-Admin-Key", "")
    if provided_key != admin_key:
        raise HTTPException(status_code=403, detail="Admin key inválida.")

    import gzip as _gzip

    try:
        # Determine target path
        db_path = os.environ.get(
            "JURISPRUDENCIA_DB_PATH",
            os.path.join(os.path.dirname(__file__), "data", "jurisprudencia.db"),
        )
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        # Read uploaded gzip file and decompress
        print(f"📥 Recebendo DB de jurisprudência...")
        content = await file.read()
        compressed_size = len(content)
        print(f"   📦 Recebido: {compressed_size / 1024 / 1024:.0f} MB comprimido")

        # Decompress
        print(f"   🔓 Descomprimindo para {db_path}...")
        decompressed = _gzip.decompress(content)
        with open(db_path, "wb") as f:
            f.write(decompressed)

        db_size = os.path.getsize(db_path)
        print(f"   💾 Salvo: {db_size / 1024 / 1024:.0f} MB")

        # Reload jurisprudence_search module connection
        global HAS_JURISPRUDENCIA, jsearch
        try:
            import jurisprudence_search as _jsearch
            _jsearch.reload_db()
            jsearch = _jsearch
            HAS_JURISPRUDENCIA = _jsearch.is_available()

            # Quick stats
            stats = _jsearch.get_stats()
            total = stats.get("total", 0)
            print(f"   ✅ DB recarregado: {total:,} acórdãos")

            return {
                "status": "ok",
                "total_acordaos": total,
                "db_size_mb": round(db_size / 1024 / 1024, 1),
                "compressed_size_mb": round(compressed_size / 1024 / 1024, 1),
            }
        except Exception as reload_err:
            print(f"   ⚠️ DB salvo mas erro ao recarregar: {reload_err}")
            return {
                "status": "ok",
                "warning": f"DB salvo ({db_size/1024/1024:.0f} MB) mas reload falhou: {str(reload_err)}",
                "db_size_mb": round(db_size / 1024 / 1024, 1),
            }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no upload: {str(e)}")


# ── Jurisprudência Research Agent (V0.5 background task) ──────────────────────

def _run_jurisprudence_research_background(task_id: str, uploaded_text: str, analysis_text: str):
    """
    Background worker: extracts legal themes from the process analysis,
    then queries the jurisprudence database for each theme.
    Uses a separate LLM context (does NOT pollute the main analysis context).
    """
    try:
        _bg_tasks[task_id]["status"] = "running"
        _bg_tasks[task_id]["progress"] = "Extraindo temas jurídicos do processo..."

        # 1. Extract key legal themes using a lightweight LLM call
        extraction_llm = be.get_llm(model_name="gpt-5.4-mini", temperature=0.1, max_tokens=500)
        extraction_prompt = (
            "Você é um assistente de pesquisa jurídica. "
            "Analise o texto processual abaixo e extraia EXATAMENTE 3 temas/questões jurídicas "
            "centrais que seriam úteis para pesquisar entendimento jurisprudencial recente.\n\n"
            "Retorne APENAS as 3 queries de busca, uma por linha, sem numeração ou prefixo. "
            "Cada query deve ser uma frase curta e precisa (máximo 15 palavras) "
            "que capture a essência jurídica do ponto controvertido.\n\n"
            "Exemplos de boas queries:\n"
            "- responsabilidade civil por inscrição indevida em cadastro de inadimplentes\n"
            "- dano moral in re ipsa negativação indevida\n"
            "- inversão do ônus da prova relação de consumo\n\n"
            "TEXTO DO PROCESSO (trecho):\n"
        )
        # Use first 3000 chars of uploaded text + first 2000 chars of analysis
        context_snippet = uploaded_text[:3000]
        if analysis_text:
            context_snippet += "\n\nANÁLISE DO ASSISTENTE:\n" + analysis_text[:2000]

        from langchain_core.messages import HumanMessage as HM, SystemMessage as SM
        extraction_response = extraction_llm.invoke([
            SM(content=extraction_prompt),
            HM(content=context_snippet),
        ])
        themes_text = be.safe_content(extraction_response).strip()
        themes = [line.strip().lstrip("- ").lstrip("0123456789.").strip()
                  for line in themes_text.split("\n")
                  if line.strip() and len(line.strip()) > 5][:3]

        if not themes:
            themes = ["questão jurídica do processo"]

        print(f"🔍 Agente Pesquisador: {len(themes)} temas extraídos: {themes}")

        _bg_tasks[task_id]["progress"] = f"Pesquisando jurisprudência para {len(themes)} temas..."

        # 2. Search jurisprudence for each theme
        import jurisprudence_search as jsearch
        research_results = []

        for i, theme in enumerate(themes):
            _bg_tasks[task_id]["progress"] = f"Pesquisando tema {i+1}/{len(themes)}: {theme[:50]}..."

            # Semantic search
            search_result = jsearch.semantic_search(
                query=theme, page=1, page_size=5,
            )
            results = search_result.get("results", [])

            # Fallback to keyword if semantic fails
            if not results:
                search_result = jsearch.search(query=theme, page=1, page_size=5)
                results = search_result.get("results", [])

            if not results:
                research_results.append({
                    "theme": theme,
                    "summary": f"Nenhum acórdão encontrado para: \"{theme}\"",
                    "results": [],
                    "total": 0,
                })
                continue

            # Build context for LLM summary
            context_parts = []
            for j, r in enumerate(results[:5], 1):
                sim_pct = round(r.get("similarity", 0) * 100)
                context_parts.append(
                    f"[{j}] {r.get('tipo_recurso', 'Acórdão')} | {r.get('comarca', '?')} | "
                    f"{r.get('data_publicacao', '?')} | Relevância: {sim_pct}%\n"
                    f"Processo: {r.get('numero_processo', '?')}\n"
                    f"Ementa: {r.get('ementa', '')[:600]}"
                )
            context = "\n\n---\n\n".join(context_parts)

            # LLM summary for this theme
            summary_llm = be.get_llm(model_name="gpt-5.4-mini", temperature=0.3, max_tokens=800)
            summary_prompt = (
                "Você é um assistente de pesquisa jurisprudencial do TJMG. "
                "Resuma em 3-5 parágrafos o entendimento predominante dos acórdãos abaixo "
                "sobre o tema pesquisado. Seja objetivo e cite os números dos processos. "
                "Destaque a TENDÊNCIA do tribunal (favorável ou desfavorável ao pedido) "
                "e valores de indenização quando aplicável. "
                "NÃO invente informações — baseie-se APENAS nas ementas."
            )
            summary_response = summary_llm.invoke([
                SM(content=summary_prompt),
                HM(content=f"**Tema:** {theme}\n\n**Acórdãos:**\n\n{context}"),
            ])
            summary_text = be.safe_content(summary_response)

            research_results.append({
                "theme": theme,
                "summary": summary_text,
                "results": results[:5],
                "total": search_result.get("total", len(results)),
            })

        # 3. Store results
        _bg_tasks[task_id]["status"] = "done"
        _bg_tasks[task_id]["result"] = {
            "themes": themes,
            "research": research_results,
            "total_themes": len(themes),
        }
        print(f"✅ Agente Pesquisador: pesquisa concluída ({len(research_results)} temas)")

    except Exception as e:
        traceback.print_exc()
        _bg_tasks[task_id]["status"] = "error"
        _bg_tasks[task_id]["error"] = f"Erro na pesquisa jurisprudencial: {str(e)}"


class JurisResearchTriggerRequest(BaseModel):
    uploaded_text: str
    analysis_text: str = ""


@app.post("/api/jurisprudencia/research")
async def trigger_jurisprudencia_research(req: JurisResearchTriggerRequest, _auth: dict = Depends(require_auth)):
    """Trigger background jurisprudence research for a process."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Banco de jurisprudência não disponível.")
    if not req.uploaded_text.strip():
        raise HTTPException(status_code=400, detail="Texto do processo é obrigatório.")

    task_id = str(uuid.uuid4())
    _bg_tasks[task_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "progress": "Iniciando pesquisa jurisprudencial...",
    }

    thread = threading.Thread(
        target=_run_jurisprudence_research_background,
        args=(task_id, req.uploaded_text, req.analysis_text),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "status": "pending"}


@app.get("/api/jurisprudencia/research/{task_id}")
async def jurisprudencia_research_status(task_id: str):
    """Poll the status of a background jurisprudence research task."""
    task = _bg_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Tarefa de pesquisa não encontrada.")

    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", ""),
    }

    if task["status"] == "done":
        response["result"] = task["result"]
        del _bg_tasks[task_id]
    elif task["status"] == "error":
        response["error"] = task.get("error", "Erro desconhecido")
        del _bg_tasks[task_id]

    return response


# ── Custom Agents CRUD (Supabase-backed) ──────────────────────────────────────

_CUSTOM_AGENTS_TABLE = "custom_agents"


class CreateAgentRequest(BaseModel):
    name: str
    prompt: str
    color: str = "#6366f1"


class ShareAgentRequest(BaseModel):
    email: str


@app.get("/api/custom-agents")
async def list_custom_agents(request: Request = None, _auth: dict = Depends(require_auth)):
    """List custom agents for the authenticated user."""
    # Extract user_id from JWT
    user_id = _extract_user_id(request)
    if not user_id:
        return {"agents": []}

    if _SUPA_URL and _SUPA_ANON_KEY:
        try:
            token = _extract_token(request)
            url = f"{_SUPA_URL.rstrip('/')}/rest/v1/{_CUSTOM_AGENTS_TABLE}?user_id=eq.{user_id}&select=*"
            resp = _requests_lib.get(url, headers=_supa_headers(token), timeout=10)
            if resp.ok:
                return {"agents": resp.json()}
        except Exception as e:
            logger.error("Failed to list custom agents: %s", e)

    return {"agents": []}


@app.post("/api/custom-agents")
async def create_custom_agent(req: CreateAgentRequest, request: Request = None, _auth: dict = Depends(require_auth)):
    """Create a custom agent."""
    user_id = _extract_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Autenticação necessária.")

    agent_id = f"custom_{uuid.uuid4().hex[:12]}"
    agent = {
        "id": agent_id,
        "user_id": user_id,
        "name": req.name,
        "prompt": req.prompt,
        "color": req.color,
    }

    if _SUPA_URL and _SUPA_ANON_KEY:
        try:
            token = _extract_token(request)
            url = f"{_SUPA_URL.rstrip('/')}/rest/v1/{_CUSTOM_AGENTS_TABLE}"
            resp = _requests_lib.post(url, json=agent, headers=_supa_headers(token), timeout=10)
            if resp.ok:
                created = resp.json()
                return created[0] if isinstance(created, list) else created
        except Exception as e:
            logger.error("Failed to create agent: %s", e)

    return agent


@app.delete("/api/custom-agents/{agent_id}")
async def delete_custom_agent(agent_id: str, request: Request = None, _auth: dict = Depends(require_auth)):
    """Delete a custom agent."""
    user_id = _extract_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Autenticação necessária.")

    if _SUPA_URL and _SUPA_ANON_KEY:
        try:
            token = _extract_token(request)
            url = f"{_SUPA_URL.rstrip('/')}/rest/v1/{_CUSTOM_AGENTS_TABLE}?id=eq.{agent_id}&user_id=eq.{user_id}"
            resp = _requests_lib.delete(url, headers=_supa_headers(token), timeout=10)
            if resp.ok:
                return {"status": "ok"}
        except Exception as e:
            logger.error("Failed to delete agent: %s", e)

    return {"status": "ok"}


@app.post("/api/custom-agents/{agent_id}/share")
async def share_custom_agent(agent_id: str, req: ShareAgentRequest, request: Request = None, _auth: dict = Depends(require_auth)):
    """Share a custom agent with another user by email."""
    # Placeholder — copies agent data for target user
    return {"status": "ok", "message": f"Compartilhamento de {agent_id} para {req.email} registrado."}


def _extract_user_id(request) -> str:
    """Extract user_id from JWT in Authorization header (with signature verification)."""
    if not request:
        return ""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return ""
    token = auth[7:]
    try:
        payload = verify_jwt(token)
        return payload.get("sub", "")
    except Exception:
        return ""


def _extract_token(request) -> str:
    """Extract raw JWT token from Authorization header."""
    if not request:
        return ""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""


# ── Jurisprudência Search ─────────────────────────────────────────────────────

try:
    import jurisprudence_search as jsearch
    HAS_JURISPRUDENCIA = jsearch.is_available()
    if HAS_JURISPRUDENCIA:
        logger.info("Banco de jurisprudência disponível.")
    else:
        logger.warning("Banco de jurisprudência não encontrado. Execute: python jurisprudence_indexer.py")
except ImportError:
    HAS_JURISPRUDENCIA = False
    jsearch = None
    logger.warning("Módulo jurisprudence_search não disponível.")


@app.get("/api/jurisprudencia/search")
async def jurisprudencia_search(
    q: str = "",
    ano_inicio: int = 0,
    ano_fim: int = 9999,
    tipo: str = "",
    page: int = 1,
    page_size: int = 20,
    mode: str = "semantic",
    _auth: dict = Depends(require_auth),
):
    """Full-text or semantic search across TJMG case law database.
    
    mode: 'keyword' (FTS5), 'semantic' (embedding similarity), 'hybrid' (both)
    Default is 'semantic' for AI-powered search.
    """
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(
            status_code=503,
            detail="Banco de jurisprudência não disponível. Execute: python jurisprudence_indexer.py",
        )
    if not q.strip():
        raise HTTPException(status_code=400, detail="Parâmetro de busca 'q' é obrigatório.")

    page_size = min(max(1, page_size), 50)  # Clamp 1-50

    if mode == "semantic":
        result = jsearch.semantic_search(
            query=q,
            ano_inicio=ano_inicio,
            ano_fim=ano_fim,
            tipo_recurso=tipo,
            page=max(1, page),
            page_size=page_size,
        )
        # If semantic fails (no key, no embeddings), fallback to keyword
        if result.get("error"):
            print(f"⚠️ Semantic search failed: {result['error']}. Falling back to keyword.")
            result = jsearch.search(
                query=q, ano_inicio=ano_inicio, ano_fim=ano_fim,
                tipo_recurso=tipo, page=max(1, page), page_size=page_size,
            )
            result["mode"] = "keyword_fallback"
    elif mode == "keyword":
        result = jsearch.search(
            query=q, ano_inicio=ano_inicio, ano_fim=ano_fim,
            tipo_recurso=tipo, page=max(1, page), page_size=page_size,
        )
        result["mode"] = "keyword"
    else:
        # Hybrid: run both and merge
        sem_result = jsearch.semantic_search(
            query=q, ano_inicio=ano_inicio, ano_fim=ano_fim,
            tipo_recurso=tipo, page=1, page_size=50,
        )
        kw_result = jsearch.search(
            query=q, ano_inicio=ano_inicio, ano_fim=ano_fim,
            tipo_recurso=tipo, page=1, page_size=50,
        )
        
        # Merge: use semantic results but boost those also found in keyword
        seen = {}
        for r in sem_result.get("results", []):
            seen[r["id"]] = r
        for r in kw_result.get("results", []):
            if r["id"] not in seen:
                r["similarity"] = 0.0
                seen[r["id"]] = r
            else:
                # Boost items found in both
                seen[r["id"]]["similarity"] = round(seen[r["id"]].get("similarity", 0) * 1.1, 4)
        
        merged = sorted(seen.values(), key=lambda x: x.get("similarity", 0), reverse=True)
        total = len(merged)
        offset = (max(1, page) - 1) * page_size
        
        result = {
            "results": merged[offset:offset + page_size],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": (total + page_size - 1) // page_size if total > 0 else 0,
            "query": q,
            "mode": "hybrid",
        }

    return result


# ── Jurisprudência Ask (RAG: semantic search + LLM summary) ───────────────────

class JurisprudenciaAskRequest(BaseModel):
    query: str
    ano_inicio: int = 0
    ano_fim: int = 9999
    tipo: str = ""

@app.post("/api/jurisprudencia/ask")
async def jurisprudencia_ask(req: JurisprudenciaAskRequest, _auth: dict = Depends(require_auth)):
    """
    RAG: busca semântica + resumo por LLM.
    Retorna um resumo inteligente dos acórdãos mais relevantes + os resultados brutos.
    """
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Banco de jurisprudência não disponível.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Parâmetro 'query' é obrigatório.")

    try:
        # 1. Semantic search — top 10 results
        search_result = jsearch.semantic_search(
            query=req.query,
            ano_inicio=req.ano_inicio,
            ano_fim=req.ano_fim,
            tipo_recurso=req.tipo,
            page=1,
            page_size=10,
        )

        results = search_result.get("results", [])
        if not results:
            # Fallback to keyword if semantic fails
            search_result = jsearch.search(
                query=req.query,
                ano_inicio=req.ano_inicio,
                ano_fim=req.ano_fim,
                tipo_recurso=req.tipo,
                page=1,
                page_size=10,
            )
            results = search_result.get("results", [])

        if not results:
            return {
                "summary": f"Não foram encontrados acórdãos relevantes para: \"{req.query}\". Tente reformular a consulta ou ajustar os filtros de ano.",
                "results": [],
                "total": 0,
                "query": req.query,
                "mode": search_result.get("mode", "unknown"),
            }

        # 2. Build context from top ementas
        context_parts = []
        for i, r in enumerate(results[:10], 1):
            sim_pct = round(r.get("similarity", 0) * 100)
            context_parts.append(
                f"[{i}] {r.get('tipo_recurso', 'Acórdão')} | {r.get('comarca', '?')} | "
                f"{r.get('data_publicacao', '?')} | Relevância: {sim_pct}%\n"
                f"Processo: {r.get('numero_processo', '?')}\n"
                f"Ementa: {r.get('ementa', '')[:800]}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 3. Enriched prompt for the LLM
        system_prompt = (
            "Você é um assistente jurídico especializado em jurisprudência do TJMG. "
            "O usuário fez uma pesquisa e você recebeu os acórdãos mais relevantes encontrados por busca semântica. "
            "Sua tarefa é:\n\n"
            "1. **Resumir o entendimento predominante** do TJMG sobre o tema pesquisado\n"
            "2. **Identificar tendências** (ex: o tribunal tende a conceder ou negar o pedido?)\n"
            "3. **Destacar os argumentos jurídicos recorrentes** nos acórdãos\n"
            "4. **Mencionar valores de indenização** quando aplicável\n"
            "5. **Citar os acórdãos mais relevantes** pelo número do processo\n\n"
            "Responda em linguagem clara e acessível, como se estivesse explicando para um advogado. "
            "Use markdown para formatar a resposta. "
            "NÃO invente informações — baseie-se APENAS nas ementas fornecidas."
        )

        user_prompt = (
            f"**Pesquisa do usuário:** {req.query}\n\n"
            f"**{len(results)} acórdãos mais relevantes encontrados:**\n\n"
            f"{context}"
        )

        # 4. Call LLM
        llm = be.get_llm(model_name="gpt-5.4-mini", temperature=0.3)
        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM
        response = llm.invoke([SM(content=system_prompt), HM(content=user_prompt)])
        summary = be.safe_content(response)

        return {
            "summary": summary,
            "results": results[:10],
            "total": search_result.get("total", len(results)),
            "query": req.query,
            "mode": search_result.get("mode", "semantic"),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na pesquisa inteligente: {str(e)}")


@app.get("/api/jurisprudencia/doc/{doc_id}")
async def jurisprudencia_doc(doc_id: int, _auth: dict = Depends(require_auth)):
    """Retrieve full text of a specific case law document."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Banco de jurisprudência não disponível.")

    doc = jsearch.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Acórdão não encontrado.")
    return doc


@app.get("/api/jurisprudencia/stats")
async def jurisprudencia_stats():
    """Get statistics about the case law database."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Banco de jurisprudência não disponível.")
    return jsearch.get_stats()


@app.get("/api/jurisprudencia/diagnostics")
async def jurisprudencia_diagnostics():
    """Diagnóstico: verifica tabelas e embeddings no banco."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Banco de jurisprudência não disponível.")

    import sqlite3 as _sqlite3

    db_path = os.environ.get(
        "JURISPRUDENCIA_DB_PATH",
        os.path.join(os.path.dirname(__file__), "data", "jurisprudencia.db"),
    )
    conn = _sqlite3.connect(db_path)
    c = conn.cursor()

    # List all tables
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]

    result = {"db_path": db_path, "tables": {}}
    for tname in table_names:
        try:
            count = c.execute(f"SELECT COUNT(*) FROM [{tname}]").fetchone()[0]
            result["tables"][tname] = {"count": count}
        except Exception as e:
            result["tables"][tname] = {"error": str(e)}

    # Check embeddings specifically
    if "embeddings" in table_names:
        try:
            sample = c.execute("SELECT length(embedding) FROM embeddings LIMIT 1").fetchone()
            if sample:
                result["embedding_dimensions"] = sample[0] // 4
                result["embedding_blob_bytes"] = sample[0]
        except Exception:
            pass

    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    result["db_size_mb"] = round(db_size / 1024 / 1024, 1)

    conn.close()
    return result


# ── SLM Engine Status ─────────────────────────────────────────────────────────

@app.get("/api/slm/status")
async def slm_status():
    """Status do SLM — verifica servidor remoto ou engine local."""
    # 1. Check remote SLM server (tunnel → MacBook)
    if HAS_REMOTE_SLM:
        remote_health = _check_remote_slm_health()
        return {
            "available": remote_health.get("status") == "online",
            "mode": "remote",
            "server_url": SLM_SERVER_URL,
            "remote": remote_health,
        }
    # 2. Check local SLM engine (dev on MacBook)
    if HAS_SLM:
        local_status = slm_engine.get_status()
        local_status["mode"] = "local"
        return local_status
    # 3. Not available
    return {"available": False, "mode": "none", "reason": "SLM não configurado (defina SLM_SERVER_URL ou rode no MacBook com MLX)"}


@app.post("/api/slm/unload")
async def slm_unload():
    """Descarrega modelos SLM da memória para liberar VRAM."""
    if not HAS_SLM:
        return {"status": "not_available"}
    slm_engine.unload()
    return {"status": "ok", "message": "Modelos descarregados da memória."}


# ── Sustentação Oral (Veredas) ───────────────────────────────────────────────
# Painel de apoio a sustentações orais: extrai dados estruturados do processo
# e oferece chat focado escopado ao texto extraído.

_sustentacao_store: dict[str, dict] = {}

_SUSTENTACAO_EXTRACT_SYSTEM = """Você é um assistente jurídico especializado em processos do TJMG.
Analise o texto do processo judicial fornecido e extraia as informações em JSON válido.
Retorne SOMENTE o JSON, sem markdown, sem explicações.

Esquema esperado:
{
  "numero_processo": "string ou null",
  "tipo_recursal": "string ou null",
  "camara": "string ou null",
  "relator": "string ou null",
  "data_sessao": "string ou null",
  "recorrente": "string ou null",
  "recorrido": "string ou null",
  "advogado_sustentante": "string ou null",
  "parte_sustentante": "string ou null",
  "teses": ["string"],
  "preliminares": ["string"],
  "sintese_decisao_1grau": "string ou null"
}

Se um campo não for encontrado, use null. Teses e preliminares devem ser listas de
strings curtas e objetivas (uma frase cada). Se não houver teses ou preliminares,
retorne lista vazia."""

_SUSTENTACAO_CHAT_SYSTEM_TEMPLATE = """Você é um assistente jurídico do desembargador, especializado no processo abaixo.
Responda às perguntas de forma objetiva, direta e técnica, com base exclusivamente
no texto do processo. Se a resposta não estiver no processo, diga isso claramente.

=== PROCESSO ===
{process_text}
=== FIM DO PROCESSO ==="""


class SustentacaoExtractRequest(BaseModel):
    text: str


class SustentacaoChatRequest(BaseModel):
    process_id: str
    messages: list[dict]


@app.post("/api/sustentacao/extract")
async def sustentacao_extract(req: SustentacaoExtractRequest, _auth: dict = Depends(require_auth)):
    """Extrai dados estruturados de um processo para apoio a sustentação oral."""
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto vazio.")

    try:
        llm = be.get_llm(model_name="gpt-5.3-chat", temperature=0.2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao inicializar LLM: {e}")

    truncated = text[:80_000]
    messages = [
        SystemMessage(content=_SUSTENTACAO_EXTRACT_SYSTEM),
        HumanMessage(content=f"Texto do processo:\n\n{truncated}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = be.safe_content(response).strip()
        # Remove cercas markdown caso o modelo escape
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"LLM retornou JSON inválido: {e}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na extração: {e}")

    process_id = str(uuid.uuid4())
    _sustentacao_store[process_id] = {"text": text, "data": data}
    # Limita o store a 50 processos (LRU simples)
    if len(_sustentacao_store) > 50:
        oldest = next(iter(_sustentacao_store))
        _sustentacao_store.pop(oldest, None)

    return {"process_id": process_id, "data": data}


@app.post("/api/sustentacao/chat")
async def sustentacao_chat(req: SustentacaoChatRequest, _auth: dict = Depends(require_auth)):
    """Chat focado em um processo de sustentação oral previamente extraído."""
    entry = _sustentacao_store.get(req.process_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Processo não encontrado. Faça o upload novamente.")
    if not req.messages:
        raise HTTPException(status_code=400, detail="Nenhuma mensagem fornecida.")

    try:
        llm = be.get_llm(model_name="gpt-5.3-chat", temperature=0.3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao inicializar LLM: {e}")

    system_content = _SUSTENTACAO_CHAT_SYSTEM_TEMPLATE.format(
        process_text=entry["text"][:80_000]
    )
    lc_messages = [SystemMessage(content=system_content)]
    for m in req.messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    try:
        response = llm.invoke(lc_messages)
        reply = be.safe_content(response)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no chat: {e}")

    return {"reply": reply}


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
