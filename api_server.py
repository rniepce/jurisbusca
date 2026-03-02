"""
FastAPI backend for Jurisbusca React frontend.
Exposes chat and file upload endpoints, wiring into existing backend.py orchestration.
"""
import os
import json
import uuid
import hashlib
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
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from jose import jwt

import backend as be
import history_db


@asynccontextmanager
async def lifespan(app):
    print("\n🚀 Registered routes:")
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        path = getattr(route, 'path', getattr(route, 'path_regex', '?'))
        print(f"   {methods or 'MOUNT'} {path}")
    print()
    yield

app = FastAPI(title="Jurisbusca API", version="1.0.0", lifespan=lifespan)

# (startup logging moved to lifespan context manager above)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory conversation store (fallback) ──────────────────────────────
conversations_fallback: dict[str, list[dict]] = {}

# ── In-memory process RAG cache (retriever per session) ──────────────────
# Key: hash of uploaded filename+size, Value: (retriever, full_text, char_count)

# ── In-memory background task store (shared: upload + xray) ───────────────
# Key: task_id, Value: {"status": str, "result": dict|None, "error": str|None, "progress": str}
_bg_tasks: dict[str, dict] = {}
_process_rag_cache: dict[str, tuple] = {}

# ── Auth (Supabase JWT Verification) ──────────────────────────────────────────

security = HTTPBearer()
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "") or os.getenv("VITE_SUPABASE_URL", "")

# Cache JWKS keys from Supabase
_jwks_cache: dict = {"keys": None, "fetched_at": 0}

def _get_jwks():
    """Fetch and cache JWKS from Supabase (refresh every 1 hour)."""
    import time, requests
    now = time.time()
    if _jwks_cache["keys"] and (now - _jwks_cache["fetched_at"]) < 3600:
        return _jwks_cache["keys"]
    
    if not SUPABASE_URL:
        return None
    
    try:
        jwks_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
        resp = requests.get(jwks_url, timeout=5)
        if resp.ok:
            jwks = resp.json()
            _jwks_cache["keys"] = jwks
            _jwks_cache["fetched_at"] = now
            print(f"✅ JWKS carregado: {len(jwks.get('keys', []))} chaves")
            return jwks
    except Exception as e:
        print(f"⚠️ Erro ao buscar JWKS: {e}")
    return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate Supabase JWT token and return user ID."""
    token = credentials.credentials
    if not SUPABASE_JWT_SECRET and not SUPABASE_URL:
        print("⚠️ AVISO: SUPABASE_JWT_SECRET e SUPABASE_URL não configurados. Ignorando auth.")
        return "development_user"
    
    # Strategy 1: Try JWKS verification (ES256 / ECC P-256 — new Supabase keys)
    jwks = _get_jwks()
    if jwks:
        try:
            # Get the signing key from JWKS
            from jose import jwk
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            
            matching_key = None
            for key_data in jwks.get("keys", []):
                if key_data.get("kid") == kid:
                    matching_key = key_data
                    break
            
            if matching_key:
                public_key = jwk.construct(matching_key)
                payload = jwt.decode(
                    token,
                    public_key,
                    algorithms=[matching_key.get("alg", "ES256")],
                    options={"verify_aud": False}
                )
                user_id = payload.get("sub")
                if not user_id:
                    raise HTTPException(status_code=401, detail="Token inválido (sem usuário)")
                return user_id
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Sessão expirada. Faça login novamente.")
        except HTTPException:
            raise
        except Exception as e:
            print(f"⚠️ JWKS verification failed, trying HS256 fallback: {e}")
    
    # Strategy 2: Fallback to HS256 with legacy JWT secret
    if SUPABASE_JWT_SECRET:
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False}
            )
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Token inválido (sem usuário)")
            return user_id
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Sessão expirada. Faça login novamente.")
        except jwt.JWTError as e:
            print(f"⚠️ HS256 JWT Error: {str(e)}")
            raise HTTPException(status_code=401, detail="Token de autenticação inválido.")
    
    raise HTTPException(status_code=401, detail="Não foi possível validar o token.")
        

class ChatRequest(BaseModel):
    message: str
    model: str = "v1"
    llm: Optional[str] = None  # LLM deployment name (e.g. 'gpt-5.2-chat', 'DeepSeek-V3.2-Speciale')
    conversation_id: Optional[str] = None
    agent_prompt: Optional[str] = None
    ocr_engine: str = "paddle"
    uploaded_text: Optional[str] = None  # pre-extracted text from uploaded file
    style_dossier: Optional[str] = None  # cloning prompt from style report
    use_rag: bool = False  # if True, use RAG retrieval instead of full text
    jurisprudence_context: Optional[str] = None  # V0.5: imported jurisprudence for minuta fundamentação


# ── Model mapping (Azure AI Foundry) ──────────────────────────────────────────
MODEL_MAP = {
    "gemini": "gpt-5.2-chat",
    "claude": "gpt-5.2-chat",
    "gpt":    "gpt-5.2-chat",
    "v0":     "gpt-5.2-chat",
    "v0.5":   "gpt-5.2-chat",
    "v1":     "gpt-5.2-chat",
    "v2":     "gpt-5.2-chat",
}


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


@app.post("/api/validate-key")
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
async def debug_routes():
    """Diagnostic: list all registered routes."""
    routes = []
    for route in app.routes:
        methods = getattr(route, 'methods', None)
        path = getattr(route, 'path', str(getattr(route, 'path_regex', '?')))
        routes.append({"methods": list(methods) if methods else ["MOUNT"], "path": str(path)})
    return {"routes": routes}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request, user_id: str = Depends(get_current_user)):
    """Process a chat message and return LLM response."""
    try:
        # Resolve LLM deployment: prefer explicit llm field, fallback to MODEL_MAP
        model_name = req.llm or MODEL_MAP.get(req.model, "gpt-5.2-chat")

        # Read Azure key from header (frontend sends it), fallback to env var
        azure_key = request.headers.get("X-Azure-Key", "").strip() or None

        # Build the LLM instance
        original_model = model_name
        try:
            llm = be.get_llm(model_name=model_name, temperature=0.3, api_key=azure_key)
            print(f"✅ LLM instanciado: {model_name}")
        except Exception as llm_err:
            print(f"⚠️ Falha ao instanciar {model_name}: {llm_err}. Fallback para gpt-5.2-chat.")
            model_name = "gpt-5.2-chat"
            llm = be.get_llm(model_name="gpt-5.2-chat", temperature=0.3, api_key=azure_key)

        conv_id = req.conversation_id or str(uuid.uuid4())

        # Fetch history (in-memory fallback, no auth)
        past_messages = []
        if conv_id not in conversations_fallback:
            conversations_fallback[conv_id] = []
        past_messages = conversations_fallback[conv_id]

        # ── Build LangChain message list with full history ──────────────
        messages = []

        # System prompt: agent instructions + document context
        system_parts = []
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
                        print(f"🎯 RAG Processo: {len(relevant_chunks)} chunks recuperados (~{total_tokens_approx} tokens vs full text)")
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
                print(f"⚠️ Process RAG falhou, usando texto completo: {e}")
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
                f"\n\n---\n📚 **JURISPRUDÊNCIA SELECIONADA PELO MAGISTRADO:**\n"
                f"O magistrado selecionou a(s) jurisprudência(s) abaixo para inclusão na fundamentação da minuta.\n"
                f"Insira na seção de FUNDAMENTAÇÃO, contextualizando o tema e transcrevendo a ementa com a citação do processo.\n\n"
                f"JURISPRUDÊNCIA SELECIONADA:\n"
                f"{req.jurisprudence_context}\n---"
            )

        # ── Phase-aware RAG: inject model context only when needed ──────
        # Phase 1 (Raio-X): No history → lightweight, just process text
        # Phase 3 (Minuta): User responded to deliberation → inject RAG + dossiê
        model_context_meta = {"mirror_used": False, "mirror_source": None, "match_quality": None, "dossier_used": False}
        
        if req.uploaded_text:
            has_history = len(past_messages) >= 2  # At least 1 assistant + 1 user response
            
            if has_history:
                # ── FASE 3: User has responded — inject RAG models + dossiê ──
                try:
                    rag_retriever = be.load_persistent_rag(collection_name="rag_templates_persistent", user_id=user_id)
                    if rag_retriever:
                        # Extract legal themes for smart query (instead of raw text)
                        themes_query = be.extract_legal_themes(req.uploaded_text)
                        
                        # Search with scores for quality-aware injection
                        scored_results = rag_retriever.invoke_scored(themes_query)
                        
                        if scored_results and scored_results[0][0] > 0.05:
                            top_score, mirror_doc = scored_results[0]
                            mirror_source = mirror_doc.metadata.get('source', '?')
                            
                            # Determine match quality
                            if top_score > 0.25:
                                quality = "alta"
                                quality_label = "GABARITO ESTRUTURAL E JURÍDICO"
                            elif top_score > 0.12:
                                quality = "media"
                                quality_label = "REFERÊNCIA ESTRUTURAL E DE ESTILO"
                            else:
                                quality = "baixa"
                                quality_label = "REFERÊNCIA DE ESTILO APENAS"
                            
                            model_context_meta = {
                                "mirror_used": True,
                                "mirror_source": mirror_source,
                                "match_quality": quality,
                                "match_score": top_score,
                                "dossier_used": False
                            }
                            
                            rag_block = (
                                f"\n\n---\n💎 **CASO ESPELHO ({quality_label} — Relevância: {quality.upper()}, Score: {top_score:.2f}):**\n"
                                f"⚠️ Modelo recuperado: `{mirror_source}`\n"
                            )
                            if quality in ("alta", "media"):
                                rag_block += (
                                    "1. Copie a macroestrutura (titulação, numeração, divisões).\n"
                                    "2. Clone o tom e vocabulário do magistrado.\n"
                                    "3. Se for o mesmo assunto, adapte apenas fatos e nomes, mantendo a fundamentação jurídica.\n"
                                )
                            else:
                                rag_block += (
                                    "⚠️ Tema diferente — copie APENAS estilo e estrutura, construa fundamentação nova.\n"
                                )
                            rag_block += f"\n--- INÍCIO DO CASO ESPELHO ---\n{mirror_doc.page_content}\n--- FIM DO CASO ESPELHO ---\n"
                            
                            # Add secondary models
                            for i, (sec_score, sec_doc) in enumerate(scored_results[1:3]):
                                if sec_score > 0.05:
                                    rag_block += f"\n[MODELO SECUNDÁRIO {i+2} — {sec_doc.metadata.get('source', '?')} (score: {sec_score:.2f})]:\n{sec_doc.page_content[:3000]}\n"
                            rag_block += "---"
                            system_parts.append(rag_block)
                            print(f"💎 Modelo espelho injetado: {mirror_source} (score: {top_score:.2f}, quality: {quality})")
                        else:
                            # No relevant model found — inject fallback template
                            system_parts.append(
                                f"\n\n---\n📝 **NENHUM MODELO DO MAGISTRADO ENCONTRADO NO ACERVO.**\n"
                                f"Use o template padrão abaixo como estrutura base:\n\n"
                                f"{be.TEMPLATE_SENTENCA_PADRAO}\n---"
                            )
                            print("📝 Nenhum modelo relevante — injetando template padrão")
                    else:
                        # No templates at all — inject fallback template
                        system_parts.append(
                            f"\n\n---\n📝 **NENHUM MODELO DISPONÍVEL NO ACERVO.**\n"
                            f"O magistrado não indexou modelos de decisão. Use o template padrão:\n\n"
                            f"{be.TEMPLATE_SENTENCA_PADRAO}\n---"
                        )
                except Exception as e:
                    print(f"⚠️ Phase 3 RAG retrieval failed (non-blocking): {e}")
                
                # ── Inject style dossier (from request, memory cache, or disk) ──
                if not req.style_dossier:
                    try:
                        dossier = be.load_style_dossier(user_id)
                        if dossier:
                            cloning = dossier.get('cloning_prompt', '')
                            glossary = dossier.get('glossary', '')
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
                                model_context_meta["dossier_used"] = True
                                print(f"🧬 Dossiê de estilo injetado (user {user_id[:8]})")
                    except Exception as e:
                        print(f"⚠️ Style dossier loading failed: {e}")
            else:
                # ── FASE 1: First interaction — no RAG needed ──
                print("🔍 Fase 1 (Raio-X): sem histórico, contexto leve (sem RAG)")

        # ── Claude rate-limit guard: truncate to stay under 30K input tokens ──
        # Anthropic Tier 1 allows 30K input tokens/min. ~4 chars ≈ 1 token.
        # We cap at 80K chars (~20K tokens) for the system prompt, leaving room
        # for conversation history, user message, and output tokens.
        if model_name.startswith("claude") and system_parts:
            CLAUDE_MAX_CHARS = 80_000
            total_chars = sum(len(p) for p in system_parts)
            if total_chars > CLAUDE_MAX_CHARS:
                print(f"✂️ Claude truncation: {total_chars} chars → ~{CLAUDE_MAX_CHARS} chars (rate limit guard)")
                # Truncate the largest part (usually uploaded_text at index 1+)
                # Strategy: keep agent prompt (first part) intact, truncate the rest proportionally
                budget = CLAUDE_MAX_CHARS
                truncated_parts = []
                for i, part in enumerate(system_parts):
                    if budget <= 0:
                        truncated_parts.append("\n\n⚠️ *[Conteúdo adicional omitido para respeitar o limite do Claude. Use GPT-5.2 para contextos maiores.]*")
                        break
                    if len(part) <= budget or i == 0:
                        # Keep agent prompt (i==0) and small parts intact
                        truncated_parts.append(part)
                        budget -= len(part)
                    else:
                        # Truncate this part to fit the remaining budget
                        truncated_parts.append(part[:budget] + "\n\n⚠️ *[Texto truncado para caber no limite do Claude (30K tokens/min). Use GPT-5.2 para o texto integral.]*")
                        budget = 0
                system_parts = truncated_parts

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
                    print(f"⚠️ V3 MoE pipeline timed out after {V3_TIMEOUT}s. Falling back to single-model GPT-5.2.")
                    # Fallback: single GPT-5.2 call with the full document
                    try:
                        fallback_llm = be.get_llm(model_name="gpt-5.2-chat")
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
                        response_text = f"{be.safe_content(fb_response)}\n\n---\n⚠️ *V3 MoE excedeu o tempo limite. Resultado gerado por GPT-5.2 (modelo único).*"
                    except Exception as fb_err:
                        response_text = f"⚠️ **V3 timeout ({V3_TIMEOUT}s) e fallback GPT-5.2 também falhou:** {str(fb_err)}"

                except Exception as e:
                    traceback.print_exc()
                    # Fallback: single GPT-5.2 call
                    try:
                        fallback_llm = be.get_llm(model_name="gpt-5.2-chat")
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
                        response_text = f"{be.safe_content(fb_response)}\n\n---\n⚠️ *V3 MoE falhou ({str(e)[:100]}). Resultado gerado por GPT-5.2 (modelo único).*"
                    except Exception as fb_err:
                        response_text = f"⚠️ **Erro no V3 Engine:** {str(e)}\n\n**Fallback GPT-5.2 também falhou:** {str(fb_err)}"
            else:
                response_text = "Erro: V3 Engine (run_autonomous_magistrate) não foi importada. Verifique se langgraph está instalado."
        else:
            # Default V0/V1 - just invoke LLM (Chat-based logic)
            try:
                response = llm.invoke(messages)
                response_text = be.safe_content(response)
            except Exception as invoke_err:
                error_str = str(invoke_err)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()

                # If model fails at invocation, retry with GPT-5.2
                if original_model != "gpt-5.2-chat":
                    if is_rate_limit:
                        print(f"⚠️ {original_model} rate limit (429). Retrying com GPT-5.2.")
                        reason = f"Modelo {original_model} excedeu o limite de taxa (rate limit). Tente novamente em alguns minutos ou use um prompt menor."
                    else:
                        print(f"⚠️ {original_model} invoke falhou: {invoke_err}. Retrying com GPT-5.2.")
                        reason = f"Modelo {original_model} indisponível ({error_str[:100]})."
                    try:
                        fallback_llm = be.get_llm(model_name="gpt-5.2-chat", temperature=0.3, api_key=azure_key)
                        response = fallback_llm.invoke(messages)
                        response_text = f"{be.safe_content(response)}\n\n---\n⚠️ *{reason} Resultado gerado por GPT-5.2.*"
                        model_name = "gpt-5.2-chat"
                    except Exception as fb_err:
                        response_text = f"⚠️ **Erro:** {reason} Fallback GPT-5.2 também falhou: {str(fb_err)[:150]}"
                else:
                    raise invoke_err

        # Persist assistant turn in history (in-memory)
        conversations_fallback[conv_id].append({"role": "assistant", "content": response_text})

        # Build response payload
        result_payload = {
            "conversation_id": conv_id,
            "response": response_text,
            "model": model_name,
            "model_context": model_context_meta,
        }
        
        # Include V2 structured sections if available
        if req.model == "v2" and 'v2_sections' in locals():
            result_payload["v2_sections"] = v2_sections

        # V0.5: jurisprudence research is now triggered manually via
        # POST /api/jurisprudencia/research (no longer auto-triggered here)

        return result_payload

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")


@app.get("/api/history")
async def get_history(user_id: str = Depends(get_current_user)):
    """Fetch conversation history."""
    return {"conversations": []}


def _run_upload_background(task_id: str, file_data: bytes, filename: str, ocr_engine: str, compress: bool):
    """Background worker: extracts text from uploaded file and stores result."""
    try:
        _bg_tasks[task_id]["status"] = "running"
        _bg_tasks[task_id]["progress"] = f"Extraindo texto de {filename}..."

        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        try:
            full_text, retriever = be.process_uploaded_file(
                open(tmp_path, "rb"),
                filename,
                ocr_engine_choice=ocr_engine,
                compress=compress,
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
            _process_rag_cache[cache_key] = (retriever, full_text, len(full_text))
            rag_available = True
            print(f"🎯 Process RAG cached: key={cache_key[:8]}")

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
        _bg_tasks[task_id]["error"] = f"Erro no upload: {str(e)}"


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    ocr_engine: str = Form("paddle"),
    compress: bool = Form(True),
    user_id: str = Depends(get_current_user)
):
    """Upload and extract text from a file. Returns task_id for polling."""
    try:
        content = await file.read()
        filename = file.filename or "documento.pdf"

        task_id = str(uuid.uuid4())
        _bg_tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "progress": f"Processando {filename}...",
        }

        thread = threading.Thread(
            target=_run_upload_background,
            args=(task_id, content, filename, ocr_engine, compress),
            daemon=True,
        )
        thread.start()

        return {"task_id": task_id, "status": "pending"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no upload: {str(e)}")


@app.get("/api/upload/{task_id}")
async def upload_status(task_id: str, user_id: str = Depends(get_current_user)):
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
async def batch_xray(
    files: list[UploadFile] = File(...),
    user_id: str = Depends(get_current_user)
):
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
        _bg_tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "progress": "Tarefa criada, aguardando início...",
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
async def xray_status(task_id: str, user_id: str = Depends(get_current_user)):
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
async def style_report(
    files: list[UploadFile] = File(...),
    user_id: str = Depends(get_current_user)
):
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


def _analyze_single_process(filename: str, text: str, agent_prompt: str, model_name: str, collection_name: str = "rag_templates_persistent", user_id: str = "default"):
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
                    rag_retriever = be.load_persistent_rag(collection_name=collection_name, user_id=user_id)
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
async def cluster_analyze(
    req: ClusterAnalyzeRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Process all files in a cluster individually, in parallel.
    Returns a list of individual analysis results.
    """
    import concurrent.futures

    try:
        # Resolve LLM deployment: prefer explicit llm field, fallback to MODEL_MAP
        model_name = req.llm or MODEL_MAP.get(req.model, "gpt-5.2-chat")

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
                    collection_name,
                    user_id
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


# ── Template Management (Persistent RAG) ─────────────────────────────────────

@app.post("/api/templates")
async def upload_templates(
    files: list[UploadFile] = File(...),
    user_id: str = Depends(get_current_user)
):
    """
    Upload and index template files in ChromaDB for persistent RAG.
    Also auto-generates the style dossier.
    """
    import io

    try:
        import time as _time
        import asyncio

        # Convert UploadFiles into file-like objects
        file_objects = []
        for f in files:
            content = await f.read()
            buf = io.BytesIO(content)
            buf.name = f.filename or "template.pdf"
            buf.seek(0)
            file_objects.append(buf)

        # 1. Index templates (100% local — no ChromaDB, no embeddings)
        t0 = _time.time()
        collection_name = "rag_templates_persistent"
        retriever, docs = be.process_templates(file_objects, None, collection_name=collection_name, user_id=user_id)
        indexed_count = len(docs) if docs else 0
        print(f"⏱️ Indexação: {_time.time()-t0:.1f}s ({indexed_count} chunks)")

        # 2. Auto-generate style dossier in BACKGROUND (don't block response)
        # The dossier calls GPT-5.2 which adds 10-30s; run it async instead.
        # Make copies of the byte buffers so background task has its own data.
        dossier_buffers = []
        for f in file_objects:
            f.seek(0)
            buf_copy = io.BytesIO(f.read())
            buf_copy.name = getattr(f, 'name', 'template.pdf')
            buf_copy.seek(0)
            dossier_buffers.append(buf_copy)

        async def _gen_dossier_bg(fobjs, uid):
            try:
                result = await asyncio.to_thread(be.generate_style_dossier, fobjs, None)
                if result and not result.get('error'):
                    # Persist dossier to disk per-user
                    be.save_style_dossier(uid, result)
                    print(f"✅ Dossiê de estilo gerado e salvo para user {uid[:8]}")
                else:
                    print(f"⚠️ Dossiê gerado com erro: {result.get('error', '?')}")
            except Exception as e:
                print(f"⚠️ Erro ao gerar dossiê em background: {e}")

        asyncio.create_task(_gen_dossier_bg(dossier_buffers, user_id))

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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao indexar modelos: {str(e)}")


@app.get("/api/templates/status")
async def templates_status(user_id: str = Depends(get_current_user)):
    """Check how many templates are indexed for the current user."""
    try:
        if user_id not in be._template_store or not be._template_store.get(user_id):
            be._load_template_store(user_id)
        count = len(be._template_store.get(user_id, []))
        has_dossier = len(be._style_dossier_cache) > 0
        return {"indexed_chunks": count, "has_dossier": has_dossier}
    except Exception:
        return {"indexed_chunks": 0, "has_dossier": False}


@app.delete("/api/templates")
async def clear_templates(user_id: str = Depends(get_current_user)):
    """Clear all indexed templates for the current user."""
    try:
        # Clear only this user's templates
        be._template_store.pop(user_id, None)
        # Remove persisted JSON for this user
        user_path = be._get_template_store_path(user_id)
        if os.path.exists(user_path):
            os.remove(user_path)
        # Clear style dossier cache
        be._style_dossier_cache.clear()
        return {"status": "ok", "message": "Modelos e cache limpos."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao limpar modelos: {str(e)}")


@app.get("/api/templates/list")
async def list_templates(user_id: str = Depends(get_current_user)):
    """List all template files for the current user with metadata."""
    try:
        templates = be.list_templates(user_id)
        return {"templates": templates, "total": len(templates)}
    except Exception as e:
        traceback.print_exc()
        return {"templates": [], "total": 0}


@app.delete("/api/templates/{filename:path}")
async def delete_single_template(filename: str, user_id: str = Depends(get_current_user)):
    """Delete a specific template file by source filename."""
    try:
        removed = be.delete_template_by_source(user_id, filename)
        if removed == 0:
            raise HTTPException(status_code=404, detail=f"Modelo '{filename}' não encontrado.")
        
        # Update remaining count
        remaining = be.list_templates(user_id)
        return {
            "status": "ok",
            "removed_chunks": removed,
            "remaining_templates": len(remaining),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao remover modelo: {str(e)}")


class TemplateAskRequest(BaseModel):
    query: str


@app.post("/api/templates/ask")
async def templates_ask(req: TemplateAskRequest, user_id: str = Depends(get_current_user)):
    """
    RAG: busca nos templates do usuário + resumo por LLM.
    Similar ao /api/jurisprudencia/ask mas para os modelos de decisão do usuário.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Parâmetro 'query' é obrigatório.")

    try:
        # 1. Search user's templates
        results = be.search_templates(user_id, req.query, k=5)
        if not results:
            return {
                "summary": f"Nenhum trecho relevante encontrado em seus modelos para: \"{req.query}\". Faça upload de modelos de decisão primeiro.",
                "results": [],
                "total": 0,
                "query": req.query,
            }

        # 2. Build context from relevant chunks
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[{i}] Fonte: {r['source']} | {r['full_length']} caracteres\n"
                f"Trecho: {r['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 3. LLM summarization
        system_prompt = (
            "Você é um assistente jurídico especializado em modelos de decisão. "
            "O usuário possui modelos de sentença/decisão indexados e fez uma busca. "
            "Você recebeu os trechos mais relevantes desses modelos.\n\n"
            "Sua tarefa é:\n"
            "1. **Resumir como os modelos tratam** o tema pesquisado\n"
            "2. **Identificar padrões de fundamentação** usados nos modelos\n"
            "3. **Destacar frases e estruturas** recorrentes que possam ser reaproveitadas\n"
            "4. **Citar os modelos** pelo nome do arquivo fonte\n\n"
            "Responda em linguagem clara. Use markdown para formatar. "
            "NÃO invente informações — baseie-se APENAS nos trechos fornecidos."
        )

        user_prompt = (
            f"**Pesquisa do usuário:** {req.query}\n\n"
            f"**{len(results)} trechos mais relevantes dos modelos:**\n\n"
            f"{context}"
        )

        llm = be.get_llm(model_name="gpt-4.1-mini", temperature=0.3)
        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM
        response = llm.invoke([SM(content=system_prompt), HM(content=user_prompt)])
        summary = be.safe_content(response)

        return {
            "summary": summary,
            "results": results,
            "total": len(results),
            "query": req.query,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na pesquisa nos modelos: {str(e)}")


# ── Template Jurisprudence Verification ───────────────────────────────────────

@app.post("/api/templates/extract-themes")
async def extract_themes(user_id: str = Depends(get_current_user)):
    """
    LLM reads user's templates and extracts the main legal themes.
    Returns a list of themes with titles and descriptions.
    """
    try:
        # Get user's template chunks
        if user_id not in be._template_store or not be._template_store.get(user_id):
            be._load_template_store(user_id)
        user_store = be._template_store.get(user_id, [])

        if not user_store:
            raise HTTPException(status_code=400, detail="Nenhum modelo indexado. Faça upload de modelos primeiro.")

        # Sample representative chunks (up to 15 chunks, spread across sources)
        import random
        sources = {}
        for chunk in user_store:
            src = chunk.get("metadata", {}).get("source", "?")
            if src not in sources:
                sources[src] = []
            sources[src].append(chunk)

        sampled = []
        for src, chunks in sources.items():
            n = min(3, len(chunks))
            sampled.extend(random.sample(chunks, n))
        if len(sampled) > 15:
            sampled = random.sample(sampled, 15)

        # Build context
        context_parts = []
        for i, chunk in enumerate(sampled, 1):
            source = chunk.get("metadata", {}).get("source", "?")
            text = chunk.get("text", "")[:2000]
            context_parts.append(f"[Trecho {i} — {source}]\n{text}")
        context = "\n\n---\n\n".join(context_parts)

        # LLM extraction
        system_prompt = (
            "Você é um analista jurídico sênior. Analise os trechos de modelos de decisão/sentença "
            "fornecidos e identifique os TEMAS JURÍDICOS CENTRAIS tratados neles.\n\n"
            "Extraia até 10 temas distintos. Para cada tema, retorne:\n"
            "- Um TÍTULO curto (máximo 8 palavras)\n"
            "- Uma DESCRIÇÃO de 1 frase explicando o tema\n\n"
            "Retorne APENAS um JSON array, sem markdown, sem explicações:\n"
            '[{"title": "...", "description": "..."}, ...]\n\n'
            "Exemplos de bons temas:\n"
            '- {"title": "Dano moral por negativação indevida", "description": "Responsabilidade civil por inscrição indevida em cadastros de inadimplentes."}\n'
            '- {"title": "Tutela de urgência", "description": "Concessão de tutela antecipada em casos de verossimilhança e perigo de dano."}\n'
        )

        llm = be.get_llm(model_name="gpt-4.1-mini", temperature=0.2, max_tokens=2000)
        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM
        response = llm.invoke([
            SM(content=system_prompt),
            HM(content=f"Analise estes trechos de modelos de decisão e extraia os temas:\n\n{context}"),
        ])
        raw = be.safe_content(response).strip()

        # Parse JSON
        import re
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            themes = json.loads(json_match.group())
        else:
            themes = json.loads(raw)

        # Add IDs
        for i, t in enumerate(themes):
            t["id"] = i + 1

        return {"themes": themes[:10], "total": len(themes[:10])}

    except HTTPException:
        raise
    except json.JSONDecodeError:
        return {"themes": [{"id": 1, "title": "Análise geral", "description": "Tema extraído dos modelos de decisão."}], "total": 1}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao extrair temas: {str(e)}")


class VerifyThemeRequest(BaseModel):
    theme: str


@app.post("/api/templates/verify-theme")
async def verify_theme(req: VerifyThemeRequest, user_id: str = Depends(get_current_user)):
    """
    For a given theme:
    1. Search user's templates for relevant excerpts
    2. Search jurisprudence DB for relevant case law
    3. GPT-5.2 compares both and classifies: aligned / divergent / no_data
    """
    if not req.theme.strip():
        raise HTTPException(status_code=400, detail="Tema não pode ser vazio.")

    try:
        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM

        # 1. Get relevant template excerpts
        template_results = be.search_templates(user_id, req.theme, k=3)
        template_context = ""
        if template_results:
            parts = []
            for i, r in enumerate(template_results, 1):
                parts.append(f"[Modelo {i} — {r['source']}]\n{r['text']}")
            template_context = "\n\n---\n\n".join(parts)

        # 2. Search jurisprudence
        juris_results = []
        juris_context = ""
        if HAS_JURISPRUDENCIA:
            try:
                search_result = jsearch.semantic_search(
                    query=req.theme, page=1, page_size=5,
                )
                juris_results = search_result.get("results", [])

                # Fallback to keyword if semantic returns nothing
                if not juris_results:
                    search_result = jsearch.search(
                        query=req.theme, page=1, page_size=5,
                    )
                    juris_results = search_result.get("results", [])

                if juris_results:
                    jp = []
                    for i, r in enumerate(juris_results[:5], 1):
                        sim = round(r.get("similarity", 0) * 100)
                        jp.append(
                            f"[Acórdão {i}] {r.get('tipo_recurso', 'Acórdão')} | "
                            f"Processo: {r.get('numero_processo', '?')} | "
                            f"{r.get('data_publicacao', '?')} | {r.get('comarca', '?')} | "
                            f"Relevância: {sim}%\n"
                            f"Ementa: {r.get('ementa', '')[:800]}"
                        )
                    juris_context = "\n\n---\n\n".join(jp)
            except Exception as e:
                print(f"⚠️ Jurisprudence search failed for theme '{req.theme}': {e}")

        # 3. If no data on either side
        if not template_context and not juris_context:
            return {
                "status": "no_data",
                "theme": req.theme,
                "summary": "Dados insuficientes para comparação. Verifique se há modelos indexados e se o banco de jurisprudência está disponível.",
                "model_approach": "",
                "majority_understanding": "",
                "comparison": "",
                "alert": None,
                "acordaos": [],
            }

        if not juris_context:
            return {
                "status": "no_data",
                "theme": req.theme,
                "summary": "Banco de jurisprudência não disponível ou sem acórdãos relevantes para este tema.",
                "model_approach": template_context[:500] if template_context else "",
                "majority_understanding": "",
                "comparison": "",
                "alert": None,
                "acordaos": [],
            }

        # 4. GPT-5.2 comparative analysis
        system_prompt = (
            "Você é um analista jurisprudencial sênior do TJMG. Sua tarefa é COMPARAR "
            "como os modelos de decisão do magistrado tratam um tema versus o entendimento "
            "MAJORITÁRIO da jurisprudência recente do TJMG.\n\n"
            "Você receberá:\n"
            "- TRECHOS DOS MODELOS DE DECISÃO do magistrado\n"
            "- ACÓRDÃOS RECENTES do TJMG sobre o mesmo tema\n\n"
            "Responda em JSON estrito (sem markdown, sem code fences):\n"
            "{\n"
            '  "status": "aligned" ou "divergent",\n'
            '  "majority_understanding": "Resumo CONCISO (máx 150 palavras) do entendimento predominante do TJMG",\n'
            '  "model_approach": "Resumo CONCISO (máx 100 palavras) de como os modelos tratam o tema",\n'
            '  "comparison": "Comparação CONCISA (máx 200 palavras). Se divergente, explique onde diverge citando processos. Se alinhado, os pontos de concordância.",\n'
            '  "alert_title": "Se divergent: título curto (1 frase). Se aligned: null",\n'
            '  "alert_detail": "Se divergent: explicação breve da divergência (máx 150 palavras). Se aligned: null"\n'
            "}\n\n"
            "REGRAS CRÍTICAS:\n"
            "- Baseie-se APENAS nos textos fornecidos\n"
            "- NÃO invente informações\n"
            "- Cite números de processo quando relevante\n"
            "- Seja CONCISO — respostas longas demais serão cortadas\n"
            "- Retorne o JSON COMPLETO e válido, sem truncar"
        )

        user_prompt = (
            f"**TEMA:** {req.theme}\n\n"
            f"## MODELOS DE DECISÃO DO MAGISTRADO:\n\n{template_context}\n\n"
            f"## ACÓRDÃOS RECENTES DO TJMG:\n\n{juris_context}"
        )

        llm = be.get_llm(model_name="gpt-5.2-chat", temperature=0.2, max_tokens=6000)
        response = llm.invoke([SM(content=system_prompt), HM(content=user_prompt)])
        raw = be.safe_content(response).strip()

        # Parse JSON response
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            analysis = json.loads(raw)

        # Build response with acórdãos metadata
        acordaos = []
        for r in juris_results[:5]:
            acordaos.append({
                "id": r.get("id"),
                "numero_processo": r.get("numero_processo", "?"),
                "tipo_recurso": r.get("tipo_recurso", "Acórdão"),
                "ementa": r.get("ementa", "")[:500],
                "data_publicacao": r.get("data_publicacao", "?"),
                "comarca": r.get("comarca", "?"),
                "relator": r.get("relator", ""),
                "similarity": r.get("similarity", 0),
            })

        return {
            "status": analysis.get("status", "no_data"),
            "theme": req.theme,
            "majority_understanding": analysis.get("majority_understanding", ""),
            "model_approach": analysis.get("model_approach", ""),
            "comparison": analysis.get("comparison", ""),
            "alert": {
                "title": analysis.get("alert_title"),
                "detail": analysis.get("alert_detail"),
            } if analysis.get("alert_title") else None,
            "acordaos": acordaos,
        }

    except json.JSONDecodeError:
        # LLM returned truncated/invalid JSON — try to salvage partial fields
        import re as _re
        def _extract(field):
            m = _re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"?', raw or "", _re.DOTALL)
            return m.group(1).replace('\\n', '\n').replace('\\"', '"') if m else ""

        status_m = _re.search(r'"status"\s*:\s*"(aligned|divergent)"', raw or "")
        salvaged_status = status_m.group(1) if status_m else "no_data"

        acordaos = []
        for r in juris_results[:5]:
            acordaos.append({
                "id": r.get("id"),
                "numero_processo": r.get("numero_processo", "?"),
                "tipo_recurso": r.get("tipo_recurso", "Acórdão"),
                "ementa": r.get("ementa", "")[:500],
                "data_publicacao": r.get("data_publicacao", "?"),
                "comarca": r.get("comarca", "?"),
                "relator": r.get("relator", ""),
                "similarity": r.get("similarity", 0),
            })

        alert_title = _extract("alert_title")
        return {
            "status": salvaged_status,
            "theme": req.theme,
            "majority_understanding": _extract("majority_understanding") or (raw[:1500] if raw else "Análise indisponível."),
            "model_approach": _extract("model_approach"),
            "comparison": _extract("comparison"),
            "alert": {"title": alert_title, "detail": _extract("alert_detail")} if alert_title else None,
            "acordaos": acordaos,
        }
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
        extraction_llm = be.get_llm(model_name="gpt-4.1-mini", temperature=0.1, max_tokens=500)
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
            summary_llm = be.get_llm(model_name="gpt-4.1-mini", temperature=0.3, max_tokens=800)
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


@app.get("/api/jurisprudencia/research/{task_id}")
async def jurisprudencia_research_status(task_id: str, user_id: str = Depends(get_current_user)):
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


class JurisResearchRequest(BaseModel):
    uploaded_text: str
    analysis_text: str = ""


@app.post("/api/jurisprudencia/research")
async def trigger_jurisprudencia_research(req: JurisResearchRequest, user_id: str = Depends(get_current_user)):
    """Manually trigger jurisprudence research for the given process text."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Módulo de jurisprudência não disponível.")

    task_id = str(uuid.uuid4())
    _bg_tasks[task_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "progress": "Extraindo temas jurídicos do processo...",
    }
    thread = threading.Thread(
        target=_run_jurisprudence_research_background,
        args=(task_id, req.uploaded_text, req.analysis_text),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "status": "pending"}


# ── Jurisprudência Search ─────────────────────────────────────────────────────

try:
    import jurisprudence_search as jsearch
    HAS_JURISPRUDENCIA = jsearch.is_available()
    if HAS_JURISPRUDENCIA:
        print("📚 Banco de jurisprudência disponível.")
    else:
        print("⚠️ Banco de jurisprudência não encontrado. Execute: python jurisprudence_indexer.py")
except ImportError:
    HAS_JURISPRUDENCIA = False
    jsearch = None
    print("⚠️ Módulo jurisprudence_search não disponível.")


@app.get("/api/jurisprudencia/search")
async def jurisprudencia_search(
    q: str = "",
    ano_inicio: int = 0,
    ano_fim: int = 9999,
    tipo: str = "",
    page: int = 1,
    page_size: int = 20,
    mode: str = "semantic",
    user_id: str = Depends(get_current_user)
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
async def jurisprudencia_ask(
    req: JurisprudenciaAskRequest,
    user_id: str = Depends(get_current_user)
):
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
        llm = be.get_llm(model_name="gpt-4.1-mini", temperature=0.3)
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
async def jurisprudencia_doc(doc_id: int, user_id: str = Depends(get_current_user)):
    """Retrieve full text of a specific case law document."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Banco de jurisprudência não disponível.")

    doc = jsearch.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Acórdão não encontrado.")
    return doc


@app.get("/api/jurisprudencia/stats")
async def jurisprudencia_stats(user_id: str = Depends(get_current_user)):
    """Get statistics about the case law database."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(status_code=503, detail="Banco de jurisprudência não disponível.")
    return jsearch.get_stats()


@app.get("/api/jurisprudencia/diagnostics")
async def jurisprudencia_diagnostics(user_id: str = Depends(get_current_user)):
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


# ── Custom Agents CRUD (Supabase REST API) ────────────────────────────────────

import requests as _requests

SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "") or os.getenv("VITE_SUPABASE_ANON_KEY", "")
# In-memory fallback if Supabase REST fails (e.g. table not created yet)
_custom_agents_fallback: dict[str, list[dict]] = {}  # user_id -> [agents]


def _supabase_headers(user_token: str = ""):
    """Build headers for Supabase REST API calls."""
    h = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    if user_token:
        h["Authorization"] = f"Bearer {user_token}"
    return h


def _get_user_token(request: Request) -> str:
    """Extract bearer token from request."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""


class CreateAgentRequest(BaseModel):
    name: str
    prompt: str
    color: str = "#8B5CF6"


class ShareAgentRequest(BaseModel):
    email: str


@app.post("/api/custom-agents")
async def create_custom_agent(req: CreateAgentRequest, request: Request, user_id: str = Depends(get_current_user)):
    """Create a custom agent for the current user."""
    agent_id = str(uuid.uuid4())
    agent = {
        "id": agent_id,
        "user_id": user_id,
        "name": req.name,
        "prompt": req.prompt,
        "color": req.color,
        "icon": "FaRobot",
    }

    # Try Supabase REST
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        try:
            token = _get_user_token(request)
            url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/custom_agents"
            payload = {**agent, "created_at": "now()"}
            resp = _requests.post(url, json=payload, headers=_supabase_headers(token), timeout=5)
            if resp.ok:
                data = resp.json()
                created = data[0] if isinstance(data, list) and data else agent
                return created
            else:
                print(f"⚠️ Supabase create agent failed ({resp.status_code}): {resp.text[:200]}. Using fallback.")
        except Exception as e:
            print(f"⚠️ Supabase create agent error: {e}. Using fallback.")

    # Fallback to in-memory
    if user_id not in _custom_agents_fallback:
        _custom_agents_fallback[user_id] = []
    _custom_agents_fallback[user_id].append(agent)
    return agent


@app.get("/api/custom-agents")
async def list_custom_agents(request: Request, user_id: str = Depends(get_current_user)):
    """List custom agents for the current user."""
    # Try Supabase REST
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        try:
            token = _get_user_token(request)
            url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/custom_agents?user_id=eq.{user_id}&order=created_at.desc"
            resp = _requests.get(url, headers=_supabase_headers(token), timeout=5)
            if resp.ok:
                agents = resp.json()
                return {"agents": agents}
            else:
                print(f"⚠️ Supabase list agents failed ({resp.status_code}): {resp.text[:200]}. Using fallback.")
        except Exception as e:
            print(f"⚠️ Supabase list agents error: {e}. Using fallback.")

    # Fallback
    agents = _custom_agents_fallback.get(user_id, [])
    return {"agents": agents}


@app.delete("/api/custom-agents/{agent_id}")
async def delete_custom_agent(agent_id: str, request: Request, user_id: str = Depends(get_current_user)):
    """Delete a custom agent (only owner)."""
    # Try Supabase REST
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        try:
            token = _get_user_token(request)
            url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/custom_agents?id=eq.{agent_id}&user_id=eq.{user_id}"
            resp = _requests.delete(url, headers=_supabase_headers(token), timeout=5)
            if resp.ok:
                return {"status": "deleted"}
            else:
                print(f"⚠️ Supabase delete agent failed ({resp.status_code}): {resp.text[:200]}. Using fallback.")
        except Exception as e:
            print(f"⚠️ Supabase delete agent error: {e}. Using fallback.")

    # Fallback
    if user_id in _custom_agents_fallback:
        _custom_agents_fallback[user_id] = [a for a in _custom_agents_fallback[user_id] if a["id"] != agent_id]
    return {"status": "deleted"}


@app.post("/api/custom-agents/{agent_id}/share")
async def share_custom_agent(agent_id: str, req: ShareAgentRequest, request: Request, user_id: str = Depends(get_current_user)):
    """Share a custom agent with another user by email."""
    # 1. Find the agent
    agent_data = None

    if SUPABASE_URL and SUPABASE_ANON_KEY:
        try:
            token = _get_user_token(request)
            url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/custom_agents?id=eq.{agent_id}&user_id=eq.{user_id}"
            resp = _requests.get(url, headers=_supabase_headers(token), timeout=5)
            if resp.ok:
                data = resp.json()
                if data:
                    agent_data = data[0]
        except Exception as e:
            print(f"⚠️ Supabase get agent error: {e}")

    # Fallback
    if not agent_data and user_id in _custom_agents_fallback:
        for a in _custom_agents_fallback[user_id]:
            if a["id"] == agent_id:
                agent_data = a
                break

    if not agent_data:
        raise HTTPException(status_code=404, detail="Agente não encontrado.")

    # 2. Find target user by email (Supabase admin API or fallback)
    target_user_id = None
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        try:
            # Use Supabase admin/service role to lookup user by email
            # Since we may not have service role key, we store the agent with email as target
            # For now, create the agent with a special marker
            new_agent = {
                "id": str(uuid.uuid4()),
                "user_id": req.email,  # Placeholder — will be resolved when user logs in
                "name": agent_data.get("name", "Agente Compartilhado"),
                "prompt": agent_data.get("prompt", ""),
                "color": agent_data.get("color", "#8B5CF6"),
                "icon": agent_data.get("icon", "FaRobot"),
                "shared_from": user_id,
            }
            url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/custom_agents"
            token = _get_user_token(request)
            resp = _requests.post(url, json=new_agent, headers=_supabase_headers(token), timeout=5)
            if resp.ok:
                return {"status": "shared", "target_email": req.email}
            else:
                print(f"⚠️ Supabase share failed ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            print(f"⚠️ Supabase share error: {e}")

    # Fallback: store in memory with email as key
    if req.email not in _custom_agents_fallback:
        _custom_agents_fallback[req.email] = []
    _custom_agents_fallback[req.email].append({
        "id": str(uuid.uuid4()),
        "user_id": req.email,
        "name": agent_data.get("name", "Agente Compartilhado"),
        "prompt": agent_data.get("prompt", ""),
        "color": agent_data.get("color", "#8B5CF6"),
        "icon": "FaRobot",
        "shared_from": user_id,
    })
    return {"status": "shared", "target_email": req.email}



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
