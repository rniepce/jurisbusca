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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

# ── In-memory conversation store (fallback) ──────────────────────────────
conversations_fallback: dict[str, list[dict]] = {}

# ── Auth Dependency ─────────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "COLOQUE_SEU_CLIENT_ID_AQUI.apps.googleusercontent.com")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validates the Google JWT and returns the user's email if valid, or None if not logged in."""
    if not credentials:
        return None
        
    token = credentials.credentials
    try:
        # id_token.verify_oauth2_token will verify signature and expiration
        id_info = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)
        return id_info.get("email")
    except ValueError as e:
        print(f"Token validation failed: {e}")
        return None


class ChatRequest(BaseModel):
    message: str
    model: str = "gemini"
    conversation_id: Optional[str] = None
    agent_prompt: Optional[str] = None
    ocr_engine: str = "paddle"
    uploaded_text: Optional[str] = None  # pre-extracted text from uploaded file
    style_dossier: Optional[str] = None  # cloning prompt from style report


# ── Model mapping ───────────────────────────────────────────────────────────
MODEL_MAP = {
    # Redireciona o antigo 'gemini' no frontend pro sonnet pra manter retrocompatibilidade
    "gemini": {"provider": "anthropic", "model": "claude-4-6-sonnet-20260220", "key_env": "ANTHROPIC_API_KEY"},
    "claude": {"provider": "anthropic", "model": "claude-4-6-sonnet-20260220", "key_env": "ANTHROPIC_API_KEY"},
    "gpt": {"provider": "openai", "model": "gpt-5.2", "key_env": "OPENAI_API_KEY"},
    "deepseek": {"provider": "openai", "model": "deepseek-reasoner", "key_env": "DEEPSEEK_API_KEY"},
    
    # NOVAS ENGINE VERSIONS DO GABINETE
    "v1": {"provider": "anthropic", "model": "claude-4-6-sonnet-20260220", "key_env": "ANTHROPIC_API_KEY"},
    "v2": {"provider": "anthropic", "model": "claude-4-6-sonnet-20260220", "key_env": "ANTHROPIC_API_KEY"},
    "v3": {"provider": "anthropic", "model": "claude-4-6-sonnet-20260220", "key_env": "ANTHROPIC_API_KEY"},
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
async def chat(req: ChatRequest, current_user: Optional[str] = Depends(get_current_user)):
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

        conv_id = req.conversation_id or str(uuid.uuid4())

        # Fetch history
        past_messages = []
        if current_user:
            past_messages = history_db.get_messages(conv_id, current_user)
            if past_messages is None:
                past_messages = []
        else:
            if conv_id not in conversations_fallback:
                conversations_fallback[conv_id] = []
            past_messages = conversations_fallback[conv_id]

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

        # ── Auto RAG: retrieve mirror context from persisted templates ──
        if req.uploaded_text:
            try:
                op_key = os.getenv("OPENAI_API_KEY", "")
                if op_key:
                    # Provide collection_name if logged in
                    collection_name = "rag_templates_persistent"
                    if current_user:
                        import hashlib
                        collection_name = f"templates_{hashlib.md5(current_user.encode()).hexdigest()[:10]}"
                        
                    rag_retriever = be.load_persistent_rag(op_key, collection_name=collection_name)
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
                print(f"⚠️ RAG auto-retrieval failed (non-blocking): {e}")

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

        # Persist user turn in history
        chat_title = "Nova Conversa"
        if len(past_messages) == 0 and user_content:
            chat_title = user_content[:50] + "..." if len(user_content) > 50 else user_content

        if current_user:
            history_db.save_message(conv_id, current_user, "user", user_content, title=chat_title)
        else:
            conversations_fallback[conv_id].append({"role": "user", "content": user_content})

        # Call LLM or specific Engine workflow
        if req.model == "v2":
            # Call Orchestrator V2 (Hybrid/Linear 3-Stage Workflow)
            if getattr(be, "run_hybrid_orchestration", None):
                keys = {
                    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
                    "google": os.getenv("GOOGLE_API_KEY"),
                    "openai": os.getenv("OPENAI_API_KEY"),
                    "deepseek": os.getenv("DEEPSEEK_API_KEY")
                }
                
                context_str = req.uploaded_text or ""
                user_msg = req.message or ""
                full_text = f"DADOS DO PROCESSO:\\n{context_str}\\n\\nPEDIDO:\\n{user_msg}"
                style_guide = req.style_dossier or ""
                
                v2_result = be.run_hybrid_orchestration(full_text, keys, style_guide)
                
                response_text = f"**Relatório V2 (Triagem):**\\n{v2_result.get('final_report', '')}\\n\\n**Minuta (Drafting):**\\n{v2_result.get('final_output', '')}\\n\\n**Auditoria (QA):**\\n{v2_result.get('audit_report', '')}"
            else:
                response_text = "Erro: V2 Engine (run_hybrid_orchestration) não importada ou indisponível"
                
        elif req.model == "v3":
            # Call Orchestrator V3 (Autonomous Magistrate)
            if getattr(be, "run_autonomous_magistrate", None):
                # Currently a placeholder - requires implementation logic for V3
                response_text = "⚠️ Modo V3 (Autônomo) acionado, mas o backend requer integração completa com o LangGraph."
            else:
                response_text = "Erro: V3 Engine não implementada."
        else:
            # Default V1 - just invoke LLM (Chat-based logic)
            response = llm.invoke(messages)
            response_text = be.safe_content(response)

        # Persist assistant turn in history
        if current_user:
            history_db.save_message(conv_id, current_user, "assistant", response_text, title=chat_title)
        else:
            conversations_fallback[conv_id].append({"role": "assistant", "content": response_text})

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


@app.get("/api/history")
async def get_history(current_user: Optional[str] = Depends(get_current_user)):
    """Fetch user's conversation history."""
    if not current_user:
        return {"conversations": []}
    
    try:
        conversations = history_db.get_conversations(current_user)
        # Format for Sidebar (Group by logical time later, just return list for now)
        return {"conversations": conversations}
    except Exception as e:
        print(f"Error fetching history: {e}")
        return {"conversations": []}


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    ocr_engine: str = Form("paddle"),
):
    """Upload and extract text from a file (PDF/DOCX/TXT)."""
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")

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
        # GPT-4o vision or text-embedding api keys
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY não configurada")

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
            "text_cache": text_cache,
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
        # Claude is used for style dossiers
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY não configurada")

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


# ── Cluster Batch Analysis (individual parallel processing) ───────────────────

class ClusterAnalyzeRequest(BaseModel):
    processes: list[dict]  # [{"filename": str, "text": str}]
    agent_prompt: Optional[str] = None
    model: str = "gemini"


def _analyze_single_process(filename: str, text: str, agent_prompt: str, model_cfg: dict, api_key: str, collection_name: str = "rag_templates_persistent"):
    """Analyze one process. Runs in a thread pool."""
    try:
        llm = be.get_llm(
            provider=model_cfg["provider"],
            model_name=model_cfg["model"],
            api_key=api_key,
            temperature=0.3,
        )

        messages = []
        system_parts = []

        if agent_prompt:
            system_parts.append(agent_prompt)

        system_parts.append(
            f"\n\n---\n📄 **DOCUMENTO ANEXADO (PEÇA PROCESSUAL):**\n\n{text}\n---"
        )

        # Auto-RAG: inject golden sample from persistent templates
        op_key = os.getenv("OPENAI_API_KEY", "")
        if op_key:
            try:
                rag_retriever = be.load_persistent_rag(op_key, collection_name=collection_name)
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
            "model": model_cfg["model"],
        }
    except Exception as e:
        return {
            "filename": filename,
            "status": "error",
            "response": f"Erro: {str(e)}",
            "model": model_cfg.get("model", "?"),
        }


@app.post("/api/cluster-analyze")
async def cluster_analyze(req: ClusterAnalyzeRequest, current_user: Optional[str] = Depends(get_current_user)):
    """
    Process all files in a cluster individually, in parallel.
    Returns a list of individual analysis results.
    """
    import concurrent.futures

    try:
        api_key = _get_api_key(req.model)
        cfg = MODEL_MAP.get(req.model, MODEL_MAP["gemini"])

        if not req.processes:
            raise HTTPException(status_code=400, detail="Nenhum processo para analisar.")

        collection_name = "rag_templates_persistent"
        if current_user:
            import hashlib
            collection_name = f"templates_{hashlib.md5(current_user.encode()).hexdigest()[:10]}"

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    _analyze_single_process,
                    p["filename"],
                    p["text"],
                    req.agent_prompt or "",
                    cfg,
                    api_key,
                    collection_name
                ): p["filename"]
                for p in req.processes
                if p.get("text")
            }
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
    current_user: str = Depends(get_current_user)
):
    """
    Upload and index template files in ChromaDB for persistent RAG.
    Also auto-generates the style dossier.
    Isolated per user.
    """
    import io

    if not current_user:
        raise HTTPException(status_code=401, detail="Precisa estar logado para salvar modelos.")

    try:
        # Templates and style dossiers require embeddings and generation
        op_key = os.getenv("OPENAI_API_KEY", "")
        anth_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        if not op_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY não configurada")
        if not anth_key:
            raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY não configurada")

        # Convert UploadFiles into file-like objects
        file_objects = []
        for f in files:
            content = await f.read()
            buf = io.BytesIO(content)
            buf.name = f.filename or "template.pdf"
            buf.seek(0)
            file_objects.append(buf)

        # 1. Index in ChromaDB (persistent, user isolated) -> Uses Embeddings (OpenAI)
        collection_name = f"templates_{hashlib.md5(current_user.encode()).hexdigest()[:10]}"
        retriever, docs = be.process_templates(file_objects, op_key, collection_name=collection_name)
        indexed_count = len(docs) if docs else 0

        # 2. Auto-generate style dossier (cached) -> Uses Core Text Generation (Anthropic)
        # Reset file positions for re-read
        for f in file_objects:
            f.seek(0)
        dossier_result = be.generate_style_dossier(file_objects, anth_key)
        has_dossier = bool(dossier_result and not dossier_result.get("error"))

        return {
            "indexed_chunks": indexed_count,
            "file_count": len(file_objects),
            "has_dossier": has_dossier,
            "dossier_preview": (dossier_result or {}).get("dossier", "")[:500],
            "cloning_prompt": (dossier_result or {}).get("cloning_prompt", ""),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao indexar modelos: {str(e)}")


@app.get("/api/templates/status")
async def templates_status(current_user: Optional[str] = Depends(get_current_user)):
    """Check how many templates are indexed in the persistent RAG for this user."""
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        count = 0
        if api_key and current_user:
            # We don't need to load the full retriever just to count, but check if db exists
            persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db_rag")
            if os.path.exists(persist_dir):
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=persist_dir)
                    collection_name = f"templates_{hashlib.md5(current_user.encode()).hexdigest()[:10]}"
                    collection = client.get_collection(collection_name)
                    count = collection.count()
                except Exception:
                    pass

        has_dossier = len(be._style_dossier_cache) > 0 # Note: globally cached right now

        return {
            "indexed_chunks": count,
            "has_dossier": has_dossier,
        }

    except Exception as e:
        return {"indexed_chunks": 0, "has_dossier": False}


@app.delete("/api/templates")
async def clear_templates(current_user: str = Depends(get_current_user)):
    """Clear all indexed templates from ChromaDB for the user."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Precisa estar logado.")
        
    try:
        persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db_rag")
        if os.path.exists(persist_dir):
            import chromadb
            client = chromadb.PersistentClient(path=persist_dir)
            try:
                collection_name = f"templates_{hashlib.md5(current_user.encode()).hexdigest()[:10]}"
                client.delete_collection(collection_name)
            except Exception:
                pass

        # Clear style dossier cache
        be._style_dossier_cache.clear()

        return {"status": "ok", "message": "Modelos e cache limpos."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao limpar modelos: {str(e)}")


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
