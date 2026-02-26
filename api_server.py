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

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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

# ── Auth (disabled — open access) ─────────────────────────────────────────


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


# ── Model mapping (Azure AI Foundry) ──────────────────────────────────────────
MODEL_MAP = {
    "gemini": "gpt-5.2-chat",
    "claude": "gpt-5.2-chat",
    "gpt":    "gpt-5.2-chat",
    "v0":     "gpt-5.2-chat",
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
async def chat(req: ChatRequest, request: Request):
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
                # If model fails at invocation, retry with GPT-5.2
                if original_model != "gpt-5.2-chat":
                    print(f"⚠️ {original_model} invoke falhou: {invoke_err}. Retrying com GPT-5.2.")
                    try:
                        fallback_llm = be.get_llm(model_name="gpt-5.2-chat", temperature=0.3, api_key=azure_key)
                        response = fallback_llm.invoke(messages)
                        response_text = f"{be.safe_content(response)}\n\n---\n⚠️ *Modelo {original_model} indisponível. Resultado gerado por GPT-5.2.*"
                        model_name = "gpt-5.2-chat"
                    except Exception as fb_err:
                        response_text = f"⚠️ **Erro:** {original_model} falhou ({str(invoke_err)[:150]}). Fallback GPT-5.2 também falhou: {str(fb_err)[:150]}"
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
        if req.model == "v2" and 'v2_sections' in locals():
            result_payload["v2_sections"] = v2_sections
        
        return result_payload

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")


@app.get("/api/history")
async def get_history():
    """Fetch conversation history (currently returns empty — no auth)."""
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
async def upload_status(task_id: str):
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
async def batch_xray(files: list[UploadFile] = File(...)):
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
async def xray_status(task_id: str):
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
async def style_report(files: list[UploadFile] = File(...)):
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
async def cluster_analyze(req: ClusterAnalyzeRequest):
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


# ── Template Management (Persistent RAG) ─────────────────────────────────────

@app.post("/api/templates")
async def upload_templates(
    files: list[UploadFile] = File(...),
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
        retriever, docs = be.process_templates(file_objects, None, collection_name=collection_name)
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

        async def _gen_dossier_bg(fobjs):
            try:
                result = await asyncio.to_thread(be.generate_style_dossier, fobjs, None)
                print(f"✅ Dossiê de estilo gerado em background: {bool(result and not result.get('error'))}")
            except Exception as e:
                print(f"⚠️ Erro ao gerar dossiê em background: {e}")

        asyncio.create_task(_gen_dossier_bg(dossier_buffers))

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
async def templates_status():
    """Check how many templates are indexed."""
    try:
        if not be._template_store:
            be._load_template_store()
        count = len(be._template_store)
        has_dossier = len(be._style_dossier_cache) > 0
        return {"indexed_chunks": count, "has_dossier": has_dossier}
    except Exception:
        return {"indexed_chunks": 0, "has_dossier": False}


@app.delete("/api/templates")
async def clear_templates():
    """Clear all indexed templates."""
    try:
        be._template_store.clear()
        # Remove persisted JSON
        if os.path.exists(be._TEMPLATE_STORE_PATH):
            os.remove(be._TEMPLATE_STORE_PATH)
        # Clear style dossier cache
        be._style_dossier_cache.clear()
        return {"status": "ok", "message": "Modelos e cache limpos."}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro ao limpar modelos: {str(e)}")


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
):
    """Full-text search across TJMG case law database."""
    if not HAS_JURISPRUDENCIA:
        raise HTTPException(
            status_code=503,
            detail="Banco de jurisprudência não disponível. Execute: python jurisprudence_indexer.py",
        )
    if not q.strip():
        raise HTTPException(status_code=400, detail="Parâmetro de busca 'q' é obrigatório.")

    page_size = min(max(1, page_size), 50)  # Clamp 1-50

    result = jsearch.search(
        query=q,
        ano_inicio=ano_inicio,
        ano_fim=ano_fim,
        tipo_recurso=tipo,
        page=max(1, page),
        page_size=page_size,
    )
    return result


@app.get("/api/jurisprudencia/doc/{doc_id}")
async def jurisprudencia_doc(doc_id: int):
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
