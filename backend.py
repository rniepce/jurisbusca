import os
import re
import traceback
import tempfile
import hashlib
import time
import random
import concurrent.futures
from typing import List, Optional, Any
import pypdf
import docx
from langchain_community.document_loaders import PyPDFLoader

# --- Imports Condicionais (podem falhar no Railway sem certas dependências) ---

# OCR Engine
HAS_OCR = False
try:
    import ocr_engine
    HAS_OCR = True
    print("✅ ocr_engine importado com sucesso.")
except ImportError as e:
    print(f"⚠️ ocr_engine não disponível (ImportError): {e}")
except Exception as e:
    import traceback
    print(f"⚠️ Erro ao importar ocr_engine: {e}")
    traceback.print_exc()

# Hybrid Chunker
HybridSemanticChunker = None
try:
    from chunking import HybridSemanticChunker
except ImportError:
    print("⚠️ HybridSemanticChunker não disponível.")
except Exception as e:
    print(f"⚠️ Erro ao importar HybridSemanticChunker: {e}")

# RAPTOR Engine
RaptorEngine = None
try:
    from raptor_engine import RaptorEngine
except ImportError:
    print("⚠️ RaptorEngine não disponível.")
except Exception as e:
    print(f"⚠️ Erro ao importar RaptorEngine: {e}")

# Planning Engine
PlanningEngine = None
try:
    from planning_engine import PlanningEngine
except ImportError:
    print("⚠️ PlanningEngine não disponível.")
except Exception as e:
    print(f"⚠️ Erro ao importar PlanningEngine: {e}")

# Style Engine
StyleEngine = None
try:
    from style_engine import StyleEngine
except ImportError:
    print("⚠️ StyleEngine não disponível.")
except Exception as e:
    print(f"⚠️ Erro ao importar StyleEngine: {e}")

# Agent Workflow
create_agent_workflow = None
try:
    from agent_workflow import create_agent_workflow
except ImportError:
    print("⚠️ create_agent_workflow não disponível.")
except Exception as e:
    print(f"⚠️ Erro ao importar create_agent_workflow: {e}")


from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings # Não usado (Railway usa Google Embeddings)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import ChatOllama # Removido para deploy Gemini Only
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Removido para deploy Gemini Only
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import gc

# Provider Integrations
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
    GEMINI_IMPORT_ERROR = None
except ImportError as e:
    HAS_GEMINI = False
    GEMINI_IMPORT_ERROR = str(e)
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from langchain_openai import AzureChatOpenAI
    HAS_AZURE_OPENAI = True
except ImportError:
    HAS_AZURE_OPENAI = False
    AzureChatOpenAI = None

# Legacy flag kept for compatibility checks elsewhere
HAS_ANTHROPIC = HAS_AZURE_OPENAI
# from prompts import PROMPT_FATOS, PROMPT_ANALISE_FORMAL, PROMPT_ANALISE_MATERIAL, PROMPT_RELATOR_FINAL
# (Re-enabling imports for V2 Ensemble)
from prompts import PROMPT_FATOS, PROMPT_ANALISE_FORMAL, PROMPT_JUIZ_DEEPSEEK, PROMPT_REDATOR_CLAUDE, PROMPT_AUDITOR_GPT
from prompts_claude import PROMPT_CLAUDE_INTEGRAL, PROMPT_GPT_AUDITOR, PROMPT_STYLE_ANALYZER, PROMPT_XRAY_BATCH, PROMPT_GPT_FIXER
# V1 Imports
# (Already imported above)

# V2 Imports (Agentic)
try:
    from v2_engine.orchestrator_v2 import run_hybrid_orchestration
    from v3_engine.orchestrator_v3 import run_autonomous_magistrate
except ImportError as e:
    # Se falhar (ex: falta langgraph), apenas V2 ficará indisponível
    print(f"Erro ao importar V2/V3 Engine: {e}")
    run_hybrid_orchestration = None
    run_autonomous_magistrate = None

# ── Module-level cache for style dossier (avoids re-running expensive analysis) ──
_style_dossier_cache = {}

def _template_cache_key(template_files):
    """Gera chave de cache baseada nos nomes dos arquivos de template."""
    import hashlib
    names = sorted([getattr(f, 'name', str(f)) for f in template_files])
    return hashlib.md5('|'.join(names).encode()).hexdigest()


def safe_content(response) -> str:
    """
    Normaliza response.content para string limpa.
    Lida com TODOS os formatos de retorno:
    - Anthropic: lista [{'type':'text','text':'...'}]
    - Gemini 2.5: lista [{'type':'thinking','thinking':'...'},{'type':'text','text':'...'}]
    - OpenAI: string pura
    - LangChain ChatGeneration: .content pode ser string ou lista
    - Casos edge: None, dict, int, etc.
    """
    # Extrair .content de objetos LangChain (AIMessage, ChatGeneration, etc.)
    if hasattr(response, 'content'):
        content = response.content
    elif hasattr(response, 'text'):
        content = response.text
    elif isinstance(response, dict):
        content = response.get('text', response.get('content', response.get('output', '')))
    else:
        content = response

    # None → string vazia
    if content is None:
        return ""

    # Se já é string, limpa e retorna
    if isinstance(content, str):
        content = content.replace("\\n", "\n")
        return content.strip()

    # Se é lista (Anthropic, Gemini 2.5 com thinking blocks)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                # Pula blocos de "thinking" do modelo (raciocínio interno)
                if item.get('type') == 'thinking':
                    continue
                if 'text' in item:
                    parts.append(str(item['text']))
                elif 'content' in item:
                    parts.append(str(item['content']))
                else:
                    parts.append(str(item))
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, 'text'):
                parts.append(str(item.text))
            elif hasattr(item, 'content'):
                parts.append(str(item.content))
            else:
                parts.append(str(item))
        content = "\n".join(parts)
        content = content.replace("\\n", "\n")
        return content.strip()

    # Se é dict (ex: JSON response acidental)
    if isinstance(content, dict):
        if 'text' in content:
            return str(content['text']).strip()
        if 'content' in content:
            return str(content['content']).strip()
        # Serializa como JSON legível
        import json as _json
        return _json.dumps(content, ensure_ascii=False, indent=2)

    # Fallback total: converte para string
    result = str(content)
    result = result.replace("\\n", "\n")
    return result.strip()


def clean_llm_text(text: str) -> str:
    """
    Sanitiza texto vindo de LLM para exibição limpa.
    Remove tags HTML, escaped newlines, artefatos de dict/JSON.
    """
    if not text or not isinstance(text, str):
        return ""
    # 1. Converte escaped newlines
    text = text.replace("\\n", "\n")
    # 2. Remove tags HTML comuns que LLMs podem gerar
    text = re.sub(r'<(?!/?(?:br|hr)\s*/?>)[^>]+>', '', text)
    # 3. Remove artefatos de dict Python vazando
    if "'extras':" in text:
        text = text.split("'extras':")[0].strip().rstrip(",").strip()
    elif '"extras":' in text:
        text = text.split('"extras":')[0].strip().rstrip(",").strip()
    # 4. Remove aspas soltas no início/fim
    text = text.strip().strip("'").strip('"')
    return text


def clean_text(text: str) -> str:
    """
    Higienização agressiva para peças jurídicas (Otimização de Context Window).
    Remove: Cabeçalhos, Rodapés, Números de Página, Espaços duplos, Assinaturas Digitais.
    """
    if not text or not isinstance(text, str):
        return ""
        
    # 1. Normalização de quebras de linha
    text = text.replace('\r', '')
    
    # 2. Remove cabeçalhos de numeração de processo (ex: "Processo nº 1234..." repetido)
    text = re.sub(r'(?i)(fls\.?\s*\d+|processo\s*nº?[:\s]*[\d\.\-]+)', '', text)
    
    # 3. Remove rodapés de escritório/sistema e assinaturas digitais
    # Padrão comum: "PJe - Assinado eletronicamente" ou "Documento assinado digitalmente"
    text = re.sub(r'(?i)(assinado\s+eletronicamente|documento\s+assinado|pje|assinatura\s+digital).*', '', text) 
    
    # 4. Remove números de página soltos
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # 5. Redução de ruído visual (traços, asteriscos)
    text = re.sub(r'[_=\-\*]{3,}', '', text)
    
    # 6. NOVO: Remove blocos de assinatura digital Base64 (longas sequências alfanuméricas)
    # Detecta strings com mais de 200 caracteres consecutivos sem espaços (típico de Base64/hash)
    text = re.sub(r'[A-Za-z0-9+/=]{200,}', '', text)
    
    # 7. NOVO: Remove chaves/colchetes JSON com conteúdo de 'signature' ou 'extras'
    text = re.sub(r"'extras':\s*\{[^}]*\}", '', text)
    text = re.sub(r"'signature':\s*'[^']*'", '', text)
    
    # 8. NOVO: Remove linhas que parecem ser metadados de certificado
    text = re.sub(r'(?i)(certificado|hash|sha\d*|md5|rsa|dsa|asn\.\d):[^\n]*\n?', '', text)
    
    # 9. Compressão de espaços (White space normalization) - PRESERVANDO quebras de linha
    # Substitui múltiplos espaços horizontais por um único
    text = re.sub(r'[ \t]+', ' ', text)
    # Limita múltiplas quebras de linha a no máximo duas
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def get_embedding_function(api_key=None):
    """
    Factory centralizada de embeddings — usa Azure OpenAI.
    api_key: se fornecida, tem prioridade sobre a variável de ambiente.
    """
    from langchain_openai import AzureOpenAIEmbeddings
    
    azure_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    
    if not azure_key or not azure_endpoint:
        raise ValueError(
            "AZURE_OPENAI_API_KEY e AZURE_OPENAI_EMBEDDING_ENDPOINT devem estar configurados "
            "no .env ou variáveis de ambiente para usar embeddings."
        )
    
    return AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        api_version="2024-12-01-preview",
    )

def process_uploaded_file(file_obj, filename: str, api_key=None, ocr_engine_choice="gpt4o_mini", compress=True):
    """
    Salva arquivo temp, faz OCR se necessário, vetoriza e retorna (full_text, retriever).
    """
    text = ""
    docs = []
    
    # Cria arquivo temporário para processamento (necessário para loaders do Langchain)
    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_obj.read())
        tmp_path = tmp_file.name

    try:
        if suffix == ".pdf":
            # ── Hybrid Extract: page-level triage (text vs OCR) ──
            try:
                from core.hybrid_extract import hybrid_extract
                ocr_choice = ocr_engine_choice
                # Normalize OCR engine choice
                if ocr_choice in ["claude_vision", "gpt4o_mini"]:
                    ocr_choice = "paddle"
                docs, stats = hybrid_extract(tmp_path, ocr_choice, compress)
                print(f"📊 {stats['text_pages']} págs texto | {stats['ocr_pages']} págs OCR | {stats['total_chars']} chars | {stats['elapsed_seconds']}s")
            except ImportError:
                print("⚠️ core.hybrid_extract não disponível. Usando extração legada.")
                # ── Fallback: extração legada (PyPDFLoader) ──
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                total_chars = sum(len(d.page_content) for d in docs)
                if total_chars < 500:
                    print(f"📉 Texto insuficiente ({total_chars} chars). Acionando OCR ({ocr_engine_choice})...")
                    if HAS_OCR:
                        ocr_text = ocr_engine.extract_text_from_pdf(tmp_path, engine="paddle")
                        if ocr_text and "[ERRO]" not in ocr_text:
                            from langchain_core.documents import Document
                            docs = [Document(page_content=ocr_text, metadata={"source": filename, "ocr": "paddle"})]
        
        elif suffix == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            
        elif suffix == ".txt":
            from langchain_community.document_loaders import TextLoader
            try:
                loader = TextLoader(tmp_path, encoding='utf-8')
                docs = loader.load()
            except Exception:
                loader = TextLoader(tmp_path, encoding='latin-1')
                docs = loader.load()
            
        else:
            return f"Formato não suportado: {filename}", None

        # Limpeza e Consolidação
        full_text = ""
        for doc in docs:
            cleaned = clean_text(doc.page_content)
            doc.page_content = cleaned
            full_text += cleaned + "\n\n"
            
        print(f"Texto extraído: {len(full_text)} caracteres.")

        # Vetorização (RAG)
        # Divide em chunks
        if not docs:
             return "Nenhum texto extraído.", None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        # Cria Vector Store em memória (ephemeral)
        # Usando Chroma com timeout para evitar hang
        try:
            embedding_function = get_embedding_function(api_key=api_key)
            
            # Processa em batches menores para evitar rate limit e timeout
            print(f"📊 Vetorizando {len(splits)} chunks...")
            
            import signal
            
            # Timeout para a vetorização inteira (5 min max)
            RAG_TIMEOUT = 300
            
            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Vetorização excedeu {RAG_TIMEOUT}s")
            
            # Configura timeout (apenas em Unix/Mac)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(RAG_TIMEOUT)
            
            try:
                vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embedding_function,
                    collection_name="temp_process_analysis"
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            finally:
                signal.alarm(0)  # Cancela o alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restaura handler
            
            print(f"✅ RAG indexado: {len(splits)} chunks vetorizados.")
            return full_text, retriever
            
        except TimeoutError as e:
            print(f"⚠️ RAG Timeout: {e}. Retornando texto sem vetorização.")
            return full_text, None
        except Exception as e:
            print(f"⚠️ Erro na vetorização RAG: {e}. Retornando texto sem retriever.")
            return full_text, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Erro ao processar arquivo: {str(e)}", None
    finally:
        # Limpa arquivo temporário
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def get_llm(model_name: str = "gpt-5.2-chat", temperature: float = 0.2, api_key: str = None, **kwargs):
    """
    Factory centralizada — todos os modelos passam pelo Azure OpenAI.
    model_name: nome do deployment no Azure (ex: 'gpt-5.2-chat', 'gpt-4.1-mini', 'DeepSeek-V3.2-Speciale').
    api_key: se fornecida, tem prioridade sobre a variável de ambiente.
    """
    deployment = model_name or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")

    # ── Serverless models (Azure AI / Models-as-a-Service) ──
    # These use the standard OpenAI SDK with a different base URL
    SERVERLESS_MODELS = {"DeepSeek-V3.2-Speciale", "Kimi-K2.5"}

    if deployment in SERVERLESS_MODELS:
        if not HAS_OPENAI:
            raise ImportError("langchain-openai não instalado. Execute: pip install langchain-openai")

        azure_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        serverless_endpoint = os.getenv(
            "AZURE_AI_SERVERLESS_ENDPOINT",
            "https://assistente-web-resource.services.ai.azure.com/openai/v1/"
        )

        if not azure_key:
            raise ValueError("AZURE_OPENAI_API_KEY deve estar configurada para usar modelos serverless.")

        # Azure OpenAI uses max_completion_tokens instead of max_tokens
        if 'max_tokens' in kwargs:
            kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')

        llm_kwargs = dict(
            model=deployment,
            base_url=serverless_endpoint,
            api_key=azure_key,
            temperature=temperature,
            **kwargs,
        )
        return ChatOpenAI(**llm_kwargs)

    # ── Standard Azure OpenAI models ──
    if not HAS_AZURE_OPENAI:
        raise ImportError("langchain-openai não instalado. Execute: pip install langchain-openai")

    azure_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")

    if not azure_key or not azure_endpoint:
        raise ValueError(
            "AZURE_OPENAI_API_KEY e AZURE_OPENAI_ENDPOINT devem estar configurados no .env ou variáveis de ambiente."
        )

    # Azure OpenAI uses max_completion_tokens instead of max_tokens
    if 'max_tokens' in kwargs:
        kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')

    # GPT-5.2 doesn't support custom temperature — only default (1)
    # For models that don't support it, we omit the parameter entirely
    models_no_temp = {"gpt-5.2-chat"}
    use_temperature = temperature if deployment not in models_no_temp else None

    llm_kwargs = dict(
        azure_deployment=deployment,
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        api_version="2024-12-01-preview",
        **kwargs,
    )
    if use_temperature is not None:
        llm_kwargs["temperature"] = use_temperature

    return AzureChatOpenAI(**llm_kwargs)

def run_reflexion_loop(draft_text, source_text, api_key):
    """
    ACTIVE AUDITOR (REFLEXION LOOP):
    1. Auditor: Critica a minuta (busca alucinações).
    2. Fixer: Se houver erro, reescreve a minuta e devolve.
    """
    try:
        # Usa Claude Sonnet via Azure para auditoria
        auditor_llm = get_llm("gpt-5.2-chat", temperature=0.0)
        
        # 1. Auditoria
        # Precisamos parsear o draft. Se for JSON, extraímos a 'minuta_final'.
        # Se for string (fallback), usamos ela mesma.
        draft_content = draft_text
        if isinstance(draft_text, str) and draft_text.strip().startswith("{"):
            try:
                import json
                # Tenta limpar wrappers
                clean = draft_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean)
                if isinstance(data, dict):
                    draft_content = data.get("minuta_final", draft_text)
            except Exception:
                pass

        print("🛡️ Iniciando Auditoria Ativa (Reflexion Loop)...")
        msg_audit = [
            SystemMessage(content=PROMPT_GPT_AUDITOR),
            HumanMessage(content=f"DADOS DO PROCESSO:\n{source_text[:50000]}\n\nMINUTA PARA AUDITORIA:\n{draft_content}")
        ]
        
        audit_resp = safe_content(auditor_llm.invoke(msg_audit))
        audit_clean = audit_resp.replace("```json", "").replace("```", "").strip()
        
        audit_json = {}
        try:
            audit_json = json.loads(audit_clean)
        except Exception:
            print(f"Erro parse auditoria: {audit_clean}")
            return draft_text, "Falha no Parse da Auditoria"

        # 2. Decisão: Aprova ou Corrige?
        if audit_json.get("aprovado") is True:
            print("✅ Auditoria Aprovada (Sem Alucinações).")
            return draft_text, audit_resp # Retorna original
            
        else:
            errors = audit_json.get("erros_criticos", [])
            print(f"❌ Auditoria Reprovou. Erros: {errors}. Iniciando Auto-Correção...")
            
            # 3. Fixer (Usa o mesmo modelo)
            fixer_llm = get_llm("gpt-5.2-chat", temperature=0.1)
            
            msg_fix = PROMPT_GPT_FIXER.format(
                draft=draft_content,
                critique=json.dumps(errors, ensure_ascii=False)
            )
            
            fix_resp = safe_content(fixer_llm.invoke([HumanMessage(content=msg_fix)]))
            
            # Se o input era JSON, precisamos reconstruir o JSON com a minuta corrigida?
            # Sim, para mater compatibilidade com o frontend que espera JSON.
            if isinstance(draft_text, str) and draft_text.strip().startswith("{"):
                try:
                    clean = draft_text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(clean)
                    if isinstance(data, dict):
                        data["minuta_final"] = fix_resp
                        data["diagnostico"]["status_auditoria"] = "Corrigido Automaticamente"
                        return json.dumps(data, ensure_ascii=False), audit_resp
                except Exception:
                    pass
            
            return fix_resp, audit_resp

    except Exception as e:
        print(f"Erro no Reflexion Loop: {e}")
        return draft_text, str(e)

def extract_text_with_gpt4o_mini(file_path, api_key):
    """
    SEMANTIC OCR (Vision API via GPT-4o-mini).
    Lê o PDF renderizando páginas como imagem e extrai texto limpo e barato.
    """
    if not HAS_OPENAI:
        return "Erro: Biblioteca langchain-openai não encontrada."

    try:
        t_start = time.time()
        import fitz
        import base64
        
        doc = fitz.open(file_path)
        base64_images = []
        
        # Limita a 25 páginas para não estourar o limite da API (4o-mini)
        for i, page in enumerate(doc):
            if i >= 25:
                print("⚠️ OCR GPT-4o-mini truncado em 25 páginas para evitar limite de Tokens.")
                break
                
            # Renderiza página com qualidade média-alta para leitura
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_bytes = pix.tobytes("jpeg")
            encoded = base64.b64encode(img_bytes).decode("utf-8")
            base64_images.append(encoded)

        # Inicializa o modelo GPT-4o-mini (Vision)
        from langchain_core.messages import HumanMessage
        
        llm = get_llm("gpt-5.2-chat", temperature=0.1)
        
        prompt_text = """
        Aja como um transcritor jurídico de elite. 
        Você está recebendo imagens de páginas de um processo.
        Extraia o texto integral deste documento nas imagens, preservando a formatação e tabelas. 
        
        ⚠️ REGRAS DE LIMPEZA JURÍDICA:
        1. Ignore cabeçalhos repetitivos de paginação.
        2. Ignore rodapés (ex: "PJe - Assinado eletronicamente").
        3. Ignore Carimbos, QR Codes, Assinaturas (hash).
        
        Retorne APENAS o texto limpo, linear e perfeitamente estruturado das páginas."
        """
        
        # Monta a estrutura da mensagem multimodal
        content_parts = [{"type": "text", "text": prompt_text}]
        for b64 in base64_images:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}
            })
            
        print(f"📤 Enviando {len(base64_images)} imagens de alta qualidade para rotina GPT-4o-mini OCR...")
        t_gen = time.time()
        
        msg = HumanMessage(content=content_parts)
        response = llm.invoke([msg])
        
        print(f"⏱️ OCR GPT-4o-mini concluído em {time.time() - t_gen:.1f}s")
        return safe_content(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Erro no Semantic OCR GPT-4o-mini: {str(e)}"

def _extract_template_texts(template_files):
    """
    Lightweight text extraction from template files (PDF/DOCX/TXT).
    Returns a list of Document objects with full text — NO chunking, NO ChromaDB.
    Used by generate_style_dossier which only needs raw text for LLM analysis.
    """
    from langchain_core.documents import Document
    documents = []
    
    for file in template_files:
        if hasattr(file, 'seek'):
            file.seek(0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            text = ""
            if file.name.endswith(".pdf"):
                reader = pypdf.PdfReader(tmp_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            elif file.name.endswith(".docx"):
                doc = docx.Document(tmp_path)
                text = "\n".join([p.text for p in doc.paragraphs])
            else:  # txt
                try:
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(tmp_path, "r", encoding="latin-1") as f:
                        text = f.read()
            
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file.name}
                ))
        finally:
            os.remove(tmp_path)
    
    return documents

def generate_style_dossier(template_files, api_key):
    """
    ETAPA 1 DO PIPELINE FORENSE: Gera o Dossiê de Identidade Decisional.
    
    Executa a análise dos 5 Pilares uma única vez (cacheado por set de templates).
    Retorna dict com: 'dossier', 'glossary', 'cloning_prompt', 'full_response'.
    """
    # Check cache first
    cache_key = _template_cache_key(template_files)
    if cache_key in _style_dossier_cache:
        print(f"✅ Dossiê de estilo recuperado do cache (key: {cache_key[:8]}...)")
        return _style_dossier_cache[cache_key]
    
    if not HAS_AZURE_OPENAI:
        print("⚠️ Azure OpenAI não instalado. Dossiê de estilo indisponível.")
        return {"error": "Azure OpenAI não instalado. Dossiê de estilo não disponível."}
    
    try:
        print("🧬 Gerando Dossiê de Identidade Decisional (5 Pilares)...")
        
        # Reset seek position on all template files (BytesIO safety)
        for f in template_files:
            if hasattr(f, 'seek'):
                f.seek(0)
        
        # 1. Extrair texto dos templates (sem ChromaDB/RAG — desnecessário para dossiê)
        all_docs = _extract_template_texts(template_files)
        
        if not all_docs:
            print("⚠️ Nenhum documento extraído dos templates.")
            return {"error": "Nenhum texto pôde ser extraído dos arquivos enviados. Verifique se são PDFs com texto (não imagens escaneadas)."}
        
        # 2. Concatenar TODOS os textos (não random sampling como antes)
        # Limita a ~80k chars para caber no context window do Flash
        full_text = ""
        for doc in all_docs:
            source = doc.metadata.get('source', 'desconhecido')
            full_text += f"\n\n=== DECISÃO ({source}) ===\n{doc.page_content}\n"
            if len(full_text) > 80000:
                full_text += "\n[... documentos adicionais truncados por limite de contexto ...]\n"
                break
        
        # 3. Chamar LLM com o prompt forense de 5 pilares
        llm = get_llm("gpt-5.2-chat", temperature=0.3, max_tokens=8000)
        
        messages = [
            SystemMessage(content=PROMPT_STYLE_ANALYZER),
            HumanMessage(content=f"Aqui está o acervo de decisões do magistrado para análise forense:\n{full_text}")
        ]
        
        response = llm.invoke(messages)
        content = safe_content(response)
        
        # 4. Parse structured response into 3 parts
        result = _parse_dossier_response(content)
        result['full_response'] = content
        
        # 5. Cache result
        _style_dossier_cache[cache_key] = result
        
        print(f"✅ Dossiê gerado com sucesso:")
        print(f"   - Dossiê: {len(result.get('dossier', ''))} chars")
        print(f"   - Glossário: {len(result.get('glossary', ''))} chars")
        print(f"   - System Prompt: {len(result.get('cloning_prompt', ''))} chars")
        
        return result
        
    except Exception as e:
        print(f"❌ Erro ao gerar Dossiê de Estilo: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Erro ao gerar dossiê: {str(e)}"}

def _parse_dossier_response(content: str) -> dict:
    """
    Parseia a resposta estruturada do prompt forense usando os delimitadores.
    Retorna dict com 'dossier', 'glossary', 'cloning_prompt'.
    """
    result = {'dossier': '', 'glossary': '', 'cloning_prompt': ''}
    
    # Try structured parsing with delimiters
    if '===PARTE_1_DOSSIE===' in content:
        parts = content.split('===PARTE_1_DOSSIE===')
        if len(parts) > 1:
            rest = parts[1]
            
            if '===PARTE_2_GLOSSARIO===' in rest:
                dossier_part, rest2 = rest.split('===PARTE_2_GLOSSARIO===', 1)
                result['dossier'] = dossier_part.strip()
                
                if '===PARTE_3_SYSTEM_PROMPT===' in rest2:
                    glossary_part, rest3 = rest2.split('===PARTE_3_SYSTEM_PROMPT===', 1)
                    result['glossary'] = glossary_part.strip()
                    
                    # Remove trailing ===FIM=== if present
                    cloning = rest3.split('===FIM===')[0] if '===FIM===' in rest3 else rest3
                    result['cloning_prompt'] = cloning.strip()
                else:
                    result['glossary'] = rest2.strip()
            else:
                result['dossier'] = rest.strip()
    else:
        # Fallback: treat entire response as dossier (unstructured)
        print("⚠️ Resposta não contém delimitadores estruturados. Usando como dossiê completo.")
        result['dossier'] = content
        result['cloning_prompt'] = content  # Use full content as cloning instruction
    
    return result

def retrieve_mirror_context(text, api_key, template_files, style_dossier=None):
    """
    Estratégia do Espelho APRIMORADA.
    Combina: System Prompt de Clonagem (do dossiê) + Golden Sample (RAG top-1).
    
    Args:
        text: Texto do processo sendo analisado
        api_key: Chave de API para embeddings
        template_files: Arquivos de modelo de decisão
        style_dossier: Dict com 'dossier', 'glossary', 'cloning_prompt' (opcional)
    """
    if not template_files: return ""
    
    try:
        rag_context = ""
        
        # ── ETAPA 1: Injetar System Prompt de Clonagem (estilo geral) ──
        if style_dossier:
            cloning_prompt = style_dossier.get('cloning_prompt', '')
            glossary = style_dossier.get('glossary', '')
            
            if cloning_prompt:
                rag_context += "\n\n## 🧬 SYSTEM PROMPT DE CLONAGEM (ESTILO DO MAGISTRADO)\n"
                rag_context += "⚠️ INSTRUÇÃO PRIMÁRIA: Você DEVE seguir rigorosamente as diretrizes abaixo para replicar o estilo deste magistrado.\n"
                rag_context += f"\n{cloning_prompt}\n"
            
            if glossary:
                rag_context += "\n\n## 📝 GLOSSÁRIO DO MAGISTRADO (CHECKLIST DE VOCABULÁRIO)\n"
                rag_context += "Use estas expressões e conectivos obrigatoriamente ao redigir:\n"
                rag_context += f"\n{glossary}\n"
        
        # ── ETAPA 2: Golden Sample / Caso Espelho (gabarito estrutural específico) ──
        retriever, _ = process_templates(template_files, api_key)
        
        if not retriever: return rag_context
        
        relevant_docs = retriever.invoke(text[:6000])
        
        if relevant_docs:
            mirror_doc = relevant_docs[0]
            other_docs = relevant_docs[1:]
            
            rag_context += "\n\n## 💎 CASO ESPELHO (GOLDEN SAMPLE - GABARITO ESTRUTURAL)\n"
            rag_context += f"⚠️ INSTRUÇÃO DE CLONAGEM ESTRUTURAL: O caso abaixo ({mirror_doc.metadata.get('source')}) é o seu GABARITO.\n"
            rag_context += "1. Copie a estrutura de tópicos (titulação, numeração).\n"
            rag_context += "2. Use os mesmos jargões e frases de transição do Glossário acima.\n"
            rag_context += "3. Se for o mesmo assunto, adapte apenas os fatos e nomes, mantendo a fundamentação jurídica.\n"
            rag_context += f"\n--- INÍCIO DO CASO ESPELHO ---\n{mirror_doc.page_content}\n--- FIM DO CASO ESPELHO ---\n"
            
            if other_docs:
                rag_context += "\n## OUTROS MODELOS DE REFERÊNCIA (CONTEXTO ADICIONAL)\n"
                for i, doc in enumerate(other_docs):
                    rag_context += f"\n[MODELO SECUNDÁRIO {i+2} - {doc.metadata.get('source')}]:\n{doc.page_content[:3000]}...\n"
                    
        return rag_context
    except Exception as e:
        print(f"Erro no retrieve_mirror_context: {e}")
        return ""


def run_standard_orchestration(text: str, main_llm_config: dict, style_llm_config: dict, status_callback=None, template_files=None, google_key=None, outline=None, style_prompt=None):
    """
    Pipeline Padrão (V1) FLEXÍVEL.
    Suporta qualquer LLM para Analista Principal e Analista de Estilo.
    """
    # Config keys: {'provider': str, 'model': str, 'key': str}
    
    def update(msg):
        if status_callback: status_callback(msg)

    try:
        # Instancia LLMs via Azure AI Foundry
        main_llm = get_llm("gpt-5.2-chat", temperature=0.2)
        style_llm = get_llm("gpt-5.2-chat", temperature=0.3)
    except Exception as e:
        return {"final_report": f"Erro na inicialização dos modelos: {str(e)}", "steps": {}}

    # PROCESSAMENTO DE TEMPLATES (RAG + DOSSIÊ FORENSE)
    rag_context = ""
    style_report = None
    style_dossier = None
    
    if template_files:
        # 1. Gerar Dossiê de Estilo Forense (cacheado) — usa Azure OpenAI internamente
        update("🧬 Gerando Dossiê de Identidade Decisional (5 Pilares)...")
        style_dossier = generate_style_dossier(template_files, None)
        style_report = style_dossier.get('dossier') if style_dossier else None
        
        # 2. Retrieve Mirror Context (agora com dossiê integrado)
        update("📚 Localizando Caso Espelho (Golden Sample)...")
        rag_context = retrieve_mirror_context(text, None, template_files, style_dossier=style_dossier)

    update(f"🧠 Iniciando Análise Profunda ({main_llm_config['model']})...")

    # 1. ANÁLISE INTEGRAL (MÉRITO/MINUTA)
    update("⚖️ Fase 1: Análise Integral e Minutagem (Analista Sênior)...")
    
    # --- LOAD KNOWLEDGE BASE (V4.5 Logic) ---
    kb_text = ""
    try:
        # NOTE: Mantendo Knowledge Base, mas REMOVENDO a substituição forçada de Prompt V3.
        # Queremos usar PROMPT_CLAUDE_INTEGRAL (JSON Mode)
        base_path = "data/knowledge_base"
        files_map = {
            "ARQUIVO A (Sobrestamento) 30.10.2025.txt": "ARQUIVO A (SOBRESTAMENTOS)",
            "ARQUIVO B (Súmulas) - 30.12.2025.txt": "ARQUIVO B (SÚMULAS)",
            "ARQUIVO C (Qualificados) - 30.12.2025.txt": "ARQUIVO C (QUALIFICADOS)"
        }
        for fname, label in files_map.items():
            fpath = os.path.join(base_path, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(fpath, "r", encoding="latin-1") as f:
                        content = f.read()
                if content.strip():
                    kb_text += f"\n=== {label} ===\n{content}\n"
        
        # GARANTE USO DO PROMPT V1 OTIMIZADO (JSON)
        final_prompt_integral = PROMPT_CLAUDE_INTEGRAL
        if kb_text:
            final_prompt_integral += f"\n\n## 6. BASE DE CONHECIMENTO VINCULANTE (CARREGADA)\n{kb_text}"
            
    except Exception as e:
        final_prompt_integral = PROMPT_CLAUDE_INTEGRAL 
        print(f"Erro KB ou Prompt: {e}")

    # Injeta contexto RAG (Estilo)
    if rag_context:
        final_prompt_integral += rag_context
        
    # INJEÇÃO DO OUTLINE (PLANEJAMENTO)
    if outline:
        final_prompt_integral += f"\n\n## 📋 ESQUELETO LÓGICO (PLANEJAMENTO OBRIGATÓRIO)\nSiga estritamente esta estrutura para redigir a decisão:\n{outline}"
        
    # INJEÇÃO DE ESTILO (FEW-SHOT)
    # Se style_prompt for um FewShotTemplate, precisamos formatar.
    # Por simplicidade, se style_prompt existir, extraímos o texto dos exemplos
    few_shot_text = ""
    if style_prompt:
         try:
             # Formatação manual rápida dos exemplos para injetar no system
             for msg in style_prompt.format_messages(page_content=""):
                 few_shot_text += f"\nExemplo: {msg.content}\n"
             if few_shot_text:
                 final_prompt_integral += f"\n\n## 🎭 CLONAGEM DE ESTILO (RAG DINÂMICO)\nEscreva NO MESMO TOM destes exemplos:\n{few_shot_text}"
         except Exception as e:
             print(f"Erro ao formatar Style Prompt: {e}")

    integral_messages = [
        SystemMessage(content=final_prompt_integral),
        HumanMessage(content=f"Realize a ANÁLISE INTEGRAL E MINUTAGEM deste processo:\n\n[AUTOS DO PROCESSO]: {text[:200000]}") 
    ]
    integral_response = safe_content(main_llm.invoke(integral_messages))
    
    # --- REFLEXION LOOP (ACTIVE AUDITOR) ---
    update("🛡️ Rodando Auditoria Ativa (Verificando Alucinações)...")
    # run_reflexion_loop agora usa get_llm() internamente (Azure)
    final_output, audit_log = run_reflexion_loop(integral_response, text, None)
    
    return {
        "final_report": final_output,
        "auditor_dashboard": audit_log, 
        "style_report": style_report,
        "steps": {
            "integral": final_output
        }
    }

def run_ensemble_orchestration(text: str, keys: dict, status_callback=None, template_files=None):
    """
    V2: LINEAR ENSEMBLE PIPELINE (A "Linha de Montagem").
    Pipeline determinístico onde cada modelo faz uma parte específica.
    
    Fluxo:
    1. Gemini 3 Pro -> Extração de Fatos e Análise Formal (Input Massivo).
    2. DeepSeek R1 -> Análise Material/Mérito e Lógica Jurídica (Reasoning).
    3. Claude 4.6 Sonnet -> Redação Final (Minuta) com base nos insumos.
    """
    def update(msg):
         if status_callback: status_callback(msg)

    # 1. Setup Models
    try:
        # Todos os modelos via Azure AI Foundry (Claude Sonnet 4.6)
        analista_fatos = get_llm("gpt-5.2-chat", temperature=0.1)

        juiz_logico = get_llm("gpt-5.2-chat", temperature=0.3)
              
        redator_final = get_llm("gpt-5.2-chat", temperature=0.2)
             
    except Exception as e:
        return {"final_report": f"Erro ao inicializar Banca Digital: {e}", "steps": {}}

    logs = {}
    
    # MIRROR STRATEGY FOR V2 (com Dossiê Forense)
    rag_context = ""
    style_dossier = None
    if template_files:
        update("🧬 (V2) Gerando Dossiê de Identidade Decisional...")
        style_dossier = generate_style_dossier(template_files, keys['google'])
        update("📚 (V2) Localizando Caso Espelho...")
        rag_context = retrieve_mirror_context(text, keys['google'], template_files, style_dossier=style_dossier)

    # === FASE 1: EXTRAÇÃO E TRIAGEM (GEMINI) ===
    update("🕵️‍♂️ Fase 1: Gemini analisando Fatos e Requisitos Formais...")
    
    # Prompt de Fatos
    msg_fatos = [SystemMessage(content=PROMPT_FATOS), HumanMessage(content=f"Autos:\n{text[:150000]}")]
    res_fatos = safe_content(analista_fatos.invoke(msg_fatos))
    logs['fatos'] = res_fatos
    
    # Prompt Formal
    msg_formal = [SystemMessage(content=PROMPT_ANALISE_FORMAL), HumanMessage(content=f"Autos:\n{text[:100000]}")] # Menos contexto ok
    res_formal = safe_content(analista_fatos.invoke(msg_formal))
    logs['analise_formal'] = res_formal
    
    # === FASE 2: RACIOCÍNIO JURÍDICO (DEEPSEEK) ===
    # === FASE 2: RACIOCÍNIO JURÍDICO (DEEPSEEK) ===
    update("🧠 Fase 2: DeepSeek deliberando sobre o Mérito (Reasoning)...")
    
    # Monta o contexto para o Juiz
    contexto_juiz = f"""
    [RESUMO DOS FATOS]:
    {res_fatos}
    
    [TRIAGEM FORMAL]:
    {res_formal}
    
    [TRECHOS RELEVANTES DOS AUTOS]:
    {text[:50000]} 
    """
    
    # Prepare Mirror Context if available
    final_style_guide = keys.get('style_guide', "")
    if rag_context:
        final_style_guide += rag_context

    msg_material = PROMPT_JUIZ_DEEPSEEK.format(
        fatos_texto=res_fatos,
        formal_json=res_formal,
        style_guide=final_style_guide or "Estilo Padrão (Sem guia específico)."
    )
    
    # Use Invoke
    res_material = safe_content(juiz_logico.invoke([HumanMessage(content=contexto_juiz), SystemMessage(content=msg_material)]))
    logs['analise_material'] = res_material
    
    # === FASE 3: REDAÇÃO DE MINUTA (CLAUDE) ===
    update("✍️ Fase 3: Claude redigindo a Minuta Final (Sentença)...")
    
    msg_redator = PROMPT_REDATOR_CLAUDE.format(
        verdict_outline=res_material,
        style_guide=final_style_guide or "Estilo Padrão (Sem guia específico)."
    )
    
    res_final = safe_content(redator_final.invoke([HumanMessage(content=msg_redator)]))
    logs['minuta_final'] = res_final
    
    # === FASE 4: AUDITORIA FINAL (GPT-4o) ===
    final_output = res_final
    audit_log = "Auditoria GPT ignorada (Sem chave ou desativado)"

    # Check for OpenAI key availability
    if keys.get('openai') and HAS_OPENAI:
        update("🛡️ Fase 4: GPT-4o Auditando (Anti-Alucinação)...")
        try:
            auditor_gpt = ChatOpenAI(model="gpt-4o", api_key=keys['openai'], temperature=0.0)
            msg_audit = [
                SystemMessage(content=PROMPT_AUDITOR_GPT),
                HumanMessage(content=f"MINUTA PARA REVISÃO:\n{res_final}\n\nAUTOS:\n{text[:20000]}")
            ]
            audit_resp = safe_content(auditor_gpt.invoke(msg_audit))
            logs['auditoria_gpt'] = audit_resp
            
            if "ERRO:" in audit_resp or "REPROVADO" in audit_resp:
                audit_log = f"⚠️ ALERTA DO AUDITOR:\n{audit_resp}"
            else:
                audit_log = "✅ Aprovado pelo GPT-4o."
                
        except Exception as e:
            audit_log = f"Erro na auditoria GPT: {e}"

    return {
        "final_report": final_output,
        "auditor_dashboard": audit_log,
        "style_report": "Ensemble Assembly Line (Sem Style Guide específico)",
        "steps": logs
    }

def process_templates(files, api_key, collection_name="rag_templates_persistent"):
    """
    Processa arquivos de template (PDF/DOCX/TXT) e cria um retriever.
    """
    documents = []
    
    # Fast local splitter only — HybridSemanticChunker was calling Azure embeddings
    # during chunking, then ChromaDB called embeddings AGAIN when indexing, doubling
    # API calls and making indexing very slow (~30-60s). RecursiveCharacterTextSplitter
    # is instant (local CPU).
    chunker = None
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in files:
        # Reset seek position (safety for BytesIO objects)
        if hasattr(file, 'seek'):
            file.seek(0)
        # Salva temporariamente para processar
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            text = ""
            if file.name.endswith(".pdf"):
                reader = pypdf.PdfReader(tmp_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            elif file.name.endswith(".docx"):
                doc = docx.Document(tmp_path)
                text = "\n".join([p.text for p in doc.paragraphs])
            else: # txt
                try:
                    with open(tmp_path, "r", encoding="utf-8") as f: text = f.read()
                except UnicodeDecodeError:
                     with open(tmp_path, "r", encoding="latin-1") as f: text = f.read()
            
            # Adiciona metadados
            if chunker:
                try:
                    doc_chunks = chunker.split_text(text, source_metadata={"source": file.name})
                except Exception as e:
                    print(f"Erro no Semantic Chunking do arquivo {file.name}: {e}. Usando fallback.")
                    doc_chunks = fallback_splitter.create_documents([text], metadatas=[{"source": file.name}])
            else:
                 doc_chunks = fallback_splitter.create_documents([text], metadatas=[{"source": file.name}])
                 
            documents.extend(doc_chunks)
        finally:
            os.remove(tmp_path)
    
    
    if not documents:
        return None, []

    # Embeddings e Vector Store (PERSISTENTE) — Azure OpenAI
    embeddings = get_embedding_function()
    
    # Define caminho persistente (Railway Volume ou Local)
    # No Railway, defina CHROMA_DB_PATH como variável de ambiente apontando para o volume (ex: /app/data)
    persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db_rag")
    
    # Delete existing collection to avoid dimension mismatch with stale embeddings
    # (e.g. collection was created with 768-dim model but current model is 3072-dim)
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        try:
            client.delete_collection(collection_name)
            print("🗑️ Collection anterior removida (evita conflito de dimensões).")
        except Exception:
            pass
    except Exception:
        pass
    
    # Instancia o banco persistente
    vectorstore = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    # Adiciona os novos documentos
    vectorstore.add_documents(documents)
    
    # Retorna o retriever e os docs para análise de estilo imediata
    return vectorstore.as_retriever(search_kwargs={"k": 5}), documents

def load_persistent_rag(api_key=None, collection_name="rag_templates_persistent"):
    """
    Tenta carregar o banco de dados persistente (se existir).
    Usa Azure OpenAI Embeddings.
    """
    try:
        persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db_rag")
        if os.path.exists(persist_dir):
            embeddings = get_embedding_function()
            vectorstore = Chroma(
                persist_directory=persist_dir, 
                embedding_function=embeddings,
                collection_name=collection_name
            )
            # Verifica se tem dados (hack simples)
            if vectorstore._collection.count() > 0:
                print(f"RAG Persistente carregado: {vectorstore._collection.count()} docs.")
                return vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        print(f"Erro ao carregar RAG persistente: {e}")
    return None

def generate_style_report(documents, api_key):
    """
    Usa um modelo rápido para ler os templates e criar um perfil de estilo.
    Migrado para Azure OpenAI (gpt-4.1-mini).
    """
    try:
        llm_flash = get_llm("gpt-4.1-mini", temperature=0.3)
        
        
        # Concatena amostras dos documentos (Random Sampling)
        sample_text = ""
        # Seleciona de 3 a 5 chunks aleatórios para ter variabilidade
        num_samples = min(5, len(documents))
        if num_samples > 0:
            selected_docs = random.sample(documents, num_samples)
            for doc in selected_docs:
                sample_text += f"\n--- AMOSTRA ({doc.metadata.get('source')}): ---\n{doc.page_content[:5000]}\n"
            
        messages = [
            SystemMessage(content=PROMPT_STYLE_ANALYZER),
            HumanMessage(content=f"Aqui estão amostras de decisões do magistrado. Crie o Dossiê de Estilo:\n{sample_text}")
        ]
        
        response = llm_flash.invoke(messages)
        content = safe_content(response)
            
        return clean_text(content)
    except Exception as e:
        return f"Erro ao gerar perfil de estilo: {str(e)}"

# OLD run_gemini_orchestration removed/replaced by run_standard_orchestration

def process_batch(files, api_key):
    """
    Processa múltiplos arquivos (PDF/DOCX) para o X-Ray.
    Retorna uma lista de strings (textos extraídos).
    """
    processed_texts = []
    
    for file in files:
        # Reutiliza a lógica de extração salvando em temp
        suffix = os.path.splitext(file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        
        try:
           # Extração simplificada, sem vetorização individual
           text_content = ""
           if suffix == ".pdf":
               loader = PyPDFLoader(tmp_path)
               docs = loader.load()
               text_content = "\n".join([d.page_content for d in docs])
           elif suffix == ".docx":
               from langchain_community.document_loaders import Docx2txtLoader
               loader = Docx2txtLoader(tmp_path)
               docs = loader.load()
               text_content = "\n".join([d.page_content for d in docs])
           elif suffix == ".txt":
               from langchain_community.document_loaders import TextLoader
               loader = TextLoader(tmp_path)
               docs = loader.load()
               text_content = "\n".join([d.page_content for d in docs])
           
           if text_content:
               processed_texts.append(f"--- PROCESSO: {file.name} ---\n{clean_text(text_content[:20000])}") # Limita chars por doc para caber no contexto
               
        except Exception as e:
            print(f"Erro ao ler {file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    return processed_texts

from prompts_claude import PROMPT_XRAY_MAP, PROMPT_XRAY_BATCH

def map_process_individual(text_content, filename, api_key=None):
    """
    ETAPA MAP: Analisa um único processo e retorna JSON estruturado.
    Usa GPT-4.1-mini (Azure) para rapidez e custo baixíssimo nesta triagem massiva paralela.
    """
    try:
        llm = get_llm("gpt-4.1-mini", temperature=0.1)
        
        # Força strict JSON no prompt
        map_prompt = PROMPT_XRAY_MAP + "\n\nCRÍTICO: Retorne APENAS UM JSON (Strict JSON). Nenhuma palavra fora das chaves {}."
        
        messages = [
            SystemMessage(content=map_prompt),
            HumanMessage(content=f"Arquivo: {filename}\n\n{text_content[:20000]}")
        ]
        response = safe_content(llm.invoke(messages))
        
        # Limpa JSON
        cleaned = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        data["filename"] = filename # Garante que o nome do arquivo persista
        return data
        
    except Exception as e:
        print(f"Falha total no Map de {filename} com gpt-4.1-mini (Azure). Erro: {e}")
        return {
            "filename": filename, 
            "error": f"Falha na leitura (GPT-4.1-mini). Err: {str(e)}", 
            "sintese_fatos": "Erro de leitura estruturada", 
            "tags_juridicas": ["ERRO"]
        }

def generate_batch_xray(files, api_key, template_files=None):
    """
    Gera o Raio-X da carteira usando estratégia MAP-REDUCE.
    1. MAP: Extrai metadados de cada processo individualmente (Paralelo).
    2. REDUCE: Envia lista de metadados para o Gemini agrupar.
    """
    try:
        # 1. PROCESSAMENTO DE TEXTO (Leitura)
        raw_texts = []
        # Precisamos ler os arquivos primeiro. Reutilizando lógica simples do process_batch mas retornando tuplas (nome, texto)
        for file in files:
            suffix = os.path.splitext(file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            
            try:
                content = ""
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    content = "\n".join([d.page_content for d in docs])
                elif suffix == ".docx":
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(tmp_path)
                    docs = loader.load()
                    content = "\n".join([d.page_content for d in docs])
                elif suffix == ".txt":
                    from langchain_community.document_loaders import TextLoader
                    try:
                        loader = TextLoader(tmp_path, encoding='utf-8')
                        content = loader.load()[0].page_content
                    except Exception:
                        loader = TextLoader(tmp_path, encoding='latin-1')
                        content = loader.load()[0].page_content
                
                if content:
                    raw_texts.append((file.name, clean_text(content)))
            except Exception as e:
                print(f"Erro lendo {file.name}: {e}")
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

        if not raw_texts:
            return {"error": "Nenhum texto extraído."}

        # 2. ETAPA MAP (Execução Paralela)
        mapped_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {
                executor.submit(map_process_individual, text, fname, api_key): fname 
                for fname, text in raw_texts
            }
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    res = future.result()
                    mapped_data.append(res)
                except Exception as e:
                    print(f"Erro no map thread: {e}")

        # 3. ETAPA REDUCE (Clusterização)
        # Prepara o JSON consolidado para o Gemini
        mapped_json_str = json.dumps(mapped_data, ensure_ascii=False, indent=2)
        
        # Cria dicionário de cache para retorno {filename: text}
        text_cache = {fname: text for fname, text in raw_texts}
        
        # Prepara Contexto de Modelos (Templates)
        models_context = ""
        if template_files:
            # Templates também poderiam passar pelo Map-Reduce se fossem muitos, 
            # mas vamos assumir que são poucos e ler direto.
            model_texts = process_batch(template_files, api_key) # Reusing legacy function just for text extraction
            if model_texts:
                 models_context = "\n\n## MODELOS DE REFERÊNCIA DISPONÍVEIS:\n" + "\n".join(model_texts)
        
        human_msg = f"""
        Aqui estão as FICHAS TÉCNICAS dos processos processados individualmente.
        Agrupe-os e gere o relatório de Raio-X.
        
        [DADOS DOS PROCESSOS (JSON)]:
        {mapped_json_str}
        
        {models_context}
        """
        
        messages = [
            SystemMessage(content=PROMPT_XRAY_BATCH),
            HumanMessage(content=human_msg)
        ]
        
        # Usa Claude 4.6 Sonnet para agregação de altíssima qualidade
        try:
            llm_reduce = get_llm("gpt-5.2-chat", temperature=0.1)
            
            # Força JSON
            reduce_prompt = PROMPT_XRAY_BATCH + "\n\nCRÍTICO: Retorne APENAS UM JSON VÁLIDO. Sem Markdown, sem formatação extra, inicie com { e termine com }."
            
            messages = [
                SystemMessage(content=reduce_prompt),
                HumanMessage(content=human_msg)
            ]
            
            response = safe_content(llm_reduce.invoke(messages))
            content = response
            
        except Exception as e:
            print(f"Erro Crítico no Reduce (Claude 4.6): {e}")
            return {"error": f"Erro na consolidação de dados. Detalhe: {str(e)}", "raw_content": ""}, text_cache

        
        # Garante que content é string (algumas versões retornam lista)
        if isinstance(content, list):
            # Se for lista de strings ou objetos com text, tenta converter
            try:
                content = "".join([str(c) for c in content])
            except Exception:
                content = str(content)
        elif not isinstance(content, str):
            content = str(content)
        
        # Limpeza do JSON
        try:
            # Tenta extrair bloco JSON delimitado por markdown
            json_match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group(1).strip()
            else:
                # Fallback: Tenta encontrar o maior bloco JSON possível ({...} ou [...])
                match = re.search(r"(\[.*\]|\{.*\})", content, re.DOTALL)
                if match:
                    cleaned_json = match.group(1).strip()
                else:
                    cleaned_json = content.replace("```json", "").replace("```", "").strip()

            try:
                return json.loads(cleaned_json), text_cache
            except json.JSONDecodeError:
                # Tenta corrigir JSON malformado (ex: aspas simples, trailing commas)
                 try:
                     import ast
                     # ast.literal_eval consegue parsear dicts python stringficados ({'key': 'val'})
                     repaired = ast.literal_eval(cleaned_json)
                     return repaired, text_cache
                 except Exception:
                     # Última tentativa: regex replace de aspas simples
                     try:
                         repaired_str = cleaned_json.replace("'", '"').replace("Mm.", "Mm").replace("Exa.", "Exa") # Hacks comuns
                         return json.loads(repaired_str), text_cache
                     except Exception:
                        pass
                 
                 # Se tudo falhar, retorna erro
                 return {"error": "Falha ao decodificar JSON do Reduce", "raw_content": content}, text_cache

        except Exception as inner_e:
             return {"error": f"Erro interno JSON: {str(inner_e)}", "raw_content": content}, text_cache
        
    except Exception as e:
        return {"error": f"Erro Geral no Pipeline: {str(e)}\n{traceback.format_exc()}"}, {}




def process_single_case_pipeline(pdf_bytes, filename, api_key, template_files=None, cached_text=None, mode="v1", keys=None, ocr_engine_choice="paddle"):
    """
    Função Worker para processar um único caso completo.
    Suporta V1 (Gemini Only) e V2 (Hybrid Agents).
    """
    try:
        # 1. Extract Text
        if cached_text:
            text_content = cached_text
            clean_content = text_content # Já deve vir limpo do cache
        else:
            # Fallback para leitura de bytes se não tiver cache
            # Precisamos salvar bytes em temp file para loaders funcionarem
            suffix = os.path.splitext(filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            
            try:
                if suffix == ".pdf":
                    # Tentativa 1: Leitura de Texto Nativo
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    text_content = "\n".join([d.page_content for d in docs])
                    
                    # Tentativa 2: OCR Avançado (se texto vazio e OCR habilitado)
                    # Se tiver menos de 100 caracteres
                    if len(text_content.strip()) < 100 and HAS_OCR:
                        print(f"⚠️ Texto insuficiente ({len(text_content)} chars) em {filename}. Iniciando OCR Avançado (OpenCV + Paddle)...")
                        try:
                            # Chama o motor escolhido
                            ocr_text = ocr_engine.extract_text_from_pdf(tmp_path, engine=ocr_engine_choice)
                            
                            # Se OCR retornou algo razoável, usa
                            if len(ocr_text) > len(text_content):
                                text_content = ocr_text
                                print(f"✅ OCR Avançado extraiu {len(text_content)} caracteres.")
                            elif "[ERRO]" in ocr_text:
                                print(f"Falha no OCR: {ocr_text}")
                                
                        except Exception as e:
                             print(f"Erro no OCR Pipeline: {e}")

                             
                elif suffix == ".docx":
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(tmp_path)
                    docs = loader.load()
                    text_content = "\n".join([d.page_content for d in docs])
                elif suffix == ".txt":
                    try:
                        with open(tmp_path, "r", encoding="utf-8") as f: text_content = f.read()
                    except UnicodeDecodeError:
                        with open(tmp_path, "r", encoding="latin-1") as f: text_content = f.read()
                else:
                    text_content = ""
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
                
            try:
                clean_content = clean_text(text_content)
                # Ensure it's a valid string, replacing bad chars if needed
                if isinstance(clean_content, bytes):
                    clean_content = clean_content.decode('utf-8', errors='replace')
            except Exception as e_clean:
                clean_content = "Erro de decodificação de texto."
                print(f"Erro no clean_text: {e_clean}")
        
        # --- RAPTOR INTEGRATION (LONG CONTEXT) ---
        # Se o texto for muito grande (> 150k chars), aciona o RAPTOR para resumir
        if len(clean_content) > 150000:
             print(f"🦖 RAPTOR ATIVADO: Texto Grande ({len(clean_content)} chars). Iniciando indexação hierárquica...")
             try:
                 # Decide key/provider
                 raptor_key = keys.get('openai') if keys and keys.get('openai') else (keys.get('google') if keys else api_key)
                 raptor_provider = "openai" if (keys and keys.get('openai')) else "google"
                 
                 raptor = RaptorEngine(api_key=raptor_key, provider=raptor_provider)
                 
                 # Gera Árvore de Resumos
                 tree_summary = raptor.build_tree(clean_content)
                 
                 # Substitui o texto original pela Árvore (que é bem menor e focada)
                 # Mantendo um prefixo identificando
                 clean_content = f" [MODO RAPTOR ATIVADO]\nO texto a seguir é um RESUMO HIERÁRQUICO do processo original.\n\n{tree_summary}"
                 print("✅ RAPTOR finalizado. Texto reduzido com sucesso.")
                 
             except Exception as e_raptor:
                 print(f"⚠️ Erro ao executar RAPTOR: {e_raptor}. Usando texto original truncado.")
        
        # --- AGENTIC WORKFLOW (PLANNER -> WRITER -> CRITIC) ---
        # Substitui a lógica manual anterior pelo Grafo do LangGraph
        
        final_draft = ""
        outline = None       # Inicialização para evitar NameError no fallback V1
        style_prompt = None  # Idem
        
        # Decide keys for agents
        agent_key = keys.get('openai') if keys and keys.get('openai') else (keys.get('google') if keys else api_key)
        agent_provider = "openai" if (keys and keys.get('openai')) else "google"

        # Verifica se o workflow está disponível
        if create_agent_workflow is not None:
            try:
                 print("🚀 Iniciando Workflow Agêntico (Planner -> Writer -> Critic)...")
                 app = create_agent_workflow()
                 
                 inputs = {
                     "facts": clean_content,
                     "api_key": agent_key,
                     "provider": agent_provider,
                     "revision_count": 0
                 }
                 
                 # Executa o Grafo
                 result_state = app.invoke(inputs)
                 final_draft = result_state.get("draft", "Erro na geração do draft.")
                 
                 print("✅ Workflow Completo com Sucesso!")
                 
            except Exception as e_workflow:
                 print(f"⚠️ Erro no Workflow Agêntico: {e_workflow}. Caindo para pipeline legado.")
                 # Fallback logic could be here if needed, but for now allow flow to continue to standard orchestration if draft is empty
                 final_draft = None
        else:
            print("⚠️ Workflow não disponível. Usando pipeline legado.")
            final_draft = None

        if final_draft:
            # Se o Workflow funcionou, retornamos direto (bypass legacy orchestration)
             return {
                "status": "success",
                "filename": filename,
                "analysis": final_draft,
                "model_used": f"Agentic Workflow V3 ({agent_provider})",
                "timestamp": time.time()
            }

        # 2. Run Pipeline (Legacy / Fallback)
        if mode == "v3" and keys:
            # V3: Autonomous Agent (Hybrid LangGraph Agents)
            if run_autonomous_magistrate is None:
                return {"error": "ERRO DE INSTALAÇÃO (V3): Engine Agente não disponível.", "filename": filename}
            
            # --- MIRROR STRATEGY FOR V3 (com Dossiê Forense) ---
            mirror_context = ""
            if template_files:
                 _dossier = generate_style_dossier(template_files, keys.get('google') or api_key)
                 mirror_context = retrieve_mirror_context(clean_content, keys.get('google') or api_key, template_files, style_dossier=_dossier)

            # Normalizar output para o formato esperado pelo front
            # returns (final_json, logs_list)
            v3_json, v3_logs = run_autonomous_magistrate(clean_content, keys)
            
            # Extract content safely
            final_minuta = v3_json.get("minuta_final", "Minuta não gerada.")
            reasoning = v3_json.get("fundamentacao_logica", "Raciocínio não disponível.")
            
            # Format reasoning string if it's a dict
            if isinstance(reasoning, dict):
                 reasoning = "\n".join([f"**{k}:** {v}" for k,v in reasoning.items()])
            
            results = {
                "final_report": final_minuta,
                "auditor_dashboard": "Auditoria Integrada ao Processo V3 (Ver Logs)",
                "style_report": "Gerado via Agentic Style Guide (V3)",
                "steps": {"logs": v3_logs},
                "diagnostic_reasoning": reasoning  # <--- NEW FIELD
            }
            
        elif mode == "v2" and keys:
            # V2: Ensemble Pipeline (Assembly Line)
            # Gemini -> DeepSeek -> Claude
            ensemble_output = run_ensemble_orchestration(clean_content, keys, template_files=template_files)
            results = ensemble_output # Já retorna no formato certo
            results["filename"] = filename # Garante filename
            
            # EXPOSE REASONING V2 (DeepSeek)
            results["diagnostic_reasoning"] = ensemble_output.get("steps", {}).get("analise_material", "Raciocínio não disponível.")
            
        else:
            # V1: Standard Pipeline (Flexible LLM)
            # V1: Standard Pipeline (Flexible LLM)
            # Precisamos mapear os parâmetros antigos para o novo formato de config
            # Se chamou via process_single_case_pipeline, api_key é a chave Google (default V1 legacy)
            # Para usar multi-model no batch, precisaremos passar 'keys' no futuro.
            # Por compatibilidade, se 'keys' não existir, assume Google Default.
            
            if keys:
                 # Novo formato: usa as chaves e modelos do keys se disponíveis
                 main_cfg = keys.get('v1_main_config', {'provider': 'google', 'model': 'gemini-3-pro-preview', 'key': api_key})
                 style_cfg = keys.get('v1_style_config', {'provider': 'google', 'model': 'gemini-3-flash-preview', 'key': api_key})
            else:
                 # Fallback legacy
                 main_cfg = {'provider': 'google', 'model': 'gemini-3-pro-preview', 'key': api_key}
                 style_cfg = {'provider': 'google', 'model': 'gemini-3-flash-preview', 'key': api_key}

            results = run_standard_orchestration(clean_content, main_cfg, style_cfg, status_callback=None, template_files=template_files, google_key=api_key, outline=outline, style_prompt=style_prompt)
        
        # === NORMALIZAÇÃO DO FINAL_REPORT ===
        # Garante que final_report seja sempre string (alguns modelos retornam lista)
        if results.get("final_report"):
            fr = results["final_report"]
            if isinstance(fr, list):
                # Extrai texto de estruturas como [{'type': 'text', 'text': '...'}]
                text_parts = []
                for item in fr:
                    if isinstance(item, dict) and 'text' in item:
                        text_parts.append(item['text'])
                    else:
                        text_parts.append(str(item))
                results["final_report"] = "\n".join(text_parts)
            elif not isinstance(fr, str):
                results["final_report"] = str(fr)
        
        # 3. Save Result
        report_id = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()
        
        # Add metadata for the UI
        results["filename"] = filename
        results["report_id"] = report_id
        results["timestamp"] = time.time()
        
        # Ensure directory exists
        os.makedirs("data/reports", exist_ok=True)
        
        with open(f"data/reports/{report_id}.json", "w") as f:
            json.dump(results, f, ensure_ascii=False)
            
        return {"report_id": report_id, "filename": filename, "status": "success"}

    except Exception as e:
        return {"error": str(e), "filename": filename}

def process_batch_parallel(files, api_key, template_files=None, text_cache_dict=None, progress_callback=None, mode="v1", keys=None, ocr_engine_choice="gpt4o_mini"):
    """
    Processa lista de arquivos EM SÉRIE (para evitar Rate Limit).
    Suporta V1/V2/V3 via worker.
    """
    # Pré-carrega bytes para evitar problemas de thread/cursor
    files_data = []
    for f in files:
        try:
            f.seek(0)
            c_text = None
            if text_cache_dict and f.name in text_cache_dict:
                c_text = text_cache_dict[f.name]
            
            # Leitura robusta de bytes
            content = f.read()
            if isinstance(content, str):
                content = content.encode('utf-8', errors='replace')
                
            files_data.append({
                "name": f.name,
                "bytes": content,
                "cached_text": c_text
            })
        except Exception as e:
            files_data.append({
                "name": f.name,
                "bytes": None,
                "cached_text": None,
                "error": f"Erro de Leitura (Upload): {str(e)}"
            })

    results_list = []
    total_files = len(files_data)
    
    print(f"🚀 Iniciando Batch Profundo (Série) para {total_files} arquivos...")
    os.makedirs("data/reports", exist_ok=True)

    # Execução em SÉRIE (Loop)
    # Motivo: Evitar Rate Limit do Google/OpenAI e uso excessivo de RAM com ChromaDB múltiplos
    for i, data in enumerate(files_data):
        try:
            filename = data["name"]
            print(f"🔄 Processando {i+1}/{total_files}: {filename}")
            
            if progress_callback:
                 progress_callback(i, total_files, filename) # 0-indexed start
            
            # Verifica se houve erro de leitura prévia
            if "error" in data:
                results_list.append({
                    "filename": filename,
                    "error": data["error"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                continue
            
            # Chama o Pipeline Completo (OCR -> RAG -> Agentes)
            res = process_single_case_pipeline(
                pdf_bytes=data["bytes"], 
                filename=filename, 
                api_key=api_key, 
                template_files=template_files,
                cached_text=data["cached_text"],
                mode=mode,
                keys=keys,
                ocr_engine_choice=ocr_engine_choice
            )
            
            # Usa report_id gerado pelo process_single_case_pipeline (evita duplicação)
            res['mode'] = mode
            res['filename'] = filename
            if 'timestamp' not in res:
                res['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")

            results_list.append(res)
            
            # Opcional: Pausa curta para aliviar API
            time.sleep(1)

        except Exception as e:
            print(f"❌ Erro Crítico em {data['name']}: {e}")
            traceback.print_exc()
            results_list.append({
                "filename": data['name'],
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
    # Final Callback
    if progress_callback:
        progress_callback(total_files, total_files, "Concluído!")
                
    return results_list
