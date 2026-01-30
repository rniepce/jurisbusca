import os
import re
import tempfile
from typing import List, Optional, Any
import pypdf
import docx
from langchain_community.document_loaders import PyPDFLoader
# PaddleOCR Imports via ocr_engine
try:
    import ocr_engine
    HAS_OCR = True
    # Import Hybrid Chunker
    from chunking import HybridSemanticChunker
    # Import RAPTOR
    from raptor_engine import RaptorEngine
    # Import Planning & Style
    from planning_engine import PlanningEngine
    from style_engine import StyleEngine
    # Import Workflow
    from agent_workflow import create_agent_workflow
except ImportError:
    HAS_OCR = False
except Exception:
    HAS_OCR = False


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
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    HAS_GEMINI = True
    GEMINI_IMPORT_ERROR = None
except ImportError as e:
    HAS_GEMINI = False
    GEMINI_IMPORT_ERROR = str(e)
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_anthropic import ChatAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
# from prompts import PROMPT_FATOS, PROMPT_ANALISE_FORMAL, PROMPT_ANALISE_MATERIAL, PROMPT_RELATOR_FINAL
# (Re-enabling imports for V2 Ensemble)
from prompts import PROMPT_FATOS, PROMPT_ANALISE_FORMAL, PROMPT_JUIZ_DEEPSEEK, PROMPT_REDATOR_CLAUDE, PROMPT_AUDITOR_GPT
from prompts_gemini import PROMPT_GEMINI_INTEGRAL, PROMPT_GEMINI_AUDITOR, PROMPT_STYLE_ANALYZER, PROMPT_XRAY_BATCH, PROMPT_GEMINI_FIXER
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
    # Detecta tipo de chave
    if api_key:
        if api_key.startswith("AIza"):
            if HAS_GEMINI:
                return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            else:
                print("⚠️ Chave Google detectada mas lib não instalada. Usando local.")
        elif api_key.startswith("sk-"):
            return OpenAIEmbeddings(openai_api_key=api_key)
            
    # Modelo leve para rodar localmente no Mac M3
    # Fallback to Environment Variable if available
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key and HAS_GEMINI:
        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=env_key)
        
    raise ValueError("Nenhum provedor de Embeddings configurado. Por favor, insira a Google API Key.")
    # return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # REMOVIDO PARA EVITAR ERRO

def process_uploaded_file(file_obj, filename: str, api_key=None):
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
            # 1. Tenta extração padrão rápida
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Verifica se extraiu texto suficiente
            # Documentos PDFs digitalizados como imagem retornam pouquíssimo texto (só metadados)
            # Aumentei o threshold para 500 chars para ser mais seguro em docs grandes
            total_chars = sum(len(d.page_content) for d in docs)
            
            # 2. Se falhar (PDF escaneado/imagem) ou texto for muito curto, aciona SEMANTIC OCR (Gemini Vision)
            if total_chars < 500:
                print(f"📉 Texto insuficiente ({total_chars} chars). Acionando Semantic OCR (Gemini Flash)...")
                # Usa chave do ambiente ou passada
                g_key = api_key if api_key and api_key.startswith("AIza") else os.getenv("GOOGLE_API_KEY")
                
                if g_key:
                    ocr_text = extract_text_with_gemini_flash(tmp_path, g_key)
                    if "Erro" not in ocr_text:
                         # Substitui o docs pelo resultado do OCR
                         # Cria um documento único pois o OCR retorna tudo junto
                         from langchain_core.documents import Document
                         docs = [Document(page_content=ocr_text, metadata={"source": filename, "ocr": "semantic_flash"})]
                         print("✅ Semantic OCR concluído com sucesso.")
                    else:
                         text += f"[ERRO OCR: {ocr_text}]\n"
                else:
                     text += "[AVISO: PDF Imagem detectado, mas sem Chave Google para OCR Semântico.]\n"
        
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
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        # Cria Vector Store em memória (ou temp dir que apagamos depois)
        # Usando Chroma
        embedding_function = get_embedding_function(api_key=api_key)
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name="temp_process_analysis" 
            # Não definimos persist_directory para ser in-memory (ephemeral)
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        return full_text, retriever
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Erro ao processar arquivo: {str(e)}", None
    finally:
        # Limpa arquivo temporário
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def get_llm(provider: str, model_name: str, api_key: str, temperature: float = 0.2):
    """
    Factory para instanciar LLMs de diferentes provedores.
    """
    if provider == "google":
        if not HAS_GEMINI: raise ImportError("langchain-google-genai não instalado.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temperature)
    
    elif provider == "openai":
        if not HAS_OPENAI: raise ImportError("langchain-openai não instalado.")
        return ChatOpenAI(model=model_name, api_key=api_key, temperature=temperature)

    elif provider == "deepseek":
        if not HAS_OPENAI: raise ImportError("langchain-openai não instalado (Necessário para DeepSeek).")
        return ChatOpenAI(
            model=model_name, 
            api_key=api_key, 
            base_url="https://api.deepseek.com", 
            temperature=temperature
        )
        
    elif provider == "anthropic":
        if not HAS_ANTHROPIC: raise ImportError("langchain-anthropic não instalado.")
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=temperature)
        
    else:
        raise ValueError(f"Provedor desconhecido: {provider}")

def run_reflexion_loop(draft_text, source_text, api_key):
    """
    ACTIVE AUDITOR (REFLEXION LOOP):
    1. Auditor: Critica a minuta (busca alucinações).
    2. Fixer: Se houver erro, reescreve a minuta e devolve.
    """
    try:
        # Usa Gemini Flash para Auditoria (Rápido e Barato)
        # Note: Flash 1.5/2.0 é ótimo para ler long context
        auditor_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key, temperature=0.0)
        
        # 1. Auditoria
        # Precisamos parsear o draft. Se for JSON (V1 atualizado), extraímos a 'minuta_final'.
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
            except:
                pass

        print("🛡️ Iniciando Auditoria Ativa (Reflexion Loop)...")
        msg_audit = [
            SystemMessage(content=PROMPT_GEMINI_AUDITOR),
            HumanMessage(content=f"DADOS DO PROCESSO:\n{source_text[:50000]}\n\nMINUTA PARA AUDITORIA:\n{draft_content}")
        ]
        
        audit_resp = auditor_llm.invoke(msg_audit).content
        audit_clean = audit_resp.replace("```json", "").replace("```", "").strip()
        
        audit_json = {}
        try:
            audit_json = json.loads(audit_clean)
        except:
            print(f"Erro parse auditoria: {audit_clean}")
            return draft_text, "Falha no Parse da Auditoria"

        # 2. Decisão: Aprova ou Corrige?
        if audit_json.get("aprovado") is True:
            print("✅ Auditoria Aprovada (Sem Alucinações).")
            return draft_text, audit_resp # Retorna original
            
        else:
            errors = audit_json.get("erros_criticos", [])
            print(f"❌ Auditoria Reprovou. Erros: {errors}. Iniciando Auto-Correção...")
            
            # 3. Fixer (Usa o mesmo modelo ou um mais capaz se quisesse, mas Flash serve)
            fixer_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key, temperature=0.1)
            
            msg_fix = PROMPT_GEMINI_FIXER.format(
                draft=draft_content,
                critique=json.dumps(errors, ensure_ascii=False)
            )
            
            fix_resp = fixer_llm.invoke([HumanMessage(content=msg_fix)]).content
            
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
                except:
                    pass
            
            return fix_resp, audit_resp

    except Exception as e:
        print(f"Erro no Reflexion Loop: {e}")
        return draft_text, str(e)

def extract_text_with_gemini_flash(file_path, api_key):
    """
    SEMANTIC OCR (Vision API via Gemini Flash).
    Lê o PDF como imagem/vídeo, extrai o texto e já faz a limpeza estrutural.
    """
    if not HAS_GEMINI:
        return "Erro: Biblioteca Google GenAI não encontrada."

    try:
        # Config API
        genai.configure(api_key=api_key)
        
        # Upload via File API (Mais robusto que converter para imagem localmente)
        print(f"📤 Uploading {os.path.basename(file_path)} to Google File API...")
        sample_file = genai.upload_file(path=file_path, display_name=os.path.basename(file_path))
        
        # Wait for processing
        while sample_file.state.name == "PROCESSING":
            time.sleep(1)
            sample_file = genai.get_file(sample_file.name)
            
        if sample_file.state.name == "FAILED":
             raise ValueError("Google File API processing failed.")
             
        # Generate Content (Vision)
        model = genai.GenerativeModel(model_name="gemini-3-flash-preview") # Version consistent with project
        
        prompt = """
        Aja como um transcritor jurídico de elite. 
        Extraia o texto integral deste documento, preservando a formatação de tópicos. 
        
        ⚠️ REGRAS DE LIMPEZA (IGNORE TUDO ISSO):
        1. Cabeçalhos repetitivos de cada página (ex: "Processo nº...").
        2. Rodapés de sistema (ex: "PJe - Assinado eletronicamente...").
        3. Carimbos, QR Codes e Assinaturas digitais (hash).
        4. Margens laterais com números de linha.
        
        Retorne APENAS o texto limpo e estruturado.
        """
        
        response = model.generate_content([sample_file, prompt])
        
        # Cleanup
        try:
            genai.delete_file(sample_file.name)
        except:
            pass
            
        return response.text
        
    except Exception as e:
        return f"Erro no Semantic OCR: {str(e)}"

def retrieve_mirror_context(text, api_key, template_files):
    """
    Função Auxiliar para implementar a Estratégia do Espelho (Mirror Strategy).
    Recupera o 'Caso Espelho' (Top-1) e monta o contexto.
    """
    if not template_files: return ""
    
    try:
        # Reutiliza process_templates (assume que ele lida com cache ou é rápido)
        retriever, _ = process_templates(template_files, api_key)
        
        if not retriever: return ""
        
        # Busca documento mais relevante (Golden Sample)
        relevant_docs = retriever.invoke(text[:6000])
        
        rag_context = ""
        if relevant_docs:
            mirror_doc = relevant_docs[0]
            other_docs = relevant_docs[1:]
            
            rag_context += "\n\n## 💎 CASO ESPELHO (GOLDEN SAMPLE - GABARITO)\n"
            rag_context += f"⚠️ INSTRUÇÃO DE CLONAGEM: O caso abaixo ({mirror_doc.metadata.get('source')}) é o seu GABARITO ESTRUTURAL OBRIGATÓRIO.\n"
            rag_context += "1. Copie a estrutura de tópicos (titulação, numeração).\n"
            rag_context += "2. Copie os jargões e frases de transição exatas.\n"
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
        # Instancia LLMs
        main_llm = get_llm(main_llm_config['provider'], main_llm_config['model'], main_llm_config['key'], temperature=0.2)
        style_llm = get_llm(style_llm_config['provider'], style_llm_config['model'], style_llm_config['key'], temperature=0.3)
    except Exception as e:
        return {"final_report": f"Erro na inicialização dos modelos: {str(e)}", "steps": {}}

    # PROCESSAMENTO DE TEMPLATES (RAG MIRROR STRATEGY)
    rag_context = ""
    style_report = None
    
    if template_files:
        update("📚 Localizando Caso Espelho (Golden Sample)...")
        # Define API Key para Embeddings (Google)
        rag_key = main_llm_config['key'] if main_llm_config['provider'] == 'google' else (google_key or os.getenv("GOOGLE_API_KEY"))
        
        # 1. Retrieve Mirror Context
        rag_context = retrieve_mirror_context(text, rag_key, template_files)
        
        # 2. Style Report (Legacy Sampling for 'Personality')
        try:
           update(f"🎨 Analisando Estilo Judicial ({style_llm_config['model']})...")
           # We still need all_docs for style analysis, so process_templates is called again
           _, all_docs = process_templates(template_files, rag_key)
           
           sample_text = ""
           num_samples = min(5, len(all_docs))
           selected_docs = random.sample(all_docs, num_samples) if num_samples > 0 else []
           for doc in selected_docs: 
               sample_text += f"\n--- AMOSTRA ({doc.metadata.get('source')}): ---\n{doc.page_content[:4000]}\n"
            
           style_msgs = [
               SystemMessage(content=PROMPT_STYLE_ANALYZER),
               HumanMessage(content=f"Aqui estão amostras de decisões do magistrado. Crie o Dossiê de Estilo:\n{sample_text}")
           ]
           style_resp = style_llm.invoke(style_msgs)
           style_content = style_resp.content
           
           # Cleaning list artifacts if needed (common in Gemini)
           if isinstance(style_content, list):
                style_content = "\n".join([x if isinstance(x, str) else str(x) for x in style_content])
           
           style_report = clean_text(str(style_content))
           
           if style_report:
                rag_context += f"\n\n## DIRETRIZES DE PERSONALIDADE (PERFIL DO JULGADOR)\nUse também este perfil:\n{style_report}\n"

        except Exception as e:
            print(f"Erro no Style Analysis: {e}")

    update(f"🧠 Iniciando Análise Profunda ({main_llm_config['model']})...")

    # 1. ANÁLISE INTEGRAL (MÉRITO/MINUTA)
    update("⚖️ Fase 1: Análise Integral e Minutagem (Analista Sênior)...")
    
    # --- LOAD KNOWLEDGE BASE (V4.5 Logic) ---
    kb_text = ""
    try:
        # NOTE: Mantendo Knowledge Base, mas REMOVENDO a substituição forçada de Prompt V3.
        # Queremos usar PROMPT_GEMINI_INTEGRAL (JSON Mode)
        base_path = "data/knowledge_base"
        files_map = {
            "sobrestamentos.txt": "ARQUIVO A (SOBRESTAMENTOS)",
            "sumulas.txt": "ARQUIVO B (SÚMULAS)",
            "qualificados.txt": "ARQUIVO C (QUALIFICADOS)"
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
        final_prompt_integral = PROMPT_GEMINI_INTEGRAL
        if kb_text:
            final_prompt_integral += f"\n\n## 6. BASE DE CONHECIMENTO VINCULANTE (CARREGADA)\n{kb_text}"
            
    except Exception as e:
        final_prompt_integral = PROMPT_GEMINI_INTEGRAL 
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
    integral_response = main_llm.invoke(integral_messages).content
    
    # --- REFLEXION LOOP (ACTIVE AUDITOR) ---
    update("🛡️ Rodando Auditoria Ativa (Verificando Alucinações)...")
    # Usa a chave google disponível (preferência pela do main_llm se for google)
    reflexion_key = main_llm_config['key'] if main_llm_config['provider'] == 'google' else (google_key or os.getenv("GOOGLE_API_KEY"))
    
    if reflexion_key:
         # Loop de autocorreção
         final_output, audit_log = run_reflexion_loop(integral_response, text, reflexion_key)
    else:
         final_output = integral_response
         audit_log = "Auditoria ignorada (Sem chave Google)"
    
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
    3. Claude 4.5 Sonnet -> Redação Final (Minuta) com base nos insumos.
    """
    def update(msg):
         if status_callback: status_callback(msg)

    # 1. Setup Models
    try:
        # Step A: Analista de Triagem (Gemini - Context Window Grande e Barato)
        analista_fatos = get_llm("google", "gemini-3-pro-preview", keys['google'], temperature=0.1)
        
        # Step B: Juiz Substituto (DeepSeek - Raciocínio Complexo)
        # Se deepseek key nao existir, fallback para openai ou google
        ds_key = keys.get('deepseek')
        if ds_key:
             juiz_logico = get_llm("deepseek", "deepseek-reasoner", ds_key, temperature=0.3) # Updated provider
        else:
             update("⚠️ Chave DeepSeek não encontrada. Usando GPT-4o para raciocínio.")
             juiz_logico = get_llm("openai", "gpt-4o", keys['openai'], temperature=0.3)
             
        # Step C: Assessor Redator (Claude - Melhor Prosa)
        # Se claude key nao existir, fallback para openai
        cl_key = keys.get('anthropic')
        if cl_key:
             redator_final = get_llm("anthropic", "claude-sonnet-4-5-20250929", cl_key, temperature=0.2)
        else:
             update("⚠️ Chave Anthropic não encontrada. Usando GPT-4o para redação.")
             redator_final = get_llm("openai", "gpt-4o", keys['openai'], temperature=0.2)
             
    except Exception as e:
        return {"final_report": f"Erro ao inicializar Banca Digital: {e}", "steps": {}}

    logs = {}
    
    # MIRROR STRATEGY FOR V2
    rag_context = ""
    if template_files:
        update("📚 (V2) Localizando Caso Espelho...")
        rag_context = retrieve_mirror_context(text, keys['google'], template_files)

    # === FASE 1: EXTRAÇÃO E TRIAGEM (GEMINI) ===
    update("🕵️‍♂️ Fase 1: Gemini analisando Fatos e Requisitos Formais...")
    
    # Prompt de Fatos
    msg_fatos = [SystemMessage(content=PROMPT_FATOS), HumanMessage(content=f"Autos:\n{text[:150000]}")]
    res_fatos = analista_fatos.invoke(msg_fatos).content
    logs['fatos'] = res_fatos
    
    # Prompt Formal
    msg_formal = [SystemMessage(content=PROMPT_ANALISE_FORMAL), HumanMessage(content=f"Autos:\n{text[:100000]}")] # Menos contexto ok
    res_formal = analista_fatos.invoke(msg_formal).content
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
    res_material = juiz_logico.invoke([HumanMessage(content=contexto_juiz), SystemMessage(content=msg_material)]).content
    logs['analise_material'] = res_material
    
    # === FASE 3: REDAÇÃO DE MINUTA (CLAUDE) ===
    update("✍️ Fase 3: Claude redigindo a Minuta Final (Sentença)...")
    
    msg_redator = PROMPT_REDATOR_CLAUDE.format(
        verdict_outline=res_material,
        style_guide=final_style_guide or "Estilo Padrão (Sem guia específico)."
    )
    
    res_final = redator_final.invoke([HumanMessage(content=msg_redator)]).content
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
            audit_resp = auditor_gpt.invoke(msg_audit).content
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

def process_templates(files, api_key):
    """
    Processa arquivos de template (PDF/DOCX/TXT) e cria um retriever.
    """
    documents = []
    
    if not HAS_GEMINI:
        print("Aviso: Google Generative AI não instalado. Pulando processamento de templates.")
        return None, []

    # Inicializa Hybrid Semantic Chunker
    # Tenta usar OpenAI key se disponível (melhor qualidade), senão Google
    # Como process_templates recebe apenas 'api_key' (que é google por padrão no código legado),
    # vamos tentar inferir ou usar variável de ambiente.
    openai_key = os.getenv("OPENAI_API_KEY")
    
    chunker = None
    if openai_key:
         print("Using OpenAI Embeddings for Chunking")
         chunker = HybridSemanticChunker(api_key=openai_key, provider="openai")
    else:
         print("Using Google Embeddings for Chunking")
         chunker = HybridSemanticChunker(api_key=api_key, provider="google")

    # Fallback splitter if chunker fails to init
    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for file in files:
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

    # Embeddings e Vector Store (PERSISTENTE)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    
    # Define caminho persistente (Railway Volume ou Local)
    # No Railway, defina CHROMA_DB_PATH como variável de ambiente apontando para o volume (ex: /app/data)
    persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db_rag")
    
    # Instancia o banco persistente
    vectorstore = Chroma(
        persist_directory=persist_dir, 
        embedding_function=embeddings,
        collection_name="rag_templates_persistent"
    )
    
    # Adiciona os novos documentos
    vectorstore.add_documents(documents)
    
    # Retorna o retriever e os docs para análise de estilo imediata
    return vectorstore.as_retriever(search_kwargs={"k": 5}), documents

def load_persistent_rag(api_key):
    """
    Tenta carregar o banco de dados persistente (se existir).
    """
    try:
        if not HAS_GEMINI: return None
        persist_dir = os.getenv("CHROMA_DB_PATH", "./chroma_db_rag")
        if os.path.exists(persist_dir):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
            vectorstore = Chroma(
                persist_directory=persist_dir, 
                embedding_function=embeddings,
                collection_name="rag_templates_persistent"
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
    Usa um modelo rápido (Flash) para ler os templates e criar um perfil de estilo.
    """
    try:
        if not HAS_GEMINI: return "Estilo não disponível (Bibliotecas Google ausentes)"
        # Gemini 3 Flash Preview (ID Correto: gemini-3-flash-preview)
        llm_flash = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key, temperature=0.3)
        
        
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
        content = response.content
        
        # Correção para formato de lista (Gemini 3 Flash Preview as vezes retorna blocos)
        if isinstance(content, list):
            # Tenta extrair texto de blocos do tipo {'type': 'text', 'text': ...}
            text_parts = []
            for part in content:
                if isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
                elif isinstance(part, str):
                    text_parts.append(part)
            return clean_text("\n".join(text_parts))
            
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

from prompts_gemini import PROMPT_XRAY_MAP, PROMPT_XRAY_BATCH

def map_process_individual(text_content, filename, api_key):
    """
    ETAPA MAP: Analisa um único processo e retorna JSON estruturado.
    Usa Gemini Flash para rapidez.
    """
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key, temperature=0.1)
        messages = [
            SystemMessage(content=PROMPT_XRAY_MAP),
            HumanMessage(content=f"Arquivo: {filename}\n\n{text_content[:20000]}")
        ]
        response = llm.invoke(messages).content
        
        # Limpa JSON
        cleaned = response.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        data["filename"] = filename # Garante que o nome do arquivo persista
        return data
    except Exception as e:
        print(f"Erro no Map de {filename}: {e}")
        return {
            "filename": filename, 
            "error": "Falha na leitura", 
            "sintese_fatos": "Erro de leitura", 
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
        
        llm_flash = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key, temperature=0.1)
        
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
        
        response = llm_flash.invoke(messages)
        content = response.content
        
        # Limpeza do JSON
        try:
            cleaned_json = content.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_json), text_cache
        except json.JSONDecodeError:
            return {"error": "Falha ao decodificar JSON do Reduce", "raw_content": content}, text_cache
        
    except Exception as e:
        return {"error": f"Erro Geral no Pipeline: {str(e)}\n{traceback.format_exc()}"}, {}

import concurrent.futures
import time
import random
import tempfile



def process_single_case_pipeline(pdf_bytes, filename, api_key, template_files=None, cached_text=None, mode="v1", keys=None):
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
                            # Chama o motor avançado
                            ocr_text = ocr_engine.extract_text_from_pdf(tmp_path)
                            
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
        
        # Decide keys for agents
        agent_key = keys.get('openai') if keys and keys.get('openai') else (keys.get('google') if keys else api_key)
        agent_provider = "openai" if (keys and keys.get('openai')) else "google"

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
            
            # --- MIRROR STRATEGY FOR V3 ---
            mirror_context = ""
            if template_files:
                 mirror_context = retrieve_mirror_context(clean_content, keys.get('google') or api_key, template_files)

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
        
        # 3. Save Result
        report_id = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()
        
        # Add metadata for the UI
        results["filename"] = filename
        results["report_id"] = report_id
        results["timestamp"] = time.time()
        
        # Ensure directory exists
        os.makedirs("data/reports", exist_ok=True)
        
        with open(f"data/reports/{report_id}.json", "w") as f:
            json.dump(results, f)
            
        return {"report_id": report_id, "filename": filename, "status": "success"}

    except Exception as e:
        return {"error": str(e), "filename": filename}

def process_batch_parallel(files, api_key, template_files=None, text_cache_dict=None, progress_callback=None, mode="v1", keys=None):
    """
    Processa lista de arquivos EM PARALELO.
    Suporta V1/V2 via worker.
    """
    results_list = []
    total_files = len(files)
    
    # Prepara dados para threads
    files_data = []
    for f in files:
        try:
            cached = text_cache_dict.get(f.name) if text_cache_dict else None
            if cached:
                files_data.append({"name": f.name, "bytes": None, "cached_text": cached})
            else:
                f.seek(0)
                # Defensive Read: Ensure bytes
                content = f.read()
                if isinstance(content, str):
                    content = content.encode('utf-8', errors='replace')
                
                files_data.append({
                    "bytes": content,
                    "name": f.name,
                    "cached_text": None
                })
        except Exception as e:
            results_list.append({"error": f"Erro de Leitura (Upload): {str(e)}", "filename": f.name})

    def _worker(data):
        return process_single_case_pipeline(
            pdf_bytes=data["bytes"], 
            filename=data["name"], 
            api_key=api_key, 
            template_files=template_files,
            cached_text=data["cached_text"],
            mode=mode,
            keys=keys
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_file = {
            executor.submit(_worker, d): d["name"]
            for d in files_data
        }
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                res = future.result()
                results_list.append(res)
            except Exception as exc:
                results_list.append({"error": str(exc), "filename": fname})
            
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total_files, fname)
                
    return results_list
