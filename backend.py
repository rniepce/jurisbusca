import os
import re
import tempfile
from typing import List, Optional, Any
import pypdf
import docx
from langchain_community.document_loaders import PyPDFLoader
# Tenta importar o loader com OCR; se não der, segue sem ele (ou avisa)
# rapidocr-onnxruntime e rapidocr-pdf devem estar instalados
try:
    from langchain_community.document_loaders import RapidOCRPDFLoader
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import ChatOllama # Removido para deploy Gemini Only
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings # Removido para deploy Gemini Only
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import gc
# from prompts import PROMPT_FATOS, PROMPT_ANALISE_FORMAL, PROMPT_ANALISE_MATERIAL, PROMPT_RELATOR_FINAL
# from prompts_auditor import PROMPT_AUDITOR_FATICO, PROMPT_AUDITOR_EFICIENCIA, PROMPT_AUDITOR_JURIDICO, PROMPT_AUDITOR_DASHBOARD
from prompts_gemini import PROMPT_GEMINI_INTEGRAL, PROMPT_GEMINI_AUDITOR, PROMPT_STYLE_ANALYZER, PROMPT_XRAY_BATCH
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False



def clean_text(text: str) -> str:
    """
    Higienização agressiva para peças jurídicas (Otimização de Context Window).
    Remove: Cabeçalhos, Rodapés, Números de Página, Espaços duplos.
    """
    # 1. Normalização de quebras de linha
    text = text.replace('\r', '')
    
    # 2. Remove cabeçalhos de numeração de processo (ex: "Processo nº 1234..." repetido)
    text = re.sub(r'(?i)(fls\.\s*\d+|processo\s*nº?[:\s]*[\d\.\-]+)', '', text)
    
    # 3. Remove rodapés de escritório/sistema
    # Padrão comum: "Rua X, nº Y... | www.advocacia..." ou "PJe - Assinado eletronicamente"
    text = re.sub(r'(?i)(assinado\s+eletronicamente|documento\s+assinado|pje).*', '', text) 
    
    # 4. Remove números de página soltos
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # 5. Redução de ruído visual (traços, asteriscos)
    text = re.sub(r'[_=\-\*]{3,}', '', text)
    
    # 6. Compressão de espaços (White space normalization)
    text = re.sub(r'\s+', ' ', text)
    
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
    print("⚠️ Usando Embeddings Locais (HuggingFace)...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
            
            # 2. Se falhar (PDF escaneado/imagem), tenta OCR
            if total_chars < 500:
                if HAS_OCR:
                    print(f"Detectado PDF imagem (apenas {total_chars} chars). Iniciando OCR...")
                    loader_ocr = RapidOCRPDFLoader(tmp_path)
                    docs = loader_ocr.load()
                else:
                    text += "[AVISO: PDF parece ser imagem e biblioteca de OCR não encontrada.]\n"
        
        elif suffix == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            
        elif suffix == ".txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(tmp_path, encoding='utf-8')
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
            chunk_size=1000,
            chunk_overlap=200,
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

def get_llm(model_name: str, api_key: str = None, temperature: float = 0.1):
    """
    Retorna a instância do LLM (Ollama ou OpenAI).
    """
    if "gpt" in model_name.lower():
        if not api_key:
            # Fallback or error is better handled in UI, but raising here is safe
            raise ValueError("API Key é obrigatória para modelos GPT.")
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature)
    
    # Configuração para modelos locais via Ollama ou MLX
    
    # 1. Verifica se é um modelo MLX registrado
    if model_name in LOCAL_MODELS:
        # Retorna o wrapper MLX
        # Nota: O ideal seria cachear isso no nível da aplicação (app.py) para não recarregar pesos
        # Mas instanciamos aqui para manter a assinatura.
        return MLXChatWrapper(model_name)

    # 2. Tenta conectar ao host local padrão (Ollama)
    return ChatOllama(model=model_name, temperature=temperature, base_url="http://localhost:11434")

def run_orchestration(text: str, model_mode: str = "auto", api_key: str = None, status_callback=None):
    """
    Executa o pipeline multi-agente com ROUTING DE MODELOS.
    :param model_mode: "auto" (usa mapa de especialistas) ou nome de modelo único.
    """
    
    # Cache local de sessão para evitar recarregar o mesmo modelo seguidamente
    current_llm = None
    current_model_key = None
    
    def get_agent_llm(role):
        nonlocal current_llm, current_model_key
        
        target_model = AGENT_MODEL_MAP.get(role, "qwen2.5-14b-juris") if model_mode == "auto" else model_mode
        
        # Se for modelo GPT/Ollama (não MLX), usa o get_llm padrão
        if target_model not in LOCAL_MODELS:
            return get_llm(target_model, api_key)
            
        # Se já estamos com o modelo certo carregado, retorna ele
        if current_llm and current_model_key == target_model:
            return current_llm
            
        # Se precisamos trocar de modelo
        if current_llm and hasattr(current_llm, "unload"):
            current_llm.unload()
            
        if status_callback:
            status_callback(f"🔄 Carregando Especialista: {target_model}...")
            
        current_llm = MLXChatWrapper(target_model)
        current_model_key = target_model
        return current_llm

    # helper para invocar
    def invoke_agent(system_prompt, user_content, agent_name, role_key):
        if status_callback:
            model_name = current_model_key if current_model_key else model_mode
            status_callback(f"🤖 {agent_name} trabalhando... (Model: {model_name})")
        
        llm = get_agent_llm(role_key)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        response = llm.invoke(messages)
        return response.content

    try:
        # --- PASSO 1: ANÁLISE FORMAL (Phi-3.5 - Rápido) ---
        formal_out = invoke_agent(PROMPT_ANALISE_FORMAL, f"Analise formalmente a petição:\n\n{text[:50000]}", "Agente de Análise Formal", "formal")
        
        # --- PASSO 2: FATOS (Qwen 14B - Contexto) ---
        fatos_out = invoke_agent(PROMPT_FATOS, f"Extraia os dados básicos deste processo:\n\n{text[:50000]}", "Agente de Fatos", "fatos")
        
        # --- PASSO 3: MATERIAL/TEMPORAL (Qwen 14B - Raciocínio) ---
        # Note: Fatos e Material usam o mesmo modelo, então não haverá reload aqui
        material_out = invoke_agent(PROMPT_ANALISE_MATERIAL, f"Analise mérito liminar, prescrição e inépcia:\n\n{text[:50000]}\n\nFatos extraídos: {fatos_out}", "Agente de Admissibilidade", "material")
        
        # --- PASSO 4: RELATOR (Gemma 9B - Escrita) ---
        relator_input = PROMPT_RELATOR_FINAL.format(
            fatos_texto=fatos_out,
            formal_json=formal_out,
            material_texto=material_out
        )
        final_report = invoke_agent(relator_input, "Gere o Relatório de Triagem Final.", "Agente Relator/Chefe de Gabinete", "relator")

        # --- PASSO 5: AUDITOR (Qwen 14B - Review) ---
        # 5.1 Auditor Fático
        auditor_fatico_out = invoke_agent(PROMPT_AUDITOR_FATICO.format(fatos_originais=fatos_out, minuta_gerada=final_report), "Valide integridade fática.", "Auditor de Conformidade (Fatos)", "auditor")
        
        # 5.2 Auditor Eficiência
        auditor_eficiencia_out = invoke_agent(PROMPT_AUDITOR_EFICIENCIA.format(minuta_gerada=final_report), "Valide eficiência (Prov. 355).", "Auditor de Conformidade (Eficiência)", "auditor")
        
        # 5.3 Auditor Jurídico
        auditor_juridico_out = invoke_agent(PROMPT_AUDITOR_JURIDICO.format(pedidos_iniciais=fatos_out, minuta_gerada=final_report), "Valide congruência jurídica.", "Auditor de Conformidade (Jurídico)", "auditor")
        
        # 5.4 Dashboard
        dashboard_out = invoke_agent(PROMPT_AUDITOR_DASHBOARD.format(
            status_fatico=auditor_fatico_out,
            status_eficiencia=auditor_eficiencia_out,
            status_juridico=auditor_juridico_out
        ), "Gere o Dashboard final.", "Gerador de Dashboard", "auditor")
        
        return {
            "final_report": final_report,
            "auditor_dashboard": dashboard_out,
            "steps": {
                "fatos": fatos_out,
                "formal": formal_out, 
                "material": material_out,
                "auditor_fatico": auditor_fatico_out,
                "auditor_eficiencia": auditor_eficiencia_out,
                "auditor_juridico": auditor_juridico_out
            }
        }
    finally:
        # Limpeza final de memória
        if current_llm and hasattr(current_llm, "unload"):
            current_llm.unload()

def process_templates(files, api_key):
    """
    Processa arquivos de template (PDF/DOCX/TXT) e cria um retriever.
    """
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

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
                with open(tmp_path, "r") as f:
                    text = f.read()
            
            # Adiciona metadados
            doc_chunks = text_splitter.create_documents([text], metadatas=[{"source": file.name}])
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
        # Usa Gemini 2.0 Flash (se disponível) ou 1.5 Flash para ser rápido
        llm_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.3)
        
        # Concatena amostras dos documentos (máx 30k chars para não gastar muito)
        sample_text = ""
        for doc in documents[:5]: # Pega primeiros 5 chunks
            sample_text += f"\n--- AMOSTRA ({doc.metadata.get('source')}): ---\n{doc.page_content[:5000]}\n"
            
        messages = [
            SystemMessage(content=PROMPT_STYLE_ANALYZER),
            HumanMessage(content=f"Aqui estão amostras de decisões do magistrado. Crie o Dossiê de Estilo:\n{sample_text}")
        ]
        
        return llm_flash.invoke(messages).content
    except Exception as e:
        return f"Erro ao gerar perfil de estilo: {str(e)}"

def run_gemini_orchestration(text: str, api_key: str, status_callback=None, template_files=None):
    """
    Pipeline PROFUNDO usando Gemini 3.0 Pro ou Flash.
    Segue a sequência complexa do usuário: Análise Integral -> Auditoria.
    Suporta RAG (Retrieval Augmented Generation) se templates forem fornecidos.
    """
    if not HAS_GEMINI:
        return {"final_report": "Erro: Pacote langchain-google-genai não instalado.", "steps": {}}
    
    if not api_key:
        return {"final_report": "Erro: API Key do Google não fornecida.", "steps": {}}

    # Instancia Gemini (modelo robusto para análise profunda)
    # Trocando para gemini-3-pro-preview (Solicitado pelo usuário)
    llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=api_key, temperature=0.2)
    
    def update(msg):
        if status_callback:
            status_callback(msg)

    # PROCESSAMENTO DE TEMPLATES (RAG)
    rag_context = ""
    if template_files:
        update("📚 Indexando Modelos de Referência (RAG)...")
        try:
            retriever, all_docs = process_templates(template_files, api_key)
            if retriever:
                # 1. RAG (Busca por similaridade)
                relevant_docs = retriever.invoke(text[:4000])
                rag_context = "\n\n## MODELOS DE REFERÊNCIA (RAG)\n"
                rag_context += "Use o ESTILO e ESTRUTURA visual destes modelos:\n"
                for i, doc in enumerate(relevant_docs):
                    rag_context += f"\n[MODELO {i+1} - {doc.metadata.get('source')}]:\n{doc.page_content}\n"
                
                # 2. STYLE ANALYZER (Flash)
                update("🎨 Analisando Estilo Judicial (Profiling com Gemini Flash)...")
                style_report = generate_style_report(all_docs, api_key)
                if style_report:
                    rag_context += f"\n\n## DIRETRIZES DE PERSONALIDADE (PERFIL DO JULGADOR)\nVocê deve emular estritamente o seguinte perfil:\n{style_report}\n"

        except Exception as e:
            update(f"⚠️ Erro ao processar modelos: {e}")

    update("🧠 Iniciando Análise Profunda (Gemini 3.0 Pro)...")

    # 1. ANÁLISE INTEGRAL (MÉRITO/MINUTA)
    update("⚖️ Fase 1: Análise Integral e Minutagem (Analista Sênior)...")
    
    # Injeta contexto RAG no prompt se houver
    final_prompt_integral = PROMPT_GEMINI_INTEGRAL
    if rag_context:
        final_prompt_integral += rag_context

    integral_messages = [
        SystemMessage(content=final_prompt_integral),
        HumanMessage(content=f"Realize a ANÁLISE INTEGRAL E MINUTAGEM deste processo:\n\n[AUTOS DO PROCESSO]: {text[:150000]}") # Aumentado context
    ]
    integral_response = llm.invoke(integral_messages).content
    
    # 2. REVISOR (AUDITOR)
    update("🛡️ Fase 2: Auditoria Final (Raio-X)...")
    auditor_messages = [
        SystemMessage(content=PROMPT_GEMINI_AUDITOR),
        HumanMessage(content=f"Audite a Minuta abaixo com base nos autos:\n\n[DADOS DOS AUTOS]: {text[:150000]}\n\n[MINUTA A SER AUDITADA]: {integral_response}")
    ]
    auditor_response = llm.invoke(auditor_messages).content
    
    # Consolida tudo
    final_output = f"""
# 🧠 RELATÓRIO DE ANÁLISE PROFUNDA (GEMINI 3.0)

---
## 1. PARECER JURÍDICO E MINUTA
{integral_response}

---
## 2. AUDITORIA DE CONFORMIDADE
{auditor_response}
    """
    
    return {
        "final_report": final_output,
        "auditor_dashboard": auditor_response,
        "style_report": style_report if 'style_report' in locals() else None,
        "steps": {
            "integral": integral_response,
            "auditor": auditor_response
        }
    }

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
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.1)
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
                    loader = TextLoader(tmp_path)
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
        
        # Prepara Contexto de Modelos (Templates)
        models_context = ""
        if template_files:
            # Templates também poderiam passar pelo Map-Reduce se fossem muitos, 
            # mas vamos assumir que são poucos e ler direto.
            model_texts = process_batch(template_files, api_key) # Reusing legacy function just for text extraction
            if model_texts:
                 models_context = "\n\n## MODELOS DE REFERÊNCIA DISPONÍVEIS:\n" + "\n".join(model_texts)
        
        llm_flash = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.1)
        
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
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            return {"error": "Falha ao decodificar JSON do Reduce", "raw_content": content}
        
    except Exception as e:
        import traceback
        return {"error": f"Erro Geral no Pipeline: {str(e)}\n{traceback.format_exc()}"}

import concurrent.futures
import hashlib
import json
import time

def process_single_case_pipeline(file_bytes, filename, api_key, template_files=None):
    """
    Executa a pipeline completa (OCR -> Gemini Integral + Auditoria) para UM caso.
    Salva o resultado em disco para persistência entre abas.
    Retorna o ID do relatório.
    """
    try:
        # 1. OCR / Extração de Texto
        # Precisamos salvar em temp para o loader ler
        suffix = os.path.splitext(filename)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
            
        text_content = ""
        # Reusing simple extraction logic for speed in batch, 
        # but calls run_gemini_orchestration which expects clean text.
        # Ideally we should use process_uploaded_file but it has streamlit dependencies/cache logic which might conflict in threads?
        # Let's use clean pure python logic.
        
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
            
        os.remove(tmp_path) # cleanup
        
        if not text_content:
            return {"error": f"Vazio: {filename}", "filename": filename}
            
        clean_content = clean_text(text_content)
        
        # 2. Run Gemini Pipeline (Deep Analysis)
        # We invoke the EXACT same function used in single mode
        results = run_gemini_orchestration(clean_content, api_key, template_files=template_files)
        
        # 3. Save Result
        report_id = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()
        
        # Add metadata for the UI
        results["filename"] = filename
        results["report_id"] = report_id
        results["timestamp"] = time.time()
        
        # Ensure directory exists
        os.makedirs(".gemini_cache/reports", exist_ok=True)
        
        with open(f".gemini_cache/reports/{report_id}.json", "w") as f:
            json.dump(results, f)
            
        return {"report_id": report_id, "filename": filename, "status": "success"}

    except Exception as e:
        return {"error": str(e), "filename": filename}

def process_batch_parallel(files, api_key, template_files=None):
    """
    Processa lista de arquivos EM PARALELO.
    Retorna lista de resultados (IDs de relatório).
    """
    results_list = []
    
    # Prepara os dados para não passar objetos Streamlit (FileUploader) para threads se não for seguro
    # Actually Streamlit files are bytesIO wrappers, usually thread safe for reading if we read them first.
    # Let's read content into RAM first.
    files_data = []
    for f in files:
        f.seek(0)
        files_data.append({
            "bytes": f.read(),
            "name": f.name
        })
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit tasks
        future_to_file = {
            executor.submit(process_single_case_pipeline, d["bytes"], d["name"], api_key, template_files): d["name"]
            for d in files_data
        }
        
        for future in concurrent.futures.as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                res = future.result()
                results_list.append(res)
            except Exception as exc:
                results_list.append({"error": str(exc), "filename": fname})
                
    return results_list
