import streamlit as st
import os
from dotenv import load_dotenv
import json
import re
import traceback
import pandas as pd
import plotly.express as px
from backend import process_uploaded_file, run_standard_orchestration, run_ensemble_orchestration, process_templates, generate_style_report, generate_style_dossier, generate_batch_xray, process_batch_parallel, load_persistent_rag, HAS_GEMINI, GEMINI_IMPORT_ERROR
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from prompts import LEGAL_ASSISTANT_PROMPT # Obsoleto com multi-agentes

# Configuração da Página
st.set_page_config(
    page_title="Assistente Rafa - Inteligência Jurídica",
    page_icon="⚖️",
    layout="wide"
)

# Carrega variáveis de ambiente
load_dotenv()


# ==============================================================================
# FUNÇÃO UTILITÁRIA: LIMPEZA DE TEXTO PARA EXIBIÇÃO
# ==============================================================================
def clean_for_display(text):
    """Sanitiza texto vindo de LLM para exibição via st.markdown()."""
    if not text:
        return ""
    if not isinstance(text, str):
        if isinstance(text, list):
            parts = []
            for item in text:
                if isinstance(item, dict) and 'text' in item:
                    parts.append(item['text'])
                else:
                    parts.append(str(item))
            text = "\n".join(parts)
        else:
            text = str(text)
    # 1. Converte escaped newlines
    text = text.replace("\\n", "\n")
    # 2. Detecta JSON inline e formata como bloco de código
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            text = "```json\n" + json.dumps(parsed, indent=2, ensure_ascii=False) + "\n```"
        except Exception:
            pass
    # 3. Remove tags HTML (exceto br/hr que st.markdown pode usar)
    text = re.sub(r'<(?!/?(?:br|hr)\s*/?>)[^>]+>', '', text)
    return text


# ==============================================================================
# FUNÇÃO UTILITÁRIA: EXTRAÇÃO ROBUSTA DE MINUTA
# ==============================================================================
def extract_minuta_from_report(data: dict) -> tuple:
    """
    Extrai minuta e diagnóstico de um relatório de forma robusta.
    Retorna (minuta_text, diagnostic_text).
    """
    steps_data = data.get("steps", {})
    if isinstance(steps_data, dict):
        integral_text = steps_data.get("integral")
    else:
        integral_text = None
        
    full_text = integral_text if integral_text else data.get("final_report", "")
    
    # Normaliza para string
    if isinstance(full_text, list):
        text_parts = []
        for item in full_text:
            if isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
            else:
                text_parts.append(str(item))
        full_text = "\n".join(text_parts)
    elif not isinstance(full_text, str):
        full_text = str(full_text if full_text is not None else "")
    
    # Tenta JSON parse primeiro (V1 modo estruturado)
    minuta_text = None
    diagnostic_text = None
    
    try:
        cleaned_json = full_text.replace("```json", "").replace("```", "").strip()
        data_json = json.loads(cleaned_json)
        if isinstance(data_json, dict):
            minuta_text = data_json.get("minuta_final", "")
            diag = data_json.get("diagnostico", {})
            fund = data_json.get("fundamentacao_logica", "")
            if diag or fund:
                diagnostic_text = f"**Diagnóstico:** {json.dumps(diag, indent=2, ensure_ascii=False)}\n\n**Fundamentação:** {fund}"
    except Exception:
        pass
    
    # Fallback: Regex patterns
    if not minuta_text:
        patterns = [
            r'##\s*3\.\s*MINUTA', r'##\s*MINUTA',
            r'\*\*DO\s+ATO\s+JUDICIAL\*\*', r'DO\s+ATO\s+JUDICIAL',
            r'\*\*SENTENÇA\*\*', r'\*\*DECISÃO\*\*',
            r'##\s*SENTENÇA', r'##\s*DECISÃO'
        ]
        for pattern in patterns:
            parts = re.split(pattern, full_text, flags=re.IGNORECASE)
            if len(parts) > 1:
                diagnostic_text = parts[0].strip()
                minuta_text = parts[1].strip()
                break
        
        if not minuta_text:
            diagnostic_text = "Diagnóstico integral."
            minuta_text = full_text
    
    # V2/V3: diagnostic_reasoning explícito
    if data.get("diagnostic_reasoning"):
        diagnostic_text = data.get("diagnostic_reasoning")
    
    # Limpeza final
    if minuta_text and isinstance(minuta_text, str):
        minuta_text = minuta_text.replace("\\n", "\n")
        if "'extras':" in minuta_text:
            minuta_text = minuta_text.split("'extras':")[0].strip().rstrip(",").strip()
        elif '"extras":' in minuta_text:
            minuta_text = minuta_text.split('"extras":')[0].strip().rstrip(",").strip()
        minuta_text = minuta_text.strip().strip("'").strip('"')
    
    return minuta_text or "", diagnostic_text or ""


# ==============================================================================
# 0. ROTEAMENTO (ROUTER) - PARA ABAS NOVAS (PRIORIDADE ALTA)
# ==============================================================================
query_params = st.query_params
if "report_id" in query_params:
    report_id = query_params["report_id"]
    try:
        # Load from persistent storage
        file_path = f"data/reports/{report_id}.json"
        
        if not os.path.exists(file_path):
             st.error(f"Relatório não encontrado: {file_path}")
             st.stop()
             
        with open(file_path, "r") as f:
            data = json.load(f)
            
        # Defensive fix for 'list' vs 'dict'
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                 st.error(f"Formato de relatório inválido (Lista): {str(data)[:100]}")
                 st.stop()
        
        # --- VIEW: PROCESSO INDIVIDUAL (NOVA ABA) ---
        st.title(f"⚖️ Processo: {data.get('filename', 'Detalhes')}")
        
        # Usa função utilitária centralizada
        minuta_text, diagnostic_text = extract_minuta_from_report(data)

        # Renderiza Decisão
        st.subheader("📝 Minuta da Decisão")
        st.text_area("Copie o texto abaixo:", value=minuta_text, height=600, label_visibility="collapsed")
        
        st.markdown("---")
        st.write("🔎 **Painel de Controle:**")
        
        with st.expander("🛠️ Debug do Texto Original (Se algo estiver cortado)"):
            st.text(f"Tamanho do Texto Original: {len(data.get('final_report', ''))}")
            st.write("JSON COMPLETO (DEBUG):")
            st.json(data)
            
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.popover("🧠 Diagnóstico", use_container_width=True):
                st.markdown(clean_for_display(diagnostic_text))
        with c2:
            if data.get("auditor_dashboard"):
                with st.popover("🛡️ Auditoria", use_container_width=True):
                    st.markdown(clean_for_display(data["auditor_dashboard"]))
        with c3:
            if data.get("style_report"):
                with st.popover("🎨 Estilo", use_container_width=True):
                    st.markdown(clean_for_display(data["style_report"]))
        with c4:
             with st.popover("⚙️ Logs", use_container_width=True):
                st.json(data.get("steps", {}))
        
        st.markdown("---")
        st.info("💬 Modo de Visualização Rápida (Sessão Simplificada)")
        
    except Exception as e:
        st.error(f"Erro ao carregar relatório: {e}")
    
    st.stop() # PARA A EXECUÇÃO AQUI PARA ESTA ABA

# --- CSS Personalizado (Design Moderno) ---
st.markdown("""
<style>
    /* Paleta de Cores Moderna */
    :root {
        --primary-color: #4F46E5; /* Indigo */
        --secondary-color: #10B981; /* Emerald */
        --background-dark: #1E1B4B;
        --text-primary: #1F2937;
        --text-secondary: #6B7280;
        --surface: #FFFFFF;
        --surface-hover: #F3F4F6;
    }
    
    /* Header Principal */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, var(--primary-color), #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
    }
    
    [data-testid="stSidebar"] h1 {
        font-size: 1.4rem !important;
        color: var(--primary-color) !important;
    }
    
    /* Botões */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), #7C3AED);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
    }
    
    /* Cards e Containers */
    .stExpander {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Status Box */
    [data-testid="stStatusWidget"] {
        border-radius: 12px;
        border: 1px solid #E5E7EB;
    }
    
    /* Code Block (Minuta) */
    .stCodeBlock {
        border-radius: 12px !important;
        border: 2px solid var(--primary-color) !important;
    }
    
    /* Popovers */
    [data-testid="stPopover"] > div {
        border-radius: 16px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        transition: border-color 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #E5E7EB;
        border-radius: 12px;
        padding: 1rem;
        transition: border-color 0.2s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-color);
    }
    
    /* Success/Info Messages */
    .stSuccess, .stInfo {
        border-radius: 10px;
    }
    
    /* Chat Messages */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: Configurações e Upload ---
with st.sidebar:
    st.title("🎛️ Controle de Testes")
    
    # Botão de Reset (Nova Conversa)
    if st.button("🗑️ Nova Análise (Manter Modelos)"):
        # Limpa chaves específicas do estado, MAS PRESERVA OS MODELOS
        keys_to_reset = ["messages", "process_text", "retriever", "current_file_name", "style_report_preview"]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        # Força recriação APENAS do uploader principal
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        st.session_state.uploader_key += 1
        st.rerun()

    # Inicializa key do uploader principal
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    # API KEY carregada automaticamente do .env
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")

    # Verifica se a chave está configurada
    if not st.session_state.google_api_key:
        st.error("⚠️ GOOGLE_API_KEY não encontrada no arquivo .env")
        st.info("Configure a variável GOOGLE_API_KEY no arquivo .env na raiz do projeto.")
        st.stop()
    
    # Tenta carregar RAG se ainda não tiver (reload de página)
    if st.session_state.get("retriever") is None and st.session_state.google_api_key:
        retriever = load_persistent_rag(st.session_state.google_api_key)
        if retriever:
            st.session_state.retriever = retriever
    
    # Status discreto da conexão
    st.success("🔑 Google API Conectada", icon="✅")
    st.markdown("---")
    
    # SELETOR DE MODO (V1 vs V2 vs V3)
    mode_option = st.radio(
        "Modo de Operação:",
        ["V1: Standard (Multi-Model)", "V2: Linha de Montagem (Ensemble)", "V3: Agente Autônomo (SOTA)"],
        index=0,
        help="V1: Rápido (1 LLM).\nV2: Potente (Gemini -> DeepSeek -> Claude).\nV3: Autônomo (Ferramentas + Python)."
    )
    
    if "V1" in mode_option:
        st.session_state.app_mode = "v1"
    elif "V2" in mode_option:
        st.session_state.app_mode = "v2"
    else:
        st.session_state.app_mode = "v3"
    
    # CONFIGURAÇÃO V1 (MULTI-MODELO)
    if st.session_state.app_mode == "v1":
         with st.expander("🛠️ Configuração do Motor (V1)", expanded=True):
             st.caption("Escolha a inteligência por trás do Analista Principal e do Analista de Estilo.")
             
             # Definição dos Modelos e Provedores
             model_options = {
                 "Gemini 3.0 Pro": {"provider": "google", "model": "gemini-3-pro-preview"},
                 "Gemini Flash (Rápido)": {"provider": "google", "model": "gemini-3-flash-preview"},
                 "DeepSeek R1 (Lógica Extrema)": {"provider": "deepseek", "model": "deepseek-reasoner"}, # Via DeepSeek API (OpenAI compat)
                 "GPT-5.1 Preview (Simulado/GPT-4o)": {"provider": "openai", "model": "gpt-4o"},
                 "GPT-5.2 (Azure OpenAI)": {"provider": "azure_openai", "model": "gpt-5.2-chat"},
                 "Amazônia IA (Soberano BR)": {"provider": "amazonia", "model": "rodrigomalossi/amazonia-a"}
             }
             
             # Seletores
             sel_main = st.selectbox("🧠 Modelo Principal (Mérito/Minuta)", list(model_options.keys()), index=0)
             sel_style = st.selectbox("🎨 Modelo de Estilo (Personalidade)", list(model_options.keys()), index=1)
             
             # Captura as configs escolhidas
             main_config = model_options[sel_main]
             style_config = model_options[sel_style]
             
             # INPUT DE CHAVES DINÂMICO
             st.divider()
             st.caption("Chaves de Acesso necessárias para os modelos escolhidos:")
             
             needed_providers = set([main_config['provider'], style_config['provider']])
             
             # Google (Já temos a session_state.google_api_key validada lá em cima)
             main_config['key'] = st.session_state.google_api_key
             style_config['key'] = st.session_state.google_api_key
             
             # OpenAI
             if 'openai' in needed_providers:
                 if "openai_key_v1" not in st.session_state: st.session_state.openai_key_v1 = ""
                 k_val = st.text_input("OpenAI API Key", value=st.session_state.openai_key_v1, type="password", key="v1_oai_key")
                 st.session_state.openai_key_v1 = k_val
                 
                 if main_config['provider'] == 'openai': main_config['key'] = k_val
                 if style_config['provider'] == 'openai': style_config['key'] = k_val
             
             # DeepSeek
             if 'deepseek' in needed_providers:
                 if "deepseek_key_v1" not in st.session_state: st.session_state.deepseek_key_v1 = ""
                 k_ds = st.text_input("DeepSeek API Key", value=st.session_state.deepseek_key_v1, type="password", key="v1_ds_key")
                 st.session_state.deepseek_key_v1 = k_ds
                 
                 if main_config['provider'] == 'deepseek': main_config['key'] = k_ds
                 if style_config['provider'] == 'deepseek': style_config['key'] = k_ds
                 
             # Anthropic
             if 'anthropic' in needed_providers:
                 if "anthropic_key_v1" not in st.session_state: st.session_state.anthropic_key_v1 = ""
                 k_ant = st.text_input("Anthropic API Key", value=st.session_state.anthropic_key_v1, type="password", key="v1_ant_key")
                 st.session_state.anthropic_key_v1 = k_ant
                 
                 if main_config['provider'] == 'anthropic': main_config['key'] = k_ant
                 if style_config['provider'] == 'anthropic': style_config['key'] = k_ant
             
             # Amazônia IA
             if 'amazonia' in needed_providers:
                 if "amazonia_key_v1" not in st.session_state: st.session_state.amazonia_key_v1 = ""
                 k_amz = st.text_input("Amazônia IA API Key", value=st.session_state.amazonia_key_v1, type="password", key="v1_amz_key")
                 st.session_state.amazonia_key_v1 = k_amz
                 
                 if main_config['provider'] == 'amazonia': main_config['key'] = k_amz
                 if style_config['provider'] == 'amazonia': style_config['key'] = k_amz
             
             # Salva no Session State para uso nos botões de ação
             st.session_state.v1_main_config = main_config
             st.session_state.v1_style_config = style_config

    # CONFIGURAÇÃO V2/V3 (Chaves Extras)
    if st.session_state.app_mode in ["v2", "v3"]:
        with st.expander("⚙️ Configurar Banca Digital (V2/V3)", expanded=True):
            st.caption("Insira as chaves para ativar a equipe completa.")
            
            # OpenAI (Input com validação visual)
            if "openai_key" not in st.session_state: st.session_state.openai_key = ""
            o_key = st.text_input("OpenAI API Key (Auditor GPT-4o)", value=st.session_state.openai_key, type="password", key="input_openai")
            if o_key: 
                st.session_state.openai_key = o_key
                if o_key.startswith("sk-"): st.success("Válida!", icon="✅")
                else: st.warning("Formato estranho...")

            # Anthropic
            if "anthropic_key" not in st.session_state: st.session_state.anthropic_key = ""
            a_key = st.text_input("Anthropic API Key (Redator Claude)", value=st.session_state.anthropic_key, type="password", key="input_anthropic")
            if a_key:
                st.session_state.anthropic_key = a_key
                if a_key.startswith("sk-ant"): st.success("Válida!", icon="✅")
                else: st.warning("Formato estranho...")

            # DeepSeek
            if "deepseek_key" not in st.session_state: st.session_state.deepseek_key = ""
            d_key = st.text_input("DeepSeek API Key (Juiz Reasoning)", value=st.session_state.deepseek_key, type="password", key="input_deepseek")
            if d_key:
                st.session_state.deepseek_key = d_key
                if d_key.startswith("sk-"): st.success("Válida!", icon="✅")
                else: st.warning("Formato estranho...")
            
            if not (st.session_state.openai_key and st.session_state.anthropic_key and st.session_state.deepseek_key):
                st.warning("⚠️ Preencha todas as chaves para usar o Modo V2/V3 (Ensemble/Agente).")

    # GESTÃO DE PRECEDENTES (VINCULAÇÃO)
    with st.expander("📚 Base Vicunlante (Knowledge)", expanded=False):
        st.caption("Arquivos de consulta obrigatória do Prompt V4.5")
        
        # Arquivo A: Sobrestamentos
        f_sobre = st.file_uploader("Arquivo A: Sobrestamentos", type=["txt"], key="upload_sobre")
        if f_sobre:
            with open("data/knowledge_base/sobrestamentos.txt", "wb") as f: f.write(f_sobre.getbuffer())
            # st.toast("Sobrestamentos Atualizados!", icon="💾") # SILENT MODE
        
        # Arquivo B: Súmulas
        f_sumula = st.file_uploader("Arquivo B: Súmulas", type=["txt"], key="upload_sumula")
        if f_sumula:
             with open("data/knowledge_base/sumulas.txt", "wb") as f: f.write(f_sumula.getbuffer())
             # st.toast("Súmulas Atualizadas!", icon="💾") # SILENT MODE

        # Arquivo C: Qualificados
        f_qualif = st.file_uploader("Arquivo C: Qualificados", type=["txt"], key="upload_qualif")
        if f_qualif:
             with open("data/knowledge_base/qualificados.txt", "wb") as f: f.write(f_qualif.getbuffer())
             # st.toast("Qualificados Atualizados!", icon="💾") # SILENT MODE
    
    google_api_key = st.session_state.google_api_key

    st.header("1. Banco de Modelos (RAG)")
    template_files = st.file_uploader(
        "Suba seus despacho/sentenças para o Gemini usar como estilo:",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="rag_templates_uploader" # Key fixa para não resetar
    )
    
    st.markdown("---")
    st.header("🔤 Configuração de OCR")
    st.caption("Escolha como o sistema deve ler PDFs de imagem (digitalizados).")
    
    ocr_option = st.radio(
        "Motor de Leitura:",
        ["Gemini 2.0 Flash (Nuvem/Rápido)", "PaddleOCR (Local/CPU)", "DeepSeek-OCR-2 (Local/GPU)"],
        index=1,
        help="Gemini: Usa API Vision (Melhor Geral). Paddle: Leve. DeepSeek: Requer GPU (Ótimo para tabelas)."
    )
    
    ocr_map = {
        "Gemini 2.0 Flash (Nuvem/Rápido)": "gpt4o_mini",
        "PaddleOCR (Local/CPU)": "paddle", 
        "DeepSeek-OCR-2 (Local/GPU)": "deepseek"
    }
    st.session_state.ocr_engine_choice = ocr_map[ocr_option]
    
    if template_files:
        st.success(f"✅ {len(template_files)} modelos recebidos!")
        
        if st.button("🧬 Gerar Dossiê de Identidade Decisional"):
             if not google_api_key:
                 st.error("Insira a Google API Key na barra lateral.")
             else:
                with st.spinner("🧬 Analisando DNA da escrita judicial (5 Pilares)..."):
                    try:
                        dossier = generate_style_dossier(template_files, google_api_key)
                        if dossier:
                            st.session_state.style_report_preview = dossier.get('full_response', 'Dossiê não gerado.')
                            st.session_state.style_dossier = dossier
                        else:
                            if not HAS_GEMINI:
                                st.error(f"⚠️ ERRO CRÍTICO: Bibliotecas do Google não instaladas (langchain-google-genai). Impossível extrair texto ou gerar embeddings.\nDetalhe do erro: {GEMINI_IMPORT_ERROR}")
                            else:
                                st.warning("Não consegui extrair texto dos arquivos. Verifique se estão corrompidos ou vazios.")
                    except Exception as e:
                        st.error(f"Erro ao gerar dossiê: {e}")
    
    st.markdown("---")

    st.header("2. Upload do Processo(s)")
    
    uploaded_files = st.file_uploader(
        "Carregue os arquivos (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"],
        help="Para análise individual ou em lote (Raio-X).",
        accept_multiple_files=True, # Agora aceita múltiplos
        key=f"uploader_{st.session_state.uploader_key}"
    )

    st.markdown("---")
    
    st.info("✨ **Modo Google Gemini Pro:**\nEste ambiente roda exclusivamente com a IA mais avançada do Google para tarefas jurídicas.")

# --- Lógica Principal ---

st.markdown('<div class="main-header">🤖 Assistente Rafa</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Inteligência Artificial para Análise Jurídica Profunda</p>', unsafe_allow_html=True)

# Exibe Preview do Estilo se houver
if "style_report_preview" in st.session_state and st.session_state.style_report_preview:
    st.info("🎨 **Perfil de Estilo Identificado (Dossiê do Magistrado):**")
    st.markdown(clean_for_display(st.session_state.style_report_preview))
    if st.button("Fechar Preview do Estilo"):
        del st.session_state.style_report_preview
        st.rerun()
    st.markdown("---")

# ==============================================================================
# LÓGICA PRINCIPAL (DASHBOARD / GABINETE)
# ==============================================================================

# Inicializa estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "process_text" not in st.session_state:
    st.session_state.process_text = ""
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "xray_report" not in st.session_state:
    st.session_state.xray_report = None

# Processamento do Arquivo
if uploaded_files:
    st.markdown("---")
    
    # Seletor de Modo (Automático com Override)
    default_index = 1 if len(uploaded_files) > 1 else 0
    mode = st.radio(
        "Modo de Operação:",
        ["🎯 Análise Profunda (Individual)", "📊 Raio-X de Carteira (Gabinete)"],
        index=default_index,
        horizontal=True,
        key="operation_mode"
    )
    
    # 1. MODO GABINETE / LOTE (Batch Processing)
    if mode == "📊 Raio-X de Carteira (Gabinete)":
        st.info(f"⚡ **Modo Gabinete Ativo:** {len(uploaded_files)} arquivos selecionados para triagem.")
        
        col_xray, col_batch = st.columns(2) # Create columns for buttons

        # Botão para Gerar Raio-X
        with col_xray:
            if st.button("⚡ Gerar Raio-X da Carteira", type="primary"):
                if not google_api_key:
                    st.error("Insira a Google API Key na barra lateral.")
                else:
                    with st.spinner("Analisando carteira e gerando Dashboard (Isso pode levar alguns segundos)..."):
                        # generate_batch_xray returns (report_dict, text_cache_dict)
                        report, text_cache = generate_batch_xray(uploaded_files, google_api_key, template_files=template_files)
                        st.session_state.xray_report = report
                        st.session_state.file_text_cache = text_cache

        # Botão para Processar Gabinete (Paralelo)
        with col_batch:
            if st.button("⚡ Análise em Lote", type="secondary"): # Changed to secondary to differentiate
                if not google_api_key:
                    st.error("Insira a Google API Key na barra lateral.")
                else:
                    with st.spinner(f"Processando {len(uploaded_files)} casos em paralelo (Isso pode levar um tempo)..."):
                        # Processa em Paralelo e Salva JSONs
                        
                        # V2 Keys
                        keys_dict = {
                            "google": google_api_key,
                            "openai": st.session_state.get("openai_key"),
                            "anthropic": st.session_state.get("anthropic_key"),
                            "deepseek": st.session_state.get("deepseek_key")
                        }
                        
                        # V1 Configs (Inject if enabled)
                        if st.session_state.get("app_mode") == "v1":
                            keys_dict['v1_main_config'] = st.session_state.get('v1_main_config')
                            keys_dict['v1_style_config'] = st.session_state.get('v1_style_config')
                        results = process_batch_parallel(
                            uploaded_files, 
                            google_api_key, 
                            template_files=template_files,
                            mode=st.session_state.get("app_mode", "v1"),
                            keys=keys_dict
                        )
                        st.session_state.batch_results = results
        
        # Exibe Raio-X se houver
        if st.session_state.xray_report:
            report_data = st.session_state.xray_report
            
            if "error" in report_data:
                st.error(f"Erro ao gerar Raio-X: {report_data['error']}")
                with st.expander("Ver RAW"):
                    st.text(report_data.get("raw_content", ""))
            else:
                st.markdown("### 📊 Raio-X da Carteira (Interativo)")
                
                # 1. Gráfico de Pizza (Plotly)
                try:
                    clusters = report_data.get("clusters", [])
                    if clusters:
                        df_clusters = pd.DataFrame(clusters)
                        fig = px.pie(df_clusters, names='nome', values='quantidade', title='Distribuição por Temas')
                        st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.warning("⚠️ Instale 'plotly' e 'pandas' para ver os gráficos.")
                except Exception as e:
                    st.error(f"Erro no gráfico: {e}")
                
                # 2. Lista de Clusters com Ação
                st.markdown("### 🧩 Grupos Identificados")
                for cluster in report_data.get("clusters", []):
                    with st.expander(f"📁 {cluster['nome']} ({cluster['quantidade']} processos)"):
                        st.markdown(f"**Descrição:** {cluster['descricao_fato']}")
                        st.markdown(f"**Sugestão:** {cluster['sugestao_minuta']}")
                        st.markdown(f"**Arquivos:** {', '.join(cluster['arquivos'])}")
                        
                        # Botão de Ação Específica para o Cluster
                        if st.button(f"⚡ Processar Grupo '{cluster['nome']}'", key=f"btn_{cluster['id']}"):
                            # Filtra os arquivos
                            target_filenames = cluster['arquivos']
                            subset_files = [f for f in uploaded_files if f.name in target_filenames]
                            
                            if not subset_files:
                                st.warning("Nenhum arquivo correspondente encontrado no upload atual (verifique os nomes).")
                            else:
                                if not google_api_key:
                                    st.error("Insira a Google API Key.")
                                else:
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    def update_progress(current, total, filename):
                                        ratio = current / total
                                        progress_bar.progress(ratio)
                                        status_text.text(f"Processando {current}/{total}: {filename}...")
                                        
                                    try:
                                        # Chama processamento com callback
                                        keys_dict = {
                                            "google": google_api_key,
                                            "openai": st.session_state.get("openai_key"),
                                            "anthropic": st.session_state.get("anthropic_key"),
                                            "deepseek": st.session_state.get("deepseek_key")
                                        }
                                        
                                        # V1 Configs (Inject if enabled)
                                        if st.session_state.get("app_mode") == "v1":
                                            keys_dict['v1_main_config'] = st.session_state.get('v1_main_config')
                                            keys_dict['v1_style_config'] = st.session_state.get('v1_style_config')
                                        results = process_batch_parallel(
                                            subset_files, 
                                            google_api_key, 
                                            template_files=template_files, 
                                            text_cache_dict=st.session_state.file_text_cache,
                                            progress_callback=update_progress,
                                            mode=st.session_state.get("app_mode", "v1"),
                                            keys=keys_dict,
                                            ocr_engine_choice=st.session_state.ocr_engine_choice
                                        )
                                    except Exception as e:
                                        st.error(f"Erro no processamento em lote: {e}")
                                        import traceback
                                        st.text(traceback.format_exc())
                                        results = []
                                        
                                    status_text.empty()
                                    progress_bar.empty()
                                    
                                    # Adiciona aos resultados existentes
                                    existing_ids = {r.get('filename') for r in st.session_state.batch_results}
                                    added_count = 0
                                    
                                    # Ensure directory exists
                                    os.makedirs("data/reports", exist_ok=True)
                                    
                                    for new_res in results:
                                        if new_res.get('filename') not in existing_ids:
                                            # Save to disk for persistence
                                            rid = new_res.get('report_id')
                                            if rid:
                                                with open(f"data/reports/{rid}.json", "w") as f:
                                                    json.dump(new_res, f)
                                            
                                            st.session_state.batch_results.append(new_res)
                                            added_count += 1
                                    
                                    if added_count > 0:
                                        st.success(f"✅ {added_count} novos processos analisados!")
                                    else:
                                        st.info("Nenhum processo novo adicionado (todos já processados).")

            st.markdown("---")

        # Exibe Resultados como Links (Grid)
        if st.session_state.batch_results:
            st.markdown("### 🗂️ Processos Analisados (Clique para abrir)")
            
            # Grid de 4 colunas para botões compactos
            cols = st.columns(4)
            for i, res in enumerate(st.session_state.batch_results):
                with cols[i % 4]:
                    if "error" in res:
                        st.error(f"❌ {res['filename']}")
                        st.caption(res['error'])
                    else:
                        # Substituindo link_button por button + callback para preservar Session State
                        # Substituindo por HTML Link para garantir Nova Aba (target="_blank")
                        
                        # LAZY SAVE / SELF-HEALING: Garante que o arquivo existe antes de gerar o link
                        # Isso corrige o erro de "Relatório não encontrado" para itens processados antes da persistence.
                        rid = res.get('report_id')
                        if rid:
                            fpath = f"data/reports/{rid}.json"
                            if not os.path.exists(fpath):
                                os.makedirs("data/reports", exist_ok=True)
                                with open(fpath, "w") as f:
                                    json.dump(res, f)

                        btn_html = f"""
                        <a href="?report_id={res['report_id']}" target="_blank" style="text-decoration:none;">
                            <div style="
                                border: none;
                                border-radius: 12px;
                                padding: 12px;
                                text-align: center;
                                background: linear-gradient(135deg, #4F46E5, #7C3AED);
                                color: white;
                                font-weight: 600;
                                box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
                                transition: transform 0.2s, box-shadow 0.2s;
                            " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 20px rgba(79, 70, 229, 0.4)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 14px rgba(79, 70, 229, 0.3)'">
                                📄 <br> {res['filename'][:15]}...
                            </div>
                        </a>
                        """
                        st.markdown(btn_html, unsafe_allow_html=True)


    # 2. MODO INDIVIDUAL (Single File)
    else:
        # Se houver múltiplos arquivos, permite escolher qual analisar em profundidade
        if len(uploaded_files) > 1:
            uploaded_file = st.selectbox(
                "Selecione o processo para análise detalhada:", 
                uploaded_files, 
                format_func=lambda x: x.name
            )
        else:
            uploaded_file = uploaded_files[0] # Pega o único arquivo
        
        # Se mudou o arquivo, limpa o estado e reprocessa
        if st.session_state.current_file_name != uploaded_file.name:
            st.session_state.messages = []
            st.session_state.process_text = ""
            st.session_state.retriever = None
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.xray_report = None # Limpa X-RAY anterior se houver
            
            with st.spinner(f"Processando {uploaded_file.name}... (OCR + Vetorização)"):
                # Reseta o buffer para o início
                uploaded_file.seek(0)
                
                # Chama backend para OCR e Vetorização
                text, retriever = process_uploaded_file(
                    uploaded_file, 
                    uploaded_file.name, 
                    api_key=google_api_key,
                    ocr_engine_choice=st.session_state.ocr_engine_choice
                )
                
                if text.startswith("Erro") or text.startswith("Formato"):
                    st.error(text)
                else:
                    st.session_state.process_text = text
                    st.session_state.retriever = retriever
                    st.success(f"Processamento concluído! {len(text)} caracteres extraídos. Vetorização ativa.")
                    
        # Mostra preview (opcional)
        with st.expander("📄 Ver conteúdo textual extraído (OCR)"):
            st.text_area("Conteúdo bruto", st.session_state.process_text, height=200)

        # Botão de Ação Principal
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🚀 Rodar Análise Jurídica", type="primary")
        
        if analyze_btn:
            # from backend import run_orchestration, run_gemini_orchestration # Já importado no topo
            
            if not st.session_state.process_text:
                st.error("O texto do arquivo está vazio.")
            else:
                # Limpa chat anterior para nova análise
                st.session_state.messages = []
                
                # Lógica de Orquestração Multi-Agente
                
                # Container de Status Expansível (Novo no Streamlit)
                status_box = st.status("🤖 Iniciando Orquestração de Agentes...", expanded=True)
                
                def update_status(msg):
                    status_box.write(msg)
                    
                try:
                    # Pipeline de Execução (V1 / V2 / V3)
                    
                    if st.session_state.app_mode == "v3":
                        # V3: AGENTE AUTÔNOMO (Agentic RLM)
                        # Nota: Placeholder enquanto V3 não está 100% implementado
                        from backend import run_hybrid_orchestration
                        keys = {
                            "openai": st.session_state.openai_key,
                            "anthropic": st.session_state.anthropic_key,
                            "deepseek": st.session_state.deepseek_key,
                            "google": google_api_key
                        }
                        if run_hybrid_orchestration:
                            results = run_hybrid_orchestration(st.session_state.process_text, keys)
                            
                            # Type guard: V3/LangGraph sometimes returns list instead of dict
                            if isinstance(results, list):
                                results = {"final_report": "\n".join([str(x) for x in results]), "logs": []}
                            elif not isinstance(results, dict):
                                results = {"final_report": str(results), "logs": []}
                            
                            # Adaptação de output do agente
                            if "final_output" in results: results["final_report"] = results["final_output"]
                            if "audit_report" in results: results["auditor_dashboard"] = results["audit_report"]
                            if "logs" in results: results["steps"] = results["logs"]
                        else:
                             st.error("Engine V3 não encontrada.")
                             st.stop()
                             
                    elif st.session_state.app_mode == "v2":
                        # V2: LINHA DE MONTAGEM (ENSEMBLE)
                        # Requer keys carregadas
                        keys = {
                            "openai": st.session_state.openai_key,
                            "anthropic": st.session_state.anthropic_key,
                            "deepseek": st.session_state.deepseek_key,
                            "google": google_api_key
                        }
                        results = run_ensemble_orchestration(
                            text=st.session_state.process_text,
                            keys=keys,
                            status_callback=update_status,
                            template_files=template_files
                        )
                        
                    else:
                        # V1: STANDARD (SIMPLIFICADO)
                        main_conf = st.session_state.get('v1_main_config', {'provider': 'google', 'model': 'gemini-3-pro-preview', 'key': google_api_key})
                        style_conf = st.session_state.get('v1_style_config', {'provider': 'google', 'model': 'gemini-3-flash-preview', 'key': google_api_key})
                        
                        results = run_standard_orchestration(
                            text=st.session_state.process_text,
                            main_llm_config=main_conf,
                            style_llm_config=style_conf,
                            status_callback=update_status,
                            template_files=template_files,
                            google_key=google_api_key
                        )
                    
                    status_box.update(label="✅ Análise e Auditoria Concluídas!", state="complete", expanded=False)
                    
                    # 1. PARSEAMENTO DO OUTPUT (Separar Diagnóstico vs Minuta)
                    # Type guard: Ensure results is always a dict
                    # results type: {type(results)}
                    if not isinstance(results, dict):
                        if isinstance(results, list):
                            results = {"final_report": "\n".join([str(x) for x in results]), "steps": {}}
                        else:
                            results = {"final_report": str(results) if results else "", "steps": {}}
                        pass  # results converted to dict
                    
                    # === PARSEAMENTO ROBUSTO (V1/V2/V3) ===
                    
                    minuta_text = ""
                    diagnostic_text = ""
                    
                    # CASO 1: V2 ou V3 (Já estruturado no dicionário)
                    if st.session_state.app_mode in ["v2", "v3"]:
                         pass  # Modo V2/V3 - Usando campos diretos
                         minuta_text = results.get("final_report", "")
                         
                         # Diagnóstico vem dos steps/logs
                         steps = results.get("steps", {})
                         diagnostic_parts = []
                         if "fatos" in steps: diagnostic_parts.append(f"**Fatos (Gemini):**\n{steps['fatos'][:500]}...")
                         if "analise_material" in steps: diagnostic_parts.append(f"**Raciocínio (DeepSeek):**\n{steps['analise_material']}")
                         if "verdict_outline" in steps: diagnostic_parts.append(f"**Esboço (DeepSeek):**\n{steps['verdict_outline']}")
                         
                         if diagnostic_parts:
                             diagnostic_text = "\n\n".join(diagnostic_parts)
                         else:
                             diagnostic_text = "Sem diagnóstico detalhado nos logs."

                    # CASO 2: V1 (Pode ser JSON ou Markdown Raw)
                    else:
                        pass  # Modo V1 - Tentando Parse JSON ou Regex
                        raw_output = results.get("final_report", "")
                        
                        # Fix for unexpected List type (e.g. [{'type': 'text', 'text': ...}])
                        if isinstance(raw_output, list):
                            pass  # raw_output is list, extracting text
                            cleaned_parts = []
                            for part in raw_output:
                                if isinstance(part, dict) and 'text' in part:
                                    cleaned_parts.append(part['text'])
                                else:
                                    cleaned_parts.append(str(part))
                            raw_output = "\n".join(cleaned_parts)

                        # Fix for Stringified Dict (e.g. "{'type': 'text', ...}") behavior
                        if isinstance(raw_output, str) and raw_output.strip().startswith("{'type': 'text'"):
                            try:
                                import ast
                                parsed = ast.literal_eval(raw_output)
                                if isinstance(parsed, dict) and 'text' in parsed:
                                    raw_output = parsed['text']
                            except Exception:
                                pass
                        
                        # Tenta Parse JSON (Prompt V3 Core)
                        try:
                            # Limpeza de markdown json wrapper
                            cleaned_json = raw_output.replace("```json", "").replace("```", "").strip()
                            data_v1 = json.loads(cleaned_json)
                            
                            if isinstance(data_v1, dict):
                                minuta_text = data_v1.get("minuta_final", "")
                                diag = data_v1.get("diagnostico", {})
                                mirror = data_v1.get("compliance_espelho", {})
                                fund = data_v1.get("fundamentacao_logica", "")
                                
                                # Formata Texto de Diagnóstico
                                diagnostic_text = f"**Diagnóstico Estruturado:**\n{json.dumps(diag, indent=2, ensure_ascii=False)}"
                                if mirror:
                                     diagnostic_text += f"\n\n**Compliance Espelho:**\n{json.dumps(mirror, indent=2, ensure_ascii=False)}"
                                if fund:
                                     diagnostic_text += f"\n\n**Fundamentação Lógica:**\n{fund}"
                                     
                                pass  # V1 JSON Parse Sucesso
                            else:
                                raise ValueError("JSON não é dict")
                                
                        except Exception as e:
                            pass  # V1 JSON Parse Fallback to Regex
                            # Fallback: Regex Splitting (Legacy Prompt)
                            full_text = raw_output
                            if isinstance(full_text, list):
                                full_text = "\n".join([str(x) for x in full_text])
                            
                            patterns = [
                                r'##\s*3\.\s*MINUTA', r'##\s*MINUTA',
                                r'\*\*DO\s+ATO\s+JUDICIAL\*\*', r'DO\s+ATO\s+JUDICIAL',
                                r'\*\*SENTENÇA\*\*', r'\*\*DECISÃO\*\*',
                                r'##\s*SENTENÇA', r'##\s*DECISÃO'
                            ]
                            
                            minuta_text = None
                            for pattern in patterns:
                                parts = re.split(pattern, full_text, flags=re.IGNORECASE)
                                if len(parts) > 1:
                                    diagnostic_text = parts[0].strip()
                                    minuta_text = parts[1].strip()
                                    break
                            
                            if not minuta_text:
                                diagnostic_text = "Diagnóstico integral (Não foi possível separar minuta)."
                                minuta_text = full_text

                    # --- CORREÇÃO DE FORMATAÇÃO E LIMPEZA FINAL ---
                    if minuta_text and isinstance(minuta_text, str):
                        # 1. Converte quebras de linha escapadas para reais
                        minuta_text = minuta_text.replace("\\n", "\n")
                        
                        # 2. Remove artefatos de dicionário Python/JSON vazando no final
                        # Solução "Nuclear": Corta tudo a partir de 'extras': {'signature'
                        # Isso previne qualquer variação de regex complexo
                        if "'extras':" in minuta_text:
                             minuta_text = minuta_text.split("'extras':")[0].strip().rstrip(",").strip()
                        elif '"extras":' in minuta_text:
                             minuta_text = minuta_text.split('"extras":')[0].strip().rstrip(",").strip()
                        
                        # 3. Remove aspas de tupla se sobrarem no início/fim
                        minuta_text = minuta_text.strip().strip("'").strip('"')

                    # 3. BOTÕES DE ACESSO (DIÁLOGOS/POPOVERS)
                    st.markdown("---")
                    st.write("🔎 **Painel de Controle:**")
                    
                    # Layout: 3 colunas iguais para alinhar os botões
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        with st.popover("🧠 Ver Diagnóstico", use_container_width=True): 
                            st.markdown("### 🧠 Raciocínio (Chain-of-Thought)")
                            # Fix escaped newlines for proper display
                            display_text = diagnostic_text.replace("\\n", "\n") if isinstance(diagnostic_text, str) else str(diagnostic_text)
                            st.markdown(clean_for_display(display_text))
                    
                    with c2:
                        dashboard_text = results.get("auditor_dashboard", "")
                        if dashboard_text:
                            with st.popover("🛡️ Ver Auditoria", use_container_width=True):
                                st.markdown("### 🛡️ Relatório do Auditor")
                                # Fix escaped newlines
                                display_audit = dashboard_text.replace("\\n", "\n") if isinstance(dashboard_text, str) else str(dashboard_text)
                                st.markdown(clean_for_display(display_audit))
                    
                    with c3:
                        style_report = results.get("style_report", "")
                        if style_report:
                            with st.popover("🎨 Ver Estilo", use_container_width=True):
                                st.markdown("### 🎨 Dossiê de Estilo Identificado")
                                # Fix escaped newlines
                                display_style = style_report.replace("\\n", "\n") if isinstance(style_report, str) else str(style_report)
                                st.markdown(clean_for_display(display_style))

                    # Removido Coluna 4 (Debug) como solicitado
                    
                    # Salva no histórico (apenas a minuta para ser útil)
                    st.session_state.messages.append({"role": "user", "content": f"Analise o processo {uploaded_file.name} (Modo Multi-Agente)"})
                    st.session_state.messages.append({"role": "assistant", "content": minuta_text})
                    
                except Exception as e:
                    st.error(f"Erro na execução da orquestração: {e}")
                    st.text(traceback.format_exc())

else:
    st.info("👈 Faça o upload de um processo (ou vários para Raio-X) na barra lateral para começar.")

# --- Área de Chat (Pós Análise com RAG) ---
if st.session_state.messages and st.session_state.retriever:
    st.markdown("---")
    st.subheader("💬 Chat Interativo (com Busca Vetorial)")
    
    # Exibe histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(clean_for_display(msg["content"]))
            
    # Input
    if prompt := st.chat_input("Faça perguntas sobre o caso..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Pesquisando nos autos e gerando resposta..."):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                if not google_api_key:
                    st.error("Insira a Google API Key na barra lateral.")
                else:
                    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=google_api_key, temperature=0.3)
                    
                    # 1. RAG Retrieval: Busca trechos relevantes para a pergunta
                    retrieved_docs = st.session_state.retriever.invoke(prompt)
                    context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # 2. Montagem do Histórico (simplificado para Gemini)
                    chat_history = [
                        SystemMessage(content="Você é um assistente jurídico especializado. Responda de forma precisa, citando os documentos quando relevante."),
                    ]
                    
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    # 3. Adiciona a Pergunta Atual com Contexto Enriquecido (RAG)
                    rag_message_content = f"""
                    Informações Relevantes encontradas nos autos através de busca vetorial:
                    {context_str}
                    
                    Pergunta do Usuário:
                    {prompt}
                    """
                    chat_history.append(HumanMessage(content=rag_message_content))
                    
                    # 4. Invoke LLM
                    response = llm.invoke(chat_history)
                    
                    with st.chat_message("assistant"):
                        st.markdown(clean_for_display(response.content if hasattr(response, 'content') else str(response)))
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                
            except Exception as e:
                st.error(f"Erro: {e}")
                st.expander("Detalhes do erro").text(traceback.format_exc())
