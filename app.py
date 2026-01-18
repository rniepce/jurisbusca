import streamlit as st
import os
from dotenv import load_dotenv
from backend import process_uploaded_file, run_gemini_orchestration, process_templates, generate_style_report
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

# --- CSS Personalizado ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
    }
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: Configurações e Upload ---
with st.sidebar:
    st.title("🎛️ Controle de Testes")
    
    # Botão de Reset (Nova Conversa)
    if st.button("🗑️ Nova Análise / Limpar Tudo"):
        # Limpa chaves específicas do estado
        keys_to_reset = ["messages", "process_text", "retriever", "current_file_name"]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        
        # Força recriação do uploader mudando a key
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
        st.session_state.uploader_key += 1
        st.rerun()

    # Inicializa key do uploader
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    st.header("1. Upload do Processo")
    
    # API KEY logo no início para liberar funções
    google_api_key = st.text_input("Google API Key (Para Gemini):", type="password")
    
    uploaded_file = st.file_uploader(
        "Carregue o arquivo (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"],
        help="O arquivo será processado (OCR se necessário) e vetorizado para análise.",
        key=f"uploader_{st.session_state.uploader_key}"
    )
    
    st.markdown("---")
    
    st.header("📂 Banco de Modelos (RAG)")
    template_files = st.file_uploader(
        "Suba seus despacho/sentenças para o Gemini usar como estilo:",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if template_files:
        st.success(f"✅ {len(template_files)} modelos recebidos!")
        
        if st.button("🎨 Gerar Relatório de Estilo (Preview)"):
            if not google_api_key:
                st.error("Insira a API Key do Google primeiro.")
            else:
                with st.spinner("Lendo modelos e criando perfil estilístico (Gemini Flash)..."):
                    try:
                        # Processa apenas para pegar os textos
                        _, docs = process_templates(template_files, google_api_key)
                        if docs:
                            report = generate_style_report(docs, google_api_key)
                            # Salva no session state para exibir na tela principal
                            st.session_state.style_report_preview = report
                        else:
                            st.warning("Não consegui extrair texto dos arquivos.")
                    except Exception as e:
                        st.error(f"Erro ao gerar estilo: {e}")

    st.markdown("---")

    # google_api_key ja foi pedido acima
    st.markdown("---")
    
    st.info("✨ **Modo Google Gemini Pro:**\nEste ambiente roda exclusivamente com a IA mais avançada do Google para tarefas jurídicas.")

# --- Lógica Principal ---

st.markdown('<div class="main-header">🤖 Assistente Rafa</div>', unsafe_allow_html=True)
st.write("Ferramenta para teste e validação de LLMs finetunados em tarefas de análise jurídica.")

# Exibe Preview do Estilo se houver
if "style_report_preview" in st.session_state and st.session_state.style_report_preview:
    st.info("🎨 **Perfil de Estilo Identificado (Dossiê do Magistrado):**")
    st.markdown(st.session_state.style_report_preview)
    if st.button("Fechar Preview do Estilo"):
        del st.session_state.style_report_preview
        st.rerun()
    st.markdown("---")

# Inicializa estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "process_text" not in st.session_state:
    st.session_state.process_text = ""
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# Processamento do Arquivo
if uploaded_file:
    # Se mudou o arquivo, limpa o estado e reprocessa
    if st.session_state.current_file_name != uploaded_file.name:
        st.session_state.messages = []
        st.session_state.process_text = ""
        st.session_state.retriever = None
        st.session_state.current_file_name = uploaded_file.name
        
        with st.spinner(f"Processando {uploaded_file.name}... (OCR + Vetorização)"):
            # Reseta o buffer para o início
            uploaded_file.seek(0)
            
            # Chama backend para OCR e Vetorização
            text, retriever = process_uploaded_file(uploaded_file, uploaded_file.name, api_key=google_api_key)
            
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
                # Pipeline exclusiva do Gemini (Railway Deploy)
                results = run_gemini_orchestration(
                    text=st.session_state.process_text,
                    api_key=google_api_key,
                    status_callback=update_status,
                    template_files=template_files
                )
                
                status_box.update(label="✅ Análise e Auditoria Concluídas!", state="complete", expanded=False)
                
                # 1. PARSEAMENTO DO OUTPUT (Separar Diagnóstico vs Minuta)
                full_text = results.get("steps", {}).get("integral", results["final_report"])
                
                # Tenta separar a Minuta (geralmente após "## 3. MINUTA" ou "## MINUTA")
                import re
                parts = re.split(r'##\s*3\.\s*MINUTA|##\s*MINUTA', full_text, flags=re.IGNORECASE)
                
                if len(parts) > 1:
                    diagnostic_text = parts[0]
                    minuta_text = parts[1].strip()
                    # Remove possível rodapé de fim de arquivo do prompt ou assinatura extra
                    minuta_text = re.split(r'---', minuta_text)[0].strip()
                else:
                    # Fallback: se não achar a divisão, mostra tudo
                    diagnostic_text = "Diagnóstico integral incorporado ao texto."
                    minuta_text = full_text

                # 2. ÂNCORA (MINUTA FINAL)
                st.subheader("📝 Minuta da Decisão (Texto Puro)")
                # 'language=None' tira as cores de markdown e 'st.code' garante o botão de copiar 
                st.code(minuta_text, language=None)
                
                # 3. BOTÕES DE ACESSO (DIÁLOGOS/POPOVERS)
                st.markdown("---")
                st.write("🔎 **Painel de Controle:**")
                
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    with st.popover("🧠 Ver Diagnóstico e Fundamentação"):
                        st.markdown("### 🧠 Raciocínio (Chain-of-Thought)")
                        st.markdown(diagnostic_text)
                
                with c2:
                    dashboard_text = results.get("auditor_dashboard", "")
                    if dashboard_text:
                        with st.popover("🛡️ Ver Auditoria (Compliance)"):
                            st.markdown("### 🛡️ Relatório do Auditor")
                            st.markdown(dashboard_text)
                
                with c3:
                    style_report = results.get("style_report", "")
                    if style_report:
                        with st.popover("🎨 Ver Análise de Estilo"):
                            st.markdown("### 🎨 Dossiê de Estilo Identificado")
                            st.markdown(style_report)

                with c4:
                    with st.popover("🕵️ Detalhes Técnicos"):
                        st.markdown("### ⚙️ Logs da Orquestração")
                        st.json(results.get("steps", {}))
                
                # Salva no histórico (apenas a minuta para ser útil)
                st.session_state.messages.append({"role": "user", "content": f"Analise o processo {uploaded_file.name} (Modo Multi-Agente)"})
                st.session_state.messages.append({"role": "assistant", "content": minuta_text})
                
            except Exception as e:
                import traceback
                st.error(f"Erro na execução da orquestração: {e}")
                st.text(traceback.format_exc())

else:
    st.info("👈 Faça o upload de um processo na barra lateral para começar.")

# --- Área de Chat (Pós Análise com RAG) ---
if st.session_state.messages and st.session_state.retriever:
    st.markdown("---")
    st.subheader("💬 Chat Interativo (com Busca Vetorial)")
    
    # Exibe histórico
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
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
                        st.markdown(response.content)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                
            except Exception as e:
                import traceback
                st.error(f"Erro: {e}")
                st.expander("Detalhes do erro").text(traceback.format_exc())
