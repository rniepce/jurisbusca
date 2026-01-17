import streamlit as st
import os
from dotenv import load_dotenv
from backend import process_uploaded_file, get_llm, run_orchestration, LOCAL_MODELS, MLXChatWrapper, process_templates, generate_style_report
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

    st.header("2. Seleção do Modelo")
    # Opções hardcoded conforme solicitação do usuário
    # Lista de opções combinando modelos locais MLX e opções antigas
    mlx_options = list(LOCAL_MODELS.keys())
    other_options = ["gpt-4o", "mistral-nemo (ollama)", "llama3 (ollama)"]
    
    model_option = st.selectbox(
        "Escolha o LLM para teste:",
        ["GEMINI 3.0 PRO (Análise Profunda Preview)", "AUTO (Melhor Agente p/ cada Tarefa)"],
        index=0
    )
    
    if model_option.startswith("AUTO"):
        st.info("🧠 **Modo Auto-Pilot:** O sistema escolherá automaticamente o melhor modelo para cada etapa (Phi-3 para formal, Qwen para mérito, Gemma para redação).")
    elif "GEMINI" in model_option:
        st.warning("✨ **Modo Gemini:** Requer API Key do Google. Executa análise profunda em 3 etapas (Triagem -> Mérito -> Auditoria).")
    
    # google_api_key ja foi pedido acima
    
    st.markdown("---")
    st.info("ℹ️ Certifique-se de que o Ollama está rodando (`ollama serve`) se escolher um modelo local.")

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
        from backend import run_orchestration, run_gemini_orchestration
        
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
                # Determina modo: AUTO ou Fixo
                selected_model_mode = "auto" if model_option.startswith("AUTO") else model_option

                if "GEMINI" in model_option:
                     # Pipeline exclusiva do Gemini
                     results = run_gemini_orchestration(
                         text=st.session_state.process_text,
                         api_key=google_api_key,
                         status_callback=update_status,
                         template_files=template_files
                     )
                else:
                    # Pipeline Local / OpenAI
                    results = run_orchestration(
                        text=st.session_state.process_text,
                        model_mode=selected_model_mode,
                        api_key=openai_api_key if "gpt" in model_option else None,
                        status_callback=update_status
                    )
                
                status_box.update(label="✅ Análise e Auditoria Concluídas!", state="complete", expanded=False)
                
                # 1. ÂNCORA (MINUTA FINAL) - DESTAQUE TOTAL
                st.subheader("📝 Minuta Sugerida (Pronta para Assinatura)")
                minuta_text = results.get("steps", {}).get("integral", results["final_report"])
                
                # Caixa de Código facilita a cópia (botão copy no canto)
                st.code(minuta_text, language="markdown")
                
                # 2. BOTÕES DE ACESSO (DIÁLOGOS/POPOVERS)
                st.markdown("---")
                st.write("🔎 **Ferramentas de Revisão:**")
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    dashboard_text = results.get("auditor_dashboard", "")
                    if dashboard_text:
                        with st.popover("🛡️ Ver Auditoria (Compliance)"):
                            st.markdown("### 🛡️ Relatório do Auditor")
                            st.markdown(dashboard_text)
                
                with c2:
                    style_report = results.get("style_report", "")
                    if style_report:
                        with st.popover("🎨 Ver Análise de Estilo"):
                            st.markdown("### 🎨 Dossiê de Estilo Identificado")
                            st.markdown(style_report)

                with c3:
                    with st.popover("🕵️ Ver Detalhes Técnicos"):
                        st.markdown("### ⚙️ Logs da Orquestração")
                        st.json(results.get("steps", {}))
                
                # Salva no histórico (apenas texto simples para não poluir)
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
                llm = get_llm(model_option, api_key=openai_api_key if "gpt" in model_option else None)
                
                # 1. RAG Retrieval: Busca trechos relevantes para a pergunta
                retrieved_docs = st.session_state.retriever.invoke(prompt)
                context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # 2. Montagem do Histórico
                # System Prompt
                chat_history = [SystemMessage(content=LEGAL_ASSISTANT_PROMPT)]
                
                # Adiciona mensagens anteriores (sem o contexto gigante da primeira análise para economizar tokens, 
                # assumindo que o RAG vai trazer o necessário para a pergunta atual)
                # Porém, para manter coerência, talvez seja bom manter as msgs.
                for msg in st.session_state.messages[:-1]: # Tudo menos a última user msg
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
