__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from backend import process_documents, get_vector_store, RAW_DOCS_DIRECTORY, list_documents, process_all_documents
from dotenv import load_dotenv

# Carrega variáveis
load_dotenv()

st.set_page_config(page_title="JurisBusca - Seu Segundo Cérebro", page_icon="⚖️", layout="wide")

# Custom CSS para estética
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-header {
        font-family: 'Helvetica', sans-serif;
        color: #2c3e50;
        text-align: center;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #27ae60;
    }
    .metadata {
        font-size: 0.8em;
        color: #7f8c8d;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚖️ JurisBusca (Nuvem)")
st.markdown("### Busca Semântica Privada em Modelos de Decisão")
st.markdown("Running on: **Local Llama-3 (8B) + MiniLM** 🧠")

# Sidebar para configurações e Upload
with st.sidebar:
    st.header("📚 Ingestão de Documentos")
    uploaded_files = st.file_uploader(
        "Carregar novos modelos (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )

    
    if st.button("Salvar na Base (Staging)"):
        if not uploaded_files:
            st.warning("Por favor, carregue arquivos primeiro.")
        else:
            saved_count = 0
            for uploaded_file in uploaded_files:
                save_path = os.path.join(RAW_DOCS_DIRECTORY, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
            st.success(f"✅ {saved_count} arquivos salvos na base de conhecimento!")

    st.markdown("---")
    st.header("⚙️ Gerenciar Base")
    
    existing_docs = list_documents()
    if existing_docs:
        st.info(f"📚 {len(existing_docs)} documentos na base.")
        with st.expander("Ver arquivos"):
            for doc in existing_docs:
                st.text(doc)
        
        if st.button("Vetorizar Base Completa"):
            with st.spinner("Processando todos os documentos (isso pode demorar, rodando localmente)..."):
                    try:
                        vectorstore, errors = process_all_documents()
                        
                        if errors:
                            st.warning("Alguns arquivos tiveram problemas:")
                            for err in errors:
                                st.error(err)
                        
                        if vectorstore:
                            st.success("✅ Base vetorizada com sucesso!")
                        else:
                            st.error("❌ Nenhum documento pôde ser processado.")
                    except Exception as e:
                        st.error(f"Erro crítico ao vetorizar: {e}")
    else:
        st.info("Nenhum documento na base.")

# Área Principal de Busca
query = st.text_input("🔍 O que você procura? (Ex: 'dano moral atraso voo', 'tutela antecipada saúde')")
search_button = st.button("Buscar")

if query:  # Busca automágica ao digitar ou clicar
    with st.spinner("Pesquisando na base neural..."):
            try:
                vectorstore = get_vector_store()
                # Busca por similaridade
                results = vectorstore.similarity_search_with_score(query, k=5)
                
                if not results:
                    st.info("Nenhum resultado encontrado.")
                else:
                    # GERAÇÃO DA RESPOSTA (RAG)
                    with st.spinner("🤖 Lendo documentos e gerando resposta... (primeira vez pode demorar para baixar o modelo)"):
                        try:
                            from backend import answer_question
                            # Pega apenas os documentos (sem score) para o contexto
                            docs_content = [doc for doc, _ in results]
                            answer = answer_question(query, docs_content)
                            
                            st.markdown("### 🤖 Resposta da IA (Llama 3)")
                            st.success(answer)
                        except Exception as e_gen:
                            st.warning(f"Erro ao gerar resposta: {e_gen}")
                            st.info("Mostrando apenas as referências abaixo.")

                    st.markdown("---")
                    st.subheader("📚 Referências Encontradas")
                    
                    for doc, score in results:
                        source = doc.metadata.get("source", "Desconhecido")
                        filename = os.path.basename(source)
                        page = doc.metadata.get("page", 0) + 1 # PyPDF é 0-indexed
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="metadata">📂 Arquivo: {filename} | 📄 Pág: {page} | 🎯 Distância: {score:.4f}</div>
                            <p>{doc.page_content}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Erro na busca. O banco de dados existe? Detalhes: {e}")
