__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from backend import process_documents, get_vector_store
from dotenv import load_dotenv

# Carrega variáveis
load_dotenv()

st.set_page_config(page_title="JurisBusca (Local) - Seu Segundo Cérebro", page_icon="⚖️", layout="wide")

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

st.title("⚖️ JurisBusca (Modo Local)")
st.markdown("### Busca Semântica Privada em Modelos de Decisão")
st.markdown("Runing on: **MacBook M3 Max** 🚀")

# Sidebar para configurações e Upload
with st.sidebar:
    st.header("📚 Ingestão de Documentos")
    uploaded_files = st.file_uploader(
        "Carregar novos modelos (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Processar Documentos"):
        if not uploaded_files:
            st.warning("Por favor, carregue arquivos primeiro.")
        else:
            with st.spinner("Processando e vetorizando localmente... Isso pode levar um momento."):
                # Salva arquivos temporariamente para o backend processar
                temp_paths = []
                with tempfile.TemporaryDirectory() as temp_dir:
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_paths.append(temp_path)
                    
                    # Chama o backend
                    try:
                        vectorstore = process_documents(temp_paths)
                        st.success(f"✅ {len(temp_paths)} documentos processados com sucesso no banco local!")
                    except Exception as e:
                        st.error(f"Erro ao processar: {e}")

# Área Principal de Busca
query = st.text_input("🔍 O que você procura? (Ex: 'dano moral atraso voo', 'tutela antecipada saúde')")
search_button = st.button("Buscar")

if query:  # Busca automágica ao digitar ou clicar
    with st.spinner("Pesquisando na base neural local..."):
        try:
            vectorstore = get_vector_store()
            # Busca por similaridade
            results = vectorstore.similarity_search_with_score(query, k=5)
            
            if not results:
                st.info("Nenhum resultado encontrado.")
            else:
                for doc, score in results:
                    # Score do Chroma é distância (menor é melhor).
                    # Para visualização, vamos apenas mostrar o score cru ou inverter se necessário.
                    # Modelos de sentence-transformers geralmente usam cosine distance.
                    
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
