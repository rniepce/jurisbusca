import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Diretório para persistência do banco vetorial
# Diretório para persistência do banco vetorial e documentos raw
PERSIST_DIRECTORY = "./chroma_db"
RAW_DOCS_DIRECTORY = "./raw_documents_db"

if not os.path.exists(RAW_DOCS_DIRECTORY):
    os.makedirs(RAW_DOCS_DIRECTORY)

def load_document(file_path: str) -> List[Document]:
    """Carrega um documento baseado na extensão do arquivo."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {ext}")
        
    return loader.load()

def get_embedding_function():
    """Retorna a função de embedding local (HuggingFace)."""
    # Modelo leve e eficiente para português
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def process_documents(file_paths: List[str]):
    """
    Carrega arquivos, divide em chunks e salva no ChromaDB.
    Retorna o objeto vectorstore e uma lista de erros.
    """
    all_docs = []
    errors = []
    
    for path in file_paths:
        try:
            docs = load_document(path)
            if not docs:
                errors.append(f"Aviso: Nenhum texto encontrado em {os.path.basename(path)}")
            else:
                all_docs.extend(docs)
        except Exception as e:
            errors.append(f"Erro ao carregar {os.path.basename(path)}: {str(e)}")
            
    if not all_docs:
        # Se não houver documentos, retornamos None e os erros
        return None, errors

    # Divisão de texto (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(all_docs)

    if not splits:
        raise ValueError("Nenhum texto pôde ser extraído dos arquivos carregados. Verifique se não são imagens escaneadas sem OCR.")
    
    # Criação/Atualização do Banco Vetorial
    embedding_function = get_embedding_function()
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    return vectorstore, errors

def get_vector_store():
    """Carrega o banco vetorial existente."""
    embedding_function = get_embedding_function()
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function
    )
    return vectorstore

def list_documents() -> List[str]:
    """Lista os arquivos salvos no diretório raw."""
    if not os.path.exists(RAW_DOCS_DIRECTORY):
        return []
    return [f for f in os.listdir(RAW_DOCS_DIRECTORY) if os.path.isfile(os.path.join(RAW_DOCS_DIRECTORY, f)) and not f.startswith(".")]

def process_all_documents():
    """Processa todos os documentos do diretório raw."""
    files = list_documents()
    file_paths = [os.path.join(RAW_DOCS_DIRECTORY, f) for f in files]
    
    if not file_paths:
        raise ValueError("Nenhum documento para processar.")
        
    return process_documents(file_paths)

# Nova função para carregar LLM Local (Mistral NeMo 12B)
def get_llm_function():
    from langchain_community.llms import CTransformers
    
    # Mistral NeMo 12B (NVIDIA + Mistral)
    # Usando quantização Q4_K_M que balanceia bem qualidade/tamanho (aprox 8GB).
    llm = CTransformers(
        model="bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        model_file="Mistral-Nemo-Instruct-2407-Q4_K_M.gguf", 
        model_type="mistral",
        config={'max_new_tokens': 2048, 'temperature': 0.1, 'context_length': 16384}
    )
    return llm

def answer_question(query, docs):
    """Gera uma resposta baseada nos documentos encontrados usando LLM Local."""
    llm = get_llm_function()
    
    # Prepara o contexto
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # Prompt Template para Mistral (ChatML style é comum, mas o simples [INST] funciona bem)
    prompt = f"""[INST] Você é um assistente jurídico especialista. Baseie-se APENAS no contexto fornecido abaixo para responder à pergunta do usuário.
Se a resposta não estiver no contexto, diga claramente que não encontrou a informação.

Contexto:
{context_text}

Pergunta: {query} [/INST]"""
    
    response = llm.invoke(prompt)
    return response
