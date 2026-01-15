import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Diretório para persistência do banco vetorial
PERSIST_DIRECTORY = "./chroma_db"

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
    """Retorna a função de embedding local (multilíngue)."""
    # Modelo leve e eficiente para português: paraphrase-multilingual-MiniLM-L12-v2
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def process_documents(file_paths: List[str]) -> Chroma:
    """
    Carrega arquivos, divide em chunks e salva no ChromaDB.
    Retorna o objeto vectorstore.
    """
    all_docs = []
    
    for path in file_paths:
        try:
            docs = load_document(path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Erro ao carregar {path}: {e}")
            
    # Divisão de texto (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(all_docs)
    
    # Criação/Atualização do Banco Vetorial
    embedding_function = get_embedding_function()
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    return vectorstore

def get_vector_store():
    """Carrega o banco vetorial existente."""
    embedding_function = get_embedding_function()
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function
    )
    return vectorstore
