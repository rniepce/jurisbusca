
import os
import json
import sys
from tqdm import tqdm
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# Importa funções do backend existente
sys.path.append(os.getcwd())
from backend import clean_text, get_llm

def load_document(file_path: str):
    """Carrega um documento dependendo da extensão."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Formato não suportado: {ext}")
    return loader.load()

RAW_DATA_DIR = "data/raw_modelos"
OUTPUT_FILE = "data/train.jsonl"

def generate_synthetic_instruction(doc_text, filename):
    """
    Usa o LLM local para criar uma instrução de usuário plausível para este documento.
    Ex: "Redija uma contestação alegando X e Y" -> Documento
    """
    try:
        # Tenta usar Ollama (mistral-nemo ou llama3)
        llm = get_llm(model_name="mistral-nemo") # Prefer Mistral Nemo for better reasoning
        
        prompt = f"""
        Você é um assistente especialista em criação de datasets para LLMs jurídicos.
        
        TAREFA:
        Analise o texto legislativo/jurídico abaixo (que é a RESPOSTA/OUTPUT desejado) e crie o PROMPT DO USUÁRIO (Input) que geraria exatamente este texto.
        
        O Prompt do Usuário deve ser direto, como um advogado pedindo ao estagiário ou à IA.
        Ex: "Escreva uma ação de dano moral por atraso de voo citando o CDC."
        
        CONTEXTO DO ARQUIVO: {filename}
        
        TEXTO ALVO (Inicio):
        {doc_text[:3000]}... [Texto truncado]
        
        RESPOSTA ESPERADA:
        Apenas o texto do Prompt do Usuário. Sem aspas, sem explicações extras.
        """
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        print(f"⚠️  Erro ao gerar instrução sintética para {filename}: {e}")
        return f"Escreva um documento jurídico baseado no arquivo {filename}."

def main():
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"📁 Diretório '{RAW_DATA_DIR}' criado. Por favor, coloque seus arquivos PDF/DOCX lá.")
        return

    files = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
    
    if not files:
        print(f"⚠️  Nenhum arquivo encontrado em '{RAW_DATA_DIR}'. Adicione arquivos para prosseguir.")
        return

    import random
    
    print(f"🚀 Iniciando extração de {len(files)} arquivos...")
    
    all_entries = []
    
    for filename in tqdm(files):
        file_path = os.path.join(RAW_DATA_DIR, filename)
        
        try:
            # 1. Extração e Limpeza
            docs = load_document(file_path)
            full_text = "\n\n".join([d.page_content for d in docs])
            cleaned_text = clean_text(full_text)
            
            if len(cleaned_text) < 100:
                print(f"⏭️  Pulando {filename} (Texto muito curto/vazio).")
                continue
            
            # 2. Geração da Instrução (Prompt Sintético)
            instruction = generate_synthetic_instruction(cleaned_text, filename)
            
            # 3. Formatação Chat (Role: User -> Assistant)
            # Formato MLX compatível
            entry = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": cleaned_text}
                ]
            }
            all_entries.append(entry)
            
        except Exception as e:
            print(f"❌ Erro em {filename}: {e}")

    # Divisão Train/Validation (90/10)
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * 0.9)
    train_data = all_entries[:split_idx]
    valid_data = all_entries[split_idx:]
    
    # Se tiver muito pouco dado, garante pelo menos 1 no valid se possível
    if not valid_data and train_data:
        valid_data = [train_data.pop()]

    # Salvar Arquivos
    with open("data/train.jsonl", 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open("data/valid.jsonl", 'w', encoding='utf-8') as f:
        for entry in valid_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Dataset concluído!")
    print(f"📊 Train: {len(train_data)} itens | Valid: {len(valid_data)} itens")
    print(f"Arquivos salvos em 'data/train.jsonl' e 'data/valid.jsonl'")
    print("Pronto para Fine-Tuning com MLX.")

if __name__ == "__main__":
    main()
