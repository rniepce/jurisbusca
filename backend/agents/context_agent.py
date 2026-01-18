from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

PROMPT_CONTEXT_AGENT = """
Você é o AGENTE DE CONTEXTO E FATOS (Estagiário Sênior).
Sua missão é ler o conteúdo bruto do processo (PDFs) e extrair os dados estruturados com precisão absoluta.

SAÍDA ESPERADA (JSON):
{
    "fatos_principais": "Resumo cronológico dos fatos...",
    "pedidos_autor": ["Pedido 1", "Pedido 2"],
    "teses_defesa": ["Tese 1", "Tese 2"],
    "datas_chave": {"data_fato": "...", "data_ajuizamento": "..."},
    "provas_citadas": ["Contrato fl. 10", "Email fl. 22"]
}
"""

def run_context_agent(text_content: str, api_key: str):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0.0)
        
        messages = [
            SystemMessage(content=PROMPT_CONTEXT_AGENT),
            HumanMessage(content=f"Analise o seguinte processo e extraia os fatos:\n\n{text_content[:100000]}")
        ]
        
        response = llm.invoke(messages).content
        cleaned = response.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"error": str(e)}
