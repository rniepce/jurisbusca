from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

PROMPT_AUDITOR_AGENT = """
Você é o AUDITOR SÊNIOR (Compliance & Estilo).
Sua missão é validar a MINUTA FINAL antes de ela ser entregue ao Magistrado.

CRITÉRIOS DE AUDITORIA:
1. **Conformidade Fática:** A minuta cita fatos que realmente existem nos dados brutos? (Alucinação Zero).
2. **Conformidade Legal:** A fundamentação jurídica faz sentido?
3. **Conformidade de Estilo:** O texto segue as diretrizes do "Estilo do Juiz"?

ESTILO DO JUIZ (RAG):
{style_guide}

DADOS BRUTOS (DOS AUTOS):
{fatos_json}

MINUTA PARA REVISÃO:
{draft_text}

SAÍDA:
Se estiver tudo OK, retorne apenas: "APROVADO".
Se houver problemas, retorne um RELATÓRIO DE CRÍTICAS pontuando onde precisa corrigir (o Redator usará isso para reescrever).
"""

from langchain_google_genai import ChatGoogleGenerativeAI

def run_auditor_agent(draft_text: str, fatos_json: dict, style_guide: str, keys: dict):
    """
    Executa Auditoria com Fallback:
    1. Tenta GPT-4o (OpenAI).
    2. Se falhar (ex: Quota Limit), tenta Gemini 1.5 Pro (Google).
    """
    openai_key = keys.get("openai")
    google_key = keys.get("google")

    formatted_prompt = PROMPT_AUDITOR_AGENT.format(
        style_guide=style_guide or "Estilo Padrão.",
        fatos_json=str(fatos_json),
        draft_text=draft_text
    )
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content="Valide a minuta.")
    ]

    # 1. TENTATIVA OPENAI (GPT-4o)
    if openai_key:
        try:
            llm = ChatOpenAI(
                api_key=openai_key,
                model="gpt-4o", 
                temperature=0.1
            )
            return llm.invoke(messages).content
        except Exception as e:
            print(f"⚠️ Erro OpenAI (Auditor): {e}. Tentando Fallback para Gemini...")
    
    # 2. TENTATIVA FALLBACK GOOGLE (GEMINI 1.5 PRO)
    if google_key:
        try:
            # Gemini 2.5 Pro (Fallback Robusto)
            llm_fallback = ChatGoogleGenerativeAI(
                google_api_key=google_key,
                model="gemini-2.5-pro",
                temperature=0.1
            )
            response = llm_fallback.invoke(messages).content
            return f"{response}\n\n[NOTA: Auditoria realizada via Gemini 2.5 Pro (Fallback Ativo)]"
        except Exception as e_google:
            return f"Erro Agente Auditor (OpenAI & Google): {str(e_google)}"
            
    return "Erro: Nenhuma chave válida (OpenAI ou Google) para auditoria."
