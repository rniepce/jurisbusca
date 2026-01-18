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

def run_auditor_agent(draft_text: str, fatos_json: dict, style_guide: str, api_key: str):
    try:
        if not api_key:
            return "Erro: Chave OpenAI não fornecida."

        llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o", # ou gpt-5 se disponível via alias
            temperature=0.1
        )
        
        formatted_prompt = PROMPT_AUDITOR_AGENT.format(
            style_guide=style_guide or "Estilo Padrão.",
            fatos_json=str(fatos_json),
            draft_text=draft_text
        )
        
        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content="Valide a minuta.")
        ]
        
        response = llm.invoke(messages).content
        return response
    except Exception as e:
        return f"Erro no Agente Auditor: {str(e)}"
