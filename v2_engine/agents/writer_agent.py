from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

PROMPT_WRITER_AGENT = """
Você é o REDATOR CHEFE (Assessor Literário).
Sua missão é transformar o ESBOÇO LÓGICO em uma MINUTA JURÍDICA PERFEITA.

CARACTERÍSTICAS DE ESTILO:
- Tom: Formal, sóbrio e direto.
- Estrutura: Relatório (breve), Fundamentação (robusta), Dispositivo (claro).
- Use o "Estilo do Juiz" abaixo se fornecido.

ESTILO DO JUIZ (RAG):
{style_guide}

ESBOÇO LÓGICO (DO JUIZ AUXILIAR):
{verdict_outline}

REGRAS DE OURO (RASTREABILIDADE):
1. CITAÇÃO DE ID É OBRIGATÓRIA: Nunca mencione um documento (Contrato, Petição, Laudo) sem citar o ID do PJe.
   - Errado: "O contrato prevê..."
   - Certo: "O contrato de adesão anexado ao ID 123456 prevê..."
2. Se o ID não foi fornecido no resumo, use "[ID NÃO LOCALIZADO]" para alertar, mas não invente.
"""

def run_writer_agent(verdict_outline: str, style_guide: str, api_key: str):
    try:
        if not api_key:
            return "Erro: Chave Anthropic não fornecida."

        llm = ChatAnthropic(
            api_key=api_key,
            model="claude-sonnet-4-5-20250929",
            temperature=0.3
        )
        
        formatted_prompt = PROMPT_WRITER_AGENT.format(
            style_guide=style_guide or "Estilo Padrão do Tribunal.",
            verdict_outline=verdict_outline
        )
        
        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content="Escreva a Minuta Final agora.")
        ]
        
        response = llm.invoke(messages).content
        return response
    except Exception as e:
        return f"Erro no Agente Redator: {str(e)}"
