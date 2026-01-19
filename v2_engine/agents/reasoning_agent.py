from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

PROMPT_REASONING_AGENT = """
Você é o AGENTE DE RACIOCÍNIO E ESTRATÉGIA (Juiz Auxiliar).
Sua missão é analisar os FATOS e a LEGISLAÇÃO para decidir o mérito.

DADOS DO CASO:
{fatos_json}

INSTRUÇÃO:
1. Analise cada pedido do autor frente às teses da defesa.
2. Aplique a jurisprudência padrão (assumida).
3. Decida: PROCEDENTE, IMPROCEDENTE ou PARCIALMENTE PROCEDENTE.
4. Explique o "Porquê" lógico (Chain of Thought).

SAÍDA:
Retorne apenas o ESBOÇO LÓGICO DA DECISÃO (Tópicos). Não escreva a sentença final ainda.
"""

def run_reasoning_agent(fatos_json: dict, api_key: str):
    try:
        # Configuração DeepSeek via interface compatível OpenAI
        # Se a chave for vazia, fallback para OpenAI ou erro
        if not api_key:
            return "Erro: Chave DeepSeek não fornecida."

        llm = ChatOpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com", 
            model="deepseek-reasoner", 
            temperature=0.2
        )
        
        formatted_prompt = PROMPT_REASONING_AGENT.format(fatos_json=str(fatos_json))
        
        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content="Gere o esboço da decisão com base nos fatos acima.")
        ]
        
        response = llm.invoke(messages).content
        return response
    except Exception as e:
        return f"Erro no Agente de Raciocínio: {str(e)}"
