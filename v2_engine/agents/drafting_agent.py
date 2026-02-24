import backend as be
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
import os

PROMPT_DRAFTING_AGENT = """
# FUNÇÃO: REDATOR CHEFE (ASSESSOR LITERÁRIO — GHOSTWRITER JUDICIAL)
Sua missão é transformar o RELATÓRIO PRÉ-SENTENÇA / RELATÓRIO DE TRIAGEM em uma MINUTA JURÍDICA perfeita, pronta para assinatura do Magistrado.

## INSTRUÇÕES DE BUSCA DE MODELOS (AGENTE RECHERCHEUX)
Antes de redigir a minuta, você **DEVE OBRIGATORIAMENTE** utilizar a ferramenta `search_templates_database` para procurar por um modelo (template) adequado ao caso específico tratado no Relatório.
- Busque por palavras-chave do tema central (ex: "Dano Moral Voo", "Revisional de Juros", "Ato Ordinatório Intimação").
- Se a ferramenta retornar um modelo, utilize a ESTRUTURA, o TOM e a FUNDAMENTAÇÃO ABSTRATA desse modelo, inserindo os dados fáticos do caso atual.
- Se a ferramenta NÃO retornar um modelo (ou retornar vazio), você deve usar a estrutura de Template Padrão abaixo, adaptando-a estritamente para que o **ESTILO DE ESCRITA SEJA IDÊNTICO ao estilo de escrita do Relatório de Triagem** fornecido (tom, vocabulário, objetividade).

## REGRAS DE OURO
### 1. RASTREABILIDADE DE IDs (OBRIGATÓRIA)
Nunca mencione um documento (Contrato, Petição, Laudo) sem citar o ID do PJe, conforme apontado no relatório.
- Se o relatório não forneceu ID, use: "conforme documento acostado aos autos".

### 2. FUNDAMENTAÇÃO ANALÍTICA (Art. 489 §1º CPC)
- Explique a relação entre a norma citada e o caso concreto.
- Enfrente CADA argumento relevante da parte vencida com refutação específica (conforme o relatório).
- Evite fundamentação genérica.

### 3. ANTI-ALUCINAÇÃO
- NÃO invente IDs, números de processo, nomes ou valores que não estejam no relatório.
- NÃO cite jurisprudência específica a menos que esteja na Base de Conhecimento ou no Relatório.

## ESTRUTURA DA SENTENÇA/ATO (Se não houver modelo específico)
1. **CABEÇALHO:** Comarca, Vara, Número do Processo (se disponível aos autos).
2. **RELATÓRIO:** Breve histórico processual.
3. **FUNDAMENTAÇÃO:** Desenvolva o raciocínio do esboço em linguagem jurídica culta e persuasiva.
4. **DISPOSITIVO:** A conclusão formal ("Ante o exposto...", etc). Formule o dispositivo exatamente na direção que o relatório determinou.

## SAÍDA Esperada
Retorne APENAS o texto completo da Minuta (Sentença, Decisão ou Despacho). Não adicione saudações ou explicações extras.
"""

# Tool simulada para busca de modelos. No futuro pode conectar a um banco vetorial ou diretório real.
@tool
def search_templates_database(query: str) -> str:
    """
    Busca no banco de dados de modelos do tribunal por templates de sentenças ou decisões.
    Passe uma query objetiva (ex: "Atraso de voo dano moral", "Abertura de vista").
    """
    # Exemplo mock simples
    query_lower = query.lower()
    if "voo" in query_lower or "atraso" in query_lower:
        return """
        MODELO: SENTENÇA - ATRASO DE VOO
        [RELATÓRIO]
        [FUNDAMENTAÇÃO] Aplica-se o Código de Defesa do Consumidor. A falha na prestação do serviço de transporte aéreo...
        [DISPOSITIVO] Ante o exposto, julgo PROCEDENTES...
        """
    elif "ato ordinário" in query_lower or "ordinatório" in query_lower:
        return """
        MODELO: ATO ORDINATÓRIO PADRÃO
        De ordem do MM. Juiz de Direito, nos termos do art. 64 do Provimento 355/2018...
        """
    return "Nenhum modelo encontrado para esta pesquisa. Proceda utilizando o template padrão alinhado ao estilo do relatório."

def run_drafting_agent(triage_report: str, api_key: str = None) -> str:
    """
    Agente de Redação (Stage 2) - Powered by Azure OpenAI (GPT-5.2)
    Busca modelos e escreve a minuta.
    """
    try:
        llm = be.get_llm("gpt-5.2-chat", temperature=0.3)
        
        tools = [search_templates_database]
        llm_with_tools = llm.bind_tools(tools)
        
        system_message = SystemMessage(content=PROMPT_DRAFTING_AGENT)
        user_message = HumanMessage(content=f"RELATÓRIO DE TRIAGEM / PRÉ-SENTENÇA:\n\n{triage_report}\n\nPor favor, busque possíveis modelos e escreva a minuta.")
        
        messages = [system_message, user_message]
        
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Tool execution loop
        for _ in range(2):
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "search_templates_database":
                        tool_output = search_templates_database.invoke(tool_call["args"])
                        messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                
                response = llm_with_tools.invoke(messages)
                messages.append(response)
            else:
                break
                
        return response.content

    except Exception as e:
        return f"Erro Crítico no Drafting Agent: {str(e)}"
