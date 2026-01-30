
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from planning_engine import PlanningEngine
from style_engine import StyleEngine
from langchain_core.prompts import ChatPromptTemplate

# --- State Definition ---
class AgentState(TypedDict):
    facts: str # O texto cru ou resumo do RAPTOR
    outline: str # O esqueleto gerado pelo Planner
    draft: str # O texto gerado pelo Writer
    critique: str # Feedback do Critic
    revision_count: int # Contador de loops
    api_key: str # Para passar as chaves
    provider: str # openai ou google

# --- Nodes ---

def planner_node(state: AgentState):
    print("🧠 [PLANNER] Gerando Esqueleto Lógico...")
    planner = PlanningEngine(api_key=state['api_key'], provider=state['provider'])
    # Usa os primeiros 50k chars para planejamento se for muito grande
    summary_for_planning = state['facts'][:50000] 
    outline = planner.generate_outline(summary_for_planning)
    return {"outline": outline, "revision_count": 0}

def writer_node(state: AgentState):
    print(f"✍️ [WRITER] Escrevendo Minuta (Revisão {state.get('revision_count', 0)})...")
    
    # Init LLM
    if state['provider'] == 'openai':
        llm = ChatOpenAI(model="gpt-4o", api_key=state['api_key'], temperature=0.7)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=state['api_key'], temperature=0.7)

    # Contexto de Estilo (Dynamic Few-Shot)
    # Tenta pegar exemplos baseados no Outline (que contem os tópicos)
    style_engine = StyleEngine(api_key=state['api_key'], provider=state['provider'])
    style_prompt = style_engine.get_style_prompt(state['outline'][:10000]) # Query com o outline
    
    style_instructions = ""
    if style_prompt:
        # Extrai exemplos
        try:
             ex_text = ""
             for msg in style_prompt.format_messages(page_content=""):
                 ex_text += f"\n---\n{msg.content}\n"
             if ex_text:
                 style_instructions = f"\n\n## 🎭 CLONAGEM DE ESTILO\nUtilize o tom de voz, vocabulário e estrutura frasal destes exemplos:\n{ex_text}"
        except: pass

    # Contexto de Crítica (se houver rejeição anterior)
    critique_instruction = ""
    if state.get("critique"):
        critique_instruction = f"\n\n🚨 ATENÇÃO: O Crítico REJEITOU a versão anterior pelo seguinte motivo:\n'{state['critique']}'\n\nCORRIJA ISSO IMEDIATAMENTE NA NOVA VERSÃO."

    # Prompt Principal
    prompt_text = f"""
Você é um Assessor Jurídico de Elite (Agente Redator).
Sua missão é escrever a fundamentação da decisão seguindo ESTRITAMENTE o Outline fornecido.

# INPUTS
## 1. FATOS DO PROCESSO
{state['facts']}

## 2. ESQUELETO LÓGICO (OUTLINE)
{state['outline']}

{style_instructions}

{critique_instruction}

# INSTRUÇÕES
- Escreva parágrafo por parágrafo seguindo os tópicos do Outline.
- CITE AS FOLHAS/PÁGINAS sempre que mencionar um fato ou prova (ex: "conforme laudo de fls. 20").
- Se não souber a folha exata, use (fl. citation needed) para o revisor, mas TENTE encontrar no texto dos Fatos.
- NÃO invente fatos. Atenha-se ao que está em 'FATOS DO PROCESSO'.

Escreva a decisão completa agora:
"""
    
    response = llm.invoke([HumanMessage(content=prompt_text)])
    return {"draft": response.content}

def critic_node(state: AgentState):
    print("🧐 [CRITIC] Analisando consistência factual...")
    
    # Init LLM (Critic deve ser rigoroso, temp=0)
    if state['provider'] == 'openai':
        llm = ChatOpenAI(model="gpt-4o", api_key=state['api_key'], temperature=0)
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=state['api_key'], temperature=0)

    prompt_text = f"""
Você é o Revisor Crítico (Agente de Compliance).
Sua única função é validar se a minuta gerada corresponde aos fatos do processo.

# FATOS ORIGINAIS (Referência)
{state['facts']}

# MINUTA GERADA (Para Análise)
{state['draft']}

# TAREFA
1. Verifique se os valores (R$) citados na minuta baterem com os Fatos.
2. Verifique se as citações de fatos existem no texto original.
3. Se houver ALUCINAÇÃO (inventou fato ou valor), REJEITE.
4. Se a minuta estiver genérica demais e fugir do Outline, REJEITE.

Se estiver tudo OK (ou erros mínimos toleráveis), responda apenas: APROVADO
Se houver erro grave, responda: REJEITADO: [Explique o motivo e o que deve ser corrigido]
"""

    response = llm.invoke([HumanMessage(content=prompt_text)])
    review = response.content
    
    if "APROVADO" in review.upper():
        return {"critique": None}
    else:
        return {"critique": review, "revision_count": state['revision_count'] + 1}

# --- Conditional Logic ---
def check_revision(state: AgentState):
    if state.get("critique") is None:
        print("✅ [CRITIC] Minuta Aprovada!")
        return "end"
    
    if state['revision_count'] > 3:
        print("⚠️ [CRITIC] Limite de revisões atingido. Entregando melhor esforço.")
        return "end"
    
    print(f"❌ [CRITIC] Rejeitado. Retornando para Writer. Motivo: {state['critique']}")
    return "retry"

# --- Graph Contruction ---
def create_agent_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    
    workflow.set_entry_point("planner")
    
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "critic")
    
    workflow.add_conditional_edges(
        "critic",
        check_revision,
        {
            "end": END,
            "retry": "writer"
        }
    )
    
    return workflow.compile()
