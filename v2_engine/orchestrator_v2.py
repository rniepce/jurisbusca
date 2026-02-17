import os
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END

# Import Agents
from .agents.context_agent import run_context_agent
from .agents.reasoning_agent import run_reasoning_agent
from .agents.writer_agent import run_writer_agent
from .agents.auditor_agent import run_auditor_agent

# Import Fixer Prompt
try:
    from prompts_gemini import PROMPT_GEMINI_FIXER
except ImportError:
    PROMPT_GEMINI_FIXER = """
    # PROMPT: EDITOR DE CORREÇÃO (SELF-CORRECTION)
    
    ## 1. CONTEXTO
    Você é um Editor Sênior.
    O Estagiário (Modelo Anterior) escreveu uma minuta, mas o Auditor encontrou ERROS.
    
    ## 2. INSUMOS
    [MINUTA ORIGINAL (COM ERROS)]:
    {draft}
    
    [RELATÓRIO DE ERROS DO AUDITOR]:
    {critique}
    
    ## 3. SUA MISSÃO
    Reescreva a minuta corrigindo APENAS os pontos apontados pelo Auditor.
    - Se o ID não existe, remova a menção ao ID ou substitua por "conforme documento anexo".
    - NÃO MUDE O ESTILO. Mantenha a estrutura, apenas corrija a verdade dos fatos.
    
    ## 4. SAÍDA
    Retorne APENAS o texto completo da Minuta Corrigida.
    """

# Define State
class AgentState(TypedDict):
    raw_text: str
    keys: Dict[str, str] # {google, openai, anthropic, deepseek}
    style_guide: str
    
    # Internal State
    fatos_json: dict
    verdict_outline: str
    draft_text: str
    audit_report: str
    audit_passed: bool
    fix_attempts: int
    final_output: str
    logs: List[str]

# Nodes
def node_context(state: AgentState):
    log = "🔄 [1/4] Estagiário (Gemini): Lendo e extraindo fatos..."
    try:
        fatos = run_context_agent(state["raw_text"], state["keys"]["google"])
        return {"fatos_json": fatos, "logs": state.get("logs", []) + [log]}
    except Exception as e:
        return {"logs": state.get("logs", []) + [f"❌ Erro Contexto: {str(e)}"]}

def node_reasoning(state: AgentState):
    log = "⚖️ [2/4] Juiz Auxiliar (DeepSeek): Definindo estratégia..."
    try:
        outline = run_reasoning_agent(state["fatos_json"], state["keys"]["deepseek"])
        return {"verdict_outline": outline, "logs": state["logs"] + [log]}
    except Exception as e:
         return {"logs": state["logs"] + [f"❌ Erro Raciocínio: {str(e)}"]}

def node_writer(state: AgentState):
    log = "✍️ [3/4] Redator (Claude): Escrevendo minuta..."
    try:
        draft = run_writer_agent(state["verdict_outline"], state["style_guide"], state["keys"]["anthropic"])
        return {"draft_text": draft, "logs": state["logs"] + [log]}
    except Exception as e:
        return {"logs": state["logs"] + [f"❌ Erro Redação: {str(e)}"]}

def node_auditor(state: AgentState):
    log = "🛡️ [4/4] Auditor (GPT-4o): Validando conformidade..."
    try:
        audit = run_auditor_agent(state["draft_text"], state["fatos_json"], state["style_guide"], state["keys"])
        
        # Determine if audit passed based on the structured veredito
        audit_upper = audit.upper()
        passed = "APROVADO" in audit_upper and "REPROVADO" not in audit_upper
        
        if passed:
            log += " ✅ (Aprovado)"
        elif "COM RESSALVAS" in audit_upper:
            log += " ⚠️ (Com Ressalvas)"
            passed = True  # Ressalvas are acceptable, fixer handles REPROVADO only
        else:
            log += " ❌ (Reprovado — encaminhando para correção)"
            
        return {
            "audit_report": audit, 
            "audit_passed": passed,
            "final_output": state["draft_text"],
            "logs": state["logs"] + [log]
        }
    except Exception as e:
        return {
            "audit_passed": True,  # Don't block on audit errors
            "logs": state["logs"] + [f"❌ Erro Auditoria: {str(e)}"]
        }

def node_fixer(state: AgentState):
    """
    Correction Node: Uses PROMPT_GEMINI_FIXER to rewrite the draft
    based on the auditor's critique. Only called when audit fails.
    """
    log = "🔧 [FIX] Editor de Correção: Reescrevendo minuta com base na auditoria..."
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        
        google_key = state["keys"].get("google")
        if not google_key:
            return {
                "logs": state["logs"] + [f"{log} ❌ Sem chave Google para correção."],
                "audit_passed": True  # Skip fix, deliver as-is
            }
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview", 
            google_api_key=google_key, 
            temperature=0.1
        )
        
        formatted_prompt = PROMPT_GEMINI_FIXER.format(
            draft=state["draft_text"],
            critique=state["audit_report"]
        )
        
        response = llm.invoke([HumanMessage(content=formatted_prompt)])
        corrected_draft = response.content
        
        fix_attempts = state.get("fix_attempts", 0) + 1
        
        return {
            "draft_text": corrected_draft,
            "fix_attempts": fix_attempts,
            "logs": state["logs"] + [f"{log} ✅ (Tentativa {fix_attempts})"]
        }
    except Exception as e:
        return {
            "audit_passed": True,  # Don't block on fixer errors
            "logs": state["logs"] + [f"{log} ❌ Erro: {str(e)}"]
        }

def should_fix_or_finish(state: AgentState):
    """
    Edge Condition after Auditor:
    - If audit passed -> END
    - If audit failed and fix_attempts < 1 -> FIXER
    - If fix_attempts >= 1 -> END (prevent infinite loops)
    """
    if state.get("audit_passed", True):
        return "end"
    
    if state.get("fix_attempts", 0) >= 1:
        return "end"  # Max 1 retry to prevent infinite loops
    
    return "fixer"

# Graph Construction
def build_v2_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("context", node_context)
    workflow.add_node("reasoning", node_reasoning)
    workflow.add_node("writer", node_writer)
    workflow.add_node("auditor", node_auditor)
    workflow.add_node("fixer", node_fixer)
    
    workflow.set_entry_point("context")
    
    workflow.add_edge("context", "reasoning")
    workflow.add_edge("reasoning", "writer")
    workflow.add_edge("writer", "auditor")
    
    # Conditional: Auditor decides if we fix or finish
    workflow.add_conditional_edges(
        "auditor",
        should_fix_or_finish,
        {
            "fixer": "fixer",
            "end": END
        }
    )
    
    # After fixing, re-audit
    workflow.add_edge("fixer", "auditor")
    
    return workflow.compile()

def run_hybrid_orchestration(text: str, keys: dict, style_guide: str = ""):
    """
    Function to be called from backend.py.
    Returns a normalized dict compatible with app.py expectations.
    """
    app = build_v2_graph()
    initial_state = {
        "raw_text": text,
        "keys": keys,
        "style_guide": style_guide,
        "audit_passed": False,
        "fix_attempts": 0,
        "logs": []
    }
    
    result = app.invoke(initial_state)
    
    # Normalize output to match what app.py expects
    normalized = {
        "final_output": result.get("final_output", ""),
        "audit_report": result.get("audit_report", ""),
        "draft_text": result.get("draft_text", ""),
        "logs": result.get("logs", []),
        "final_report": result.get("final_output", ""),
        "auditor_dashboard": result.get("audit_report", ""),
        "steps": result.get("logs", []),
    }
    return normalized
