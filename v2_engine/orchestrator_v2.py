import os
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END

# Import Agents
from .agents.context_agent import run_context_agent
from .agents.reasoning_agent import run_reasoning_agent
from .agents.writer_agent import run_writer_agent
from .agents.auditor_agent import run_auditor_agent

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
        
        # Lógica Simples: Se aprovado, finaliza. Se não, idealmente voltaria (loop), 
        # mas no MVP vamos apenas anexar o relatório de auditoria à minuta.
        final_text = state["draft_text"]
        
        if "APROVADO" not in audit.upper():
            # final_text += "\n\n--- 🛡️ NOTAS DE AUDITORIA ---\n" + audit # REMOVIDO: Auditoria fica separada
            log += " (Com ressalvas)"
        else:
            log += " (Aprovado)"
            
        return {"audit_report": audit, "final_output": final_text, "logs": state["logs"] + [log]}
    except Exception as e:
        return {"logs": state["logs"] + [f"❌ Erro Auditoria: {str(e)}"]}

# Graph Construction
def build_v2_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("context", node_context)
    workflow.add_node("reasoning", node_reasoning)
    workflow.add_node("writer", node_writer)
    workflow.add_node("auditor", node_auditor)
    
    workflow.set_entry_point("context")
    
    workflow.add_edge("context", "reasoning")
    workflow.add_edge("reasoning", "writer")
    workflow.add_edge("writer", "auditor")
    workflow.add_edge("auditor", END)
    
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
