import os
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END

# Import Agents
from .agents.triage_agent import run_triage_agent
from .agents.drafting_agent import run_drafting_agent
from .agents.revision_agent import run_revision_agent

# Define State
class AgentState(TypedDict):
    raw_text: str
    keys: Dict[str, str] # {google, openai, anthropic, deepseek}
    knowledge_base: str
    
    # Internal State
    triage_report: str
    draft_text: str
    revision_dashboard: str
    logs: List[str]

# Nodes
def node_triage(state: AgentState):
    log = "🔄 [1/3] Assistente de Gabinete (Claude): Realizando triagem e roteamento..."
    try:
        report = run_triage_agent(state["raw_text"], state["keys"].get("anthropic"), state["knowledge_base"])
        return {"triage_report": report, "logs": state.get("logs", []) + [log]}
    except Exception as e:
        return {"logs": state.get("logs", []) + [f"❌ Erro Triagem: {str(e)}"]}

def node_drafting(state: AgentState):
    log = "✍️ [2/3] Redator Chefe (Claude): Buscando modelos e escrevendo minuta..."
    try:
        draft = run_drafting_agent(state["triage_report"], state["keys"].get("anthropic"))
        return {"draft_text": draft, "logs": state["logs"] + [log]}
    except Exception as e:
        return {"logs": state["logs"] + [f"❌ Erro Redação: {str(e)}"]}

def node_revision(state: AgentState):
    log = "🛡️ [3/3] Auditor QA (Claude): Avaliando conformidade legal, fática e de eficiência..."
    try:
        dashboard = run_revision_agent(
            state["triage_report"], 
            state["draft_text"], 
            state["keys"].get("anthropic"),
            state["knowledge_base"]
        )
        return {
            "revision_dashboard": dashboard, 
            "logs": state["logs"] + [log]
        }
    except Exception as e:
        return {
            "logs": state["logs"] + [f"❌ Erro Revisão (QA): {str(e)}"]
        }

# Graph Construction
def build_v2_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("triage", node_triage)
    workflow.add_node("drafting", node_drafting)
    workflow.add_node("revision", node_revision)
    
    def route_after_triage(state: AgentState):
        """Circuit breaker: abort pipeline if triage produced no output."""
        if not state.get("triage_report", "").strip():
            return END
        return "drafting"

    workflow.set_entry_point("triage")

    workflow.add_conditional_edges("triage", route_after_triage, {"drafting": "drafting", END: END})
    workflow.add_edge("drafting", "revision")
    workflow.add_edge("revision", END)
    
    return workflow.compile()

def run_hybrid_orchestration(text: str, keys: dict, style_guide: str = ""):
    """
    Function to be called from backend.py.
    Returns a normalized dict compatible with app.py expectations.
    (Note: style_guide is repurposed here as knowledge_base/templates where needed, 
    but the main knowledge_base could also be fetched from another source).
    """
    app = build_v2_graph()
    initial_state = {
        "raw_text": text,
        "keys": keys,
        "knowledge_base": style_guide, # utilizing style_guide parameter as knowledge_base proxy if provided
        "triage_report": "",
        "draft_text": "",
        "revision_dashboard": "",
        "logs": []
    }
    
    result = app.invoke(initial_state)
    
    # Normalize output to match what app.py expects
    normalized = {
        "final_output": result.get("draft_text", ""),
        "audit_report": result.get("revision_dashboard", ""),
        "draft_text": result.get("draft_text", ""),
        "logs": result.get("logs", []),
        "final_report": result.get("triage_report", ""),
        "auditor_dashboard": result.get("revision_dashboard", ""),
        "steps": result.get("logs", []),
    }
    return normalized
