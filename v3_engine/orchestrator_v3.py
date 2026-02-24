import os
import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import internal tools
try:
    from .tools.legal_repl import LegalREPL
except ImportError:
    from tools.legal_repl import LegalREPL

# Import Prompts 
from prompts_magistrate_v3 import PROMPT_V3_MAGISTRATE_CORE, PROMPT_V3_HYBRID_FALLBACK
from knowledge_base_loader import KNOWLEDGE_BASE

# Internal Imports for LLM
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
import backend as be

# ─── STATE DEFINITION ────────────────────────────────────────────────────────

class MagistrateState(TypedDict):
    raw_text: str
    keys: Dict[str, str]
    repl_tool: LegalREPL
    
    # State evolution through MoE
    condensed_facts: str
    draft_decision: str
    final_json: dict
    
    # Tool looping for DeepSeek
    messages: List[any] 
    iterations: int
    logs: List[str]

# ─── NODES (MIXTURE OF EXPERTS) ────────────────────────────────────────────────

def node_kimi_reader(state: MagistrateState):
    """
    EXPERT 1: O Investigador (Kimi K2.5)
    Lê os autos massivos e extrai o suprassumo fático.
    """
    llm = be.get_llm(model_name="Kimi-K2.5", temperature=0.0)
    
    sys_prompt = """Você é um Investigador Jurídico Senior com memória massiva (Kimi).
Sua missão é ler centenas de páginas de um processo judicial e extrair APENAS os fatos relevantes, os pedidos do autor e as defesas/contestações do réu.
NÃO julgue o mérito. NÃO elabore fundamentação legal. 
Crie um resumo hiper-denso, cronológico e preciso (com datas e valores) para que o Juiz Relator possa decidir depois.
Formate com markdown claro, usando tópicos: "1. Fatos Principais", "2. Pedidos Iniciais", "3. Teses da Defesa"."""

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"AUTOS TOTAIS DO PROCESSO:\n{state['raw_text']}")
    ]
    
    response = llm.invoke(messages)
    condensed = be.safe_content(response)
    
    return {
        "condensed_facts": condensed,
        "logs": state["logs"] + ["🦊 [Kimi K2.5] Investigação concluída. Fatos extraídos da massa de dados."]
    }


def node_deepseek_reasoner(state: MagistrateState):
    """
    EXPERT 2: O Juiz Relator (DeepSeek V3.2)
    Recebe os fatos resumidos e pensa a decisão. Pode usar o REPL para cálculos.
    """
    llm = be.get_llm(model_name="DeepSeek-V3.2-Speciale", temperature=0.1)
    
    @tool
    def run_python_code(code: str) -> str:
        """Executa scripts Python. Acesse o texto cru do processo via variavel 'text'."""
        pass
        
    llm_with_tools = llm.bind_tools([run_python_code])

    if not state["messages"]:
        sys_prompt = f"""Você é um Juiz Relator brilhante (DeepSeek V3.2), mestre em raciocínio lógico forense.
Sua missão é ler o resumo fático abaixo, proferir o julgamento e escrever a fundamentação/decisão.

**Fatos Condensados (Previamente extraídos por seu assistente):**
{state.get("condensed_facts", "")}

Se você precisar validar alguma data específica, cruzar valores monetários, liquidar sentenças, ou fazer buscas exatas (regex) no amontoado caótico original, use a ferramenta 'run_python_code'.
Não gere a minuta em JSON final, apenas redija o VOTO / DECISÃO DE FORMA CLARA E LÓGICA."""
        
        knowledge_section = "\n\n# JURISPRUDÊNCIA LOCAL\n" + KNOWLEDGE_BASE if KNOWLEDGE_BASE else ""
        sys_msg = SystemMessage(content=sys_prompt + knowledge_section)
        
        messages = [sys_msg, HumanMessage(content="Use sua perícia lógica para julgar estes fatos. Se precisar do texto cru das folhas, use o Python.")]
    else:
        messages = state["messages"]

    response = llm_with_tools.invoke(messages)
    
    # Se a resposta nao for tool call (decisão finalizada pelo DeepSeek)
    draft = be.safe_content(response) if not hasattr(response, "tool_calls") or not response.tool_calls else state.get("draft_decision", "")

    return {
        "messages": messages + [response],
        "iterations": state["iterations"] + 1,
        "draft_decision": draft,
        "logs": state["logs"] + ["🐉 [DeepSeek V3.2] Deliberou o mérito / raciocínio lógico."]
    }

def node_computer(state: MagistrateState):
    """
    The Tool Executor. Runs Python code for DeepSeek.
    """
    last_msg = state["messages"][-1]
    tool_outputs = []
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        repl = state.get("repl_tool")
        if not repl:
            repl = LegalREPL(state["raw_text"])
            
        for tool_call in last_msg.tool_calls:
            if tool_call["name"] == "run_python_code":
                code = tool_call["args"].get("code", "")
                result = repl.run_code(code)
                tool_output = f"OBSERVATION (PYTHON):\n{result}"
                tool_outputs.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
                
        return {
            "messages": state["messages"] + tool_outputs,
            "logs": state["logs"] + ["💻 Código executado nativamente (Tool Calling)."]
        }
    
    # Regex Fallback
    content = last_msg.content
    tool_output = "NO_CODE_FOUND"
    
    if isinstance(content, str) and ("```python" in content.lower() or "```" in content):
        import re
        code_match = re.search(r"```(?:python)?(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if code_match:
            code = code_match.group(1).strip().replace("=>", "=")           
            if code:
                repl = state.get("repl_tool")
                if not repl:
                    repl = LegalREPL(state["raw_text"])
                result = repl.run_code(code)
                tool_output = f"OBSERVATION (PYTHON):\n{result}"
            else:
                tool_output = "ERROR: Code block was empty."
        else:
            tool_output = "ERROR: Failed to parse Python block."
            
    return {
        "messages": state["messages"] + [HumanMessage(content=tool_output)],
        "logs": state["logs"] + ["💻 Código executado via Regex Fallback."]
    }

def node_gpt_formatter(state: MagistrateState):
    """
    EXPERT 3: O Assessor Sênior (GPT-5.2)
    Formata o rascunho de decisão no JSON estrito.
    """
    llm = be.get_llm(model_name="gpt-5.2-chat", temperature=0.0)
    
    sys_prompt = f"""Você é um Assessor Sênior rigoroso e formal (GPT-5.2).
Sua ÚNICA missão é envelopar a Decisão do Juiz Relator num formato JSON estritamente tipado.
NÃO invente novos fatos nem mude o mérito. 

Seu JSON deve seguir exatamente a seguinte estrutura:
```json
{{
  "minuta_final": "TEXTO DA DECISÃO AQUI, FORMATADO EM MARKDOWN E PRONTO PARA USO",
  "pontos_chave": ["ponto 1", "ponto 2"]
}}
```"""

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"DECISÃO DO JUIZ RELATOR:\n\n{state['draft_decision']}\n\nEnvolva isso no JSON estrito.")
    ]
    
    response = llm.invoke(messages)
    content = be.safe_content(response)
    
    return {
        "draft_decision": content,  # Reuse just to carry over final text to extract
        "logs": state["logs"] + ["🤖 [GPT-5.2] Formatação JSON e conformidade de estilo concluída."]
    }


# ─── EDGES ─────────────────────────────────────────────────────────────────────

def should_continue_reasoning(state: MagistrateState):
    """
    Decide se o DeepSeek quer rodar código ou se já fechou o raciocínio.
    """
    last_msg = state["messages"][-1]
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "computer"
        
    content = last_msg.content
    if isinstance(content, str):
        if "```python" in content:
            return "computer"
            
    if state["iterations"] > 5:
        return "formatter" # Fail safe: move to format
        
    return "formatter"

# ─── GRAPH CONSTRUCTION ────────────────────────────────────────────────────────

def build_v3_graph():
    workflow = StateGraph(MagistrateState)
    
    workflow.add_node("reader", node_kimi_reader)
    workflow.add_node("reasoner", node_deepseek_reasoner)
    workflow.add_node("computer", node_computer)
    workflow.add_node("formatter", node_gpt_formatter)
    
    workflow.set_entry_point("reader")
    
    # Kimi (Reader) -> DeepSeek (Reasoner)
    workflow.add_edge("reader", "reasoner")
    
    # DeepSeek (Reasoner) loops with Computer, or goes to Formatter
    workflow.add_conditional_edges(
        "reasoner",
        should_continue_reasoning,
        {
            "computer": "computer",
            "formatter": "formatter"
        }
    )
    workflow.add_edge("computer", "reasoner")
    
    # Formatter -> End
    workflow.add_edge("formatter", END)
    
    return workflow.compile()

def run_autonomous_magistrate(text: str, keys: dict, model_name: str = "v3-moe"):
    """
    Entry point for V3 - Mixture of Experts.
    (model_name is ignored as MoE uses specific models natively)
    """
    app = build_v3_graph()
    
    repl = LegalREPL(text)
    
    initial_state = {
        "raw_text": text,
        "keys": keys,
        "repl_tool": repl,
        "condensed_facts": "",
        "draft_decision": "",
        "messages": [],
        "iterations": 0,
        "final_json": {},
        "logs": ["🚀 Iniciando V3 Mixture of Experts (Kimi -> DeepSeek -> GPT)"]
    }
    
    final_state = app.invoke(initial_state)
    
    # Extract Final JSON from GPT formatter output
    last_content = final_state["draft_decision"]
    try:
        import re
        json_match = re.search(r"```json(.*?)```", last_content, re.DOTALL)
        if json_match:
            final_json = json.loads(json_match.group(1).strip())
            return final_json, final_state["logs"]
        else:
            return {"error": "No JSON found in final output", "raw": last_content}, final_state["logs"]
    except Exception as e:
        return {"error": f"JSON Parse Error: {str(e)}", "raw": last_content}, final_state["logs"]
