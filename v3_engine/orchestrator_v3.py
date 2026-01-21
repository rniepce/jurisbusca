import os
import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import internal tools
# Adjust import path based on execution context if needed
try:
    from .tools.legal_repl import LegalREPL
except ImportError:
    from tools.legal_repl import LegalREPL

# Import Prompts
# Assuming prompts_magistrate_v3 is in root or accessible
try:
    from prompts_magistrate_v3 import PROMPT_V3_MAGISTRATE_CORE, PROMPT_V3_HYBRID_FALLBACK
except ImportError:
    # Fallback or local import if necessary
    from prompts_magistrate_v3 import PROMPT_V3_MAGISTRATE_CORE
    # Define fallback prompt if not imported
    PROMPT_V3_HYBRID_FALLBACK = """
    # MODO HÍBRIDO (CODE FIRST)
    1. Tente encontrar a informação via CÓDIGO (search_dates, grep).
    2. Se retornar 'NOT_FOUND', use sua LEITURA SEMÂNTICA para encontrar a resposta no texto.
    """

# Internal Imports for LLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Define State
class MagistrateState(TypedDict):
    raw_text: str
    keys: Dict[str, str]
    repl_tool: LegalREPL # Object, not serializable usually, but fine for in-mem graph
    
    # Conversation
    messages: List[any] # Chat history
    iterations: int
    final_json: dict
    logs: List[str]

# --- NODES ---

def node_magistrate(state: MagistrateState):
    """
    The Brain. Decides whether to use Code Tool or Finalize.
    """
    keys = state["keys"]
    
    # 1. Select Model (Prefer Gemini 1.5 Pro for massive context + tool use)
    # Or GPT-4o.
    llm = None
    if keys.get("google"):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=keys["google"], temperature=0.1)
    elif keys.get("openai"):
        llm = ChatOpenAI(model="gpt-4o", api_key=keys["openai"], temperature=0.1)
    
    if not llm:
        return {"logs": state["logs"] + ["❌ No suitable LLM found for Magistrate."]}

    # 2. System Prompt Injection
    if not state["messages"]:
        sys_msg = SystemMessage(content=PROMPT_V3_MAGISTRATE_CORE + "\n" + PROMPT_V3_HYBRID_FALLBACK)
        # Simplify raw text for prompt context if too large? 
        # No, Gemini 1.5 Pro handles 1M tokens. Pass directly.
        user_msg = HumanMessage(content=f"AUTOS DO PROCESSO:\n{state['raw_text']}")
        messages = [sys_msg, user_msg]
    else:
        messages = state["messages"]

    # 3. Invoke
    response = llm.invoke(messages)
    
    return {
        "messages": messages + [response],
        "iterations": state["iterations"] + 1,
        "logs": state["logs"] + ["🧠 Juiz deliberou."]
    }

def node_computer(state: MagistrateState):
    """
    The Tool Executor. Runs Python code.
    """
    last_msg = state["messages"][-1]
    content = last_msg.content
    
    tool_output = "NO_CODE_FOUND"
    
    # Parse Code Block
    # Look for ```python ... ```
    if "```python" in content:
        import re
        code_match = re.search(r"```python(.*?)```", content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            
            # Execute
            repl = state.get("repl_tool")
            if not repl:
                # Init if missing (should be in state)
                repl = LegalREPL(state["raw_text"])
                
            result = repl.run_code(code)
            tool_output = f"OBSERVATION (PYTHON):\n{result}"
        else:
            tool_output = "ERROR: Failed to parse Python block."
    
    return {
        "messages": state["messages"] + [HumanMessage(content=tool_output)],
        "logs": state["logs"] + ["💻 Código executado."]
    }

def should_continue(state: MagistrateState):
    """
    Edge Condition: 
    - If LAST message contains '```json' -> FINISH
    - If LAST message contains '```python' -> COMPUTER
    - If Iterations > 5 -> FORCE FINISH
    """
    last_msg = state["messages"][-1]
    content = last_msg.content
    
    if "```json" in content and "minuta_final" in content:
        return "end"
    
    if "```python" in content:
        return "computer"
        
    if state["iterations"] > 5:
        # Force finish or return final
        return "end" # Fail safe
        
    # If standard text, maybe it's asking a question? 
    # For V3 Autonomous, it shouldn't ask user. 
    # It might mean it's "Thinking" out loud. We loop back to Magistrate.
    return "magistrate" # Self-correction or continued thought

# --- GRAPH ---

def build_v3_graph():
    workflow = StateGraph(MagistrateState)
    
    workflow.add_node("magistrate", node_magistrate)
    workflow.add_node("computer", node_computer)
    
    workflow.set_entry_point("magistrate")
    
    workflow.add_conditional_edges(
        "magistrate",
        should_continue,
        {
            "computer": "computer",
            "end": END,
            "magistrate": "magistrate"
        }
    )
    
    workflow.add_edge("computer", "magistrate")
    
    return workflow.compile()

def run_autonomous_magistrate(text: str, keys: dict):
    """
    Entry point for V3.
    """
    app = build_v3_graph()
    
    # Initialize REPL just once
    repl = LegalREPL(text)
    
    initial_state = {
        "raw_text": text,
        "keys": keys,
        "repl_tool": repl,
        "messages": [],
        "iterations": 0,
        "final_json": {},
        "logs": []
    }
    
    final_state = app.invoke(initial_state)
    
    # Extract Final JSON
    last_content = final_state["messages"][-1].content
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
