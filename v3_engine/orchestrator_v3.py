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
# Both variables now correctly exported from prompts_magistrate_v3.py (typo ROMPT_ -> PROMPT_ fixed)
from prompts_magistrate_v3 import PROMPT_V3_MAGISTRATE_CORE, PROMPT_V3_HYBRID_FALLBACK
from knowledge_base_loader import KNOWLEDGE_BASE

# Internal Imports for LLM
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

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
    # Use Azure OpenAI GPT-5.2-chat
    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2-chat")
    
    if not azure_key or not azure_endpoint:
        return {"logs": state["logs"] + ["❌ Azure OpenAI not configured (AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT)."]}

    llm = AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        api_version="2024-12-01-preview",
    )

    @tool
    def run_python_code(code: str) -> str:
        """
        Executes Python code in a secure REPL environment.
        The full process text is available as a string variable named `text` and `PROCESS_TEXT`.
        Helper functions available: search_dates(keyword_context, window), search_money(keyword_context), search_parties(), grep(pattern).
        Always use `print()` or output the result of your code so that you can see it.
        """
        pass
        
    llm_with_tools = llm.bind_tools([run_python_code])

    # 2. System Prompt Injection
    if not state["messages"]:
        # Format the prompt with tribunal_local (default: TJMG)
        tribunal = state.get("tribunal_local", "TJMG")
        core_prompt = PROMPT_V3_MAGISTRATE_CORE.replace("{tribunal_local}", tribunal)
        
        # Adding a specific guidance for tool utilization with Claude
        gpt_tool_instruction = "\n\nCRÍTICO: Use a ferramenta 'run_python_code' para ler o processo usando Python. Após ler os dados, elabore o JSON com a 'minuta_final'."
        
        knowledge_section = "\n\n# BASE DE CONHECIMENTO (ARQUIVOS A, B e C)\n" + KNOWLEDGE_BASE if KNOWLEDGE_BASE else ""
        sys_msg = SystemMessage(content=core_prompt + "\n" + PROMPT_V3_HYBRID_FALLBACK + knowledge_section + gpt_tool_instruction)
        # Claude handles large contexts — pass the full text directly.
        user_msg = HumanMessage(content=f"AUTOS DO PROCESSO:\n{state['raw_text']}")
        messages = [sys_msg, user_msg]
    else:
        messages = state["messages"]

    # 3. Invoke
    response = llm_with_tools.invoke(messages)
    
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
            "logs": state["logs"] + ["💻 Código executado via Native Tool Calling."]
        }
    
    # Fallback to old regex parsing just in case LLM hallucinated
    content = last_msg.content
    tool_output = "NO_CODE_FOUND"
    
    if isinstance(content, str) and "```python" in content:
        import re
        code_match = re.search(r"```python(.*?)```", content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            repl = state.get("repl_tool")
            if not repl:
                repl = LegalREPL(state["raw_text"])
                
            result = repl.run_code(code)
            tool_output = f"OBSERVATION (PYTHON):\n{result}"
        else:
            tool_output = "ERROR: Failed to parse Python block."
            
    return {
        "messages": state["messages"] + [HumanMessage(content=tool_output)],
        "logs": state["logs"] + ["💻 Código executado via Regex Fallback."]
    }

def should_continue(state: MagistrateState):
    """
    Edge Condition: 
    - If LAST message requested tool call -> COMPUTER
    - If LAST message contains '```json' -> FINISH
    - If Iterations > 5 -> FORCE FINISH
    """
    last_msg = state["messages"][-1]
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "computer"
        
    content = last_msg.content
    if isinstance(content, str):
        if "```json" in content and "minuta_final" in content:
            return "end"
        if "```python" in content:
            return "computer"
            
    if state["iterations"] > 5:
        # Force finish or return final
        return "end" # Fail safe
        
    # If standard text, it might mean it's "Thinking" out loud. We loop back to Magistrate.
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
