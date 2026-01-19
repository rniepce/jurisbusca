from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import json

PROMPT_CONTEXT_AGENT = """
Você é o AGENTE DE CONTEXTO E FATOS (Estagiário Sênior V3 - Powered by Python).
O texto completo do processo NÃO está no seu prompt, mas está carregado na variável Python `PROCESS_TEXT`.

SUA MISSÃO:
1. Use a ferramenta `navigator` para rodar scripts Python e explorar o `PROCESS_TEXT`.
2. USE A FUNÇÃO `smart_search(regex)` para encontrar termos chave. Ela já retorna o contexto (+/- 500 caracteres) automaticamente.
3. Extraia os fatos com precisão cirúrgica baseada no retorno do código.

SAÍDA ESPERADA (JSON FINAL):
{
    "fatos_principais": "Resumo cronológico...",
    "pedidos_autor": ["..."],
    "teses_defesa": ["..."],
    "datas_chave": {"data_fato": "DD/MM/AAAA", ...},
    "provas_citadas": ["Contrato (fls. encontrados via código)", ...]
}

IMPORTANTE:
- NÃO TENTE "ADIVINHAR" O TEXTO. USE O CÓDIGO PARA LER.
- Use `smart_search(r'AUDIÊNCIA')` para achar trechos relevantes com contexto seguro.
- **SAFETY FALLBACK:** Se `smart_search` não encontrar nada, LEIA O INÍCIO DO TEXTO DIRETAMENTE rodando: `print(PROCESS_TEXT[:5000])`. Não retorne JSON vazio sem antes ler o texto bruto!
"""

def smart_search_impl(text: str, pattern: str, window: int = 500) -> str:
    import re
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return f"Nenhum match encontrado para: {pattern}"
    
    results = []
    results.append(f"Encontrados {len(matches)} resultados para '{pattern}':")
    for i, m in enumerate(matches[:5]): # Top 5 matches
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        excerpt = text[start:end].replace('\\n', ' ')
        results.append(f"--- MATCH {i+1} (Pos {m.start()}) ---\n...{excerpt}...\n")
    return "\\n".join(results)

def run_context_agent(text_content: str, api_key: str):
    """
    Agente V3 que usa Python REPL para navegar no texto bruto.
    """
    try:
        # 1. Configura Ambiente REPL com o texto carregado e Helper Functions
        repl = PythonREPL()
        repl.globals["PROCESS_TEXT"] = text_content
        
        # Injeta a função smart_search no escopo do REPL
        # Como o REPL roda string, precisamos passar o código da função ou injetar a lambda/wrapper
        # Mas PythonREPL.run executa em seu proprio escopo. A melhor forma é definir a funçao no setup code.
        setup_code = """
import re
def smart_search(pattern, window=500):
    text = PROCESS_TEXT
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        print(f"Nenhum match encontrado para: {pattern}")
        return
    
    print(f"Encontrados {len(matches)} resultados para '{pattern}':")
    for i, m in enumerate(matches[:5]): 
        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        excerpt = text[start:end].replace('\\n', ' ')
        print(f"--- MATCH {i+1} (Pos {m.start()}) ---")
        print(f"...{excerpt}...")
        print("-" * 20)
"""
        repl.run(setup_code)

        @tool
        def navigator(code: str):
            """
            Executa código Python para ler/buscar no `PROCESS_TEXT`. 
            Use para encontrar padrões (Regex), datas ou ler trechos.
            A variável `PROCESS_TEXT` contém todo o processo.
            """
            try:
                # Captura prints
                return repl.run(code)
            except Exception as e:
                return f"Erro de Execução: {e}"

        tools = [navigator]
        
        # 2. Inicializa LLM com Tools
        llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=api_key, temperature=0.0)
        llm_with_tools = llm.bind_tools(tools)
        
        # 3. Loop de Execução (ReAct Manual Simplificado)
        messages = [
            SystemMessage(content=PROMPT_CONTEXT_AGENT),
            HumanMessage(content="Comece a exploração. Leia o início do processo e busque os fatos principais via código.")
        ]
        
        # Turno 1: LLM decide rodar código
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Loop de Agente (até 3 tentativas de código para não ficar lento)
        for _ in range(3):
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "navigator":
                        # Executa ferramenta
                        tool_output = navigator.invoke(tool_call["args"])
                        messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                
                # Devolve output para LLM
                response = llm_with_tools.invoke(messages)
                messages.append(response)
            else:
                break
        
        # 4. Parsing Final
        text_out = response.content
        if "```json" in text_out:
            cleaned = text_out.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        else:
            # Fallback se não retornou JSON limpo
            return {"fatos_principais": text_out, "iso_mode": "raw_output"}

    except Exception as e:
        return {"error": str(e)}
