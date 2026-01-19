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
2. Encontre datas, valores, nomes e folhas (páginas) específicas usando Regex ou fatiamento de string.
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
- Exemplo: `print(PROCESS_TEXT[:500])` para ler o início.
- Exemplo: `import re; print(re.findall(r'VALOR DA CAUSA: R\$ [\d,.]+', PROCESS_TEXT))`
"""

def run_context_agent(text_content: str, api_key: str):
    """
    Agente V3 que usa Python REPL para navegar no texto bruto.
    """
    try:
        # 1. Configura Ambiente REPL com o texto carregado
        repl = PythonREPL()
        repl.globals["PROCESS_TEXT"] = text_content
        
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
        # Modelo Rápido e Barato (Flash) para leitura de grandes volumes
        llm = ChatGoogleGenerativeAI(model="gemini-3.0-flash-preview", google_api_key=api_key, temperature=0.0)
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
