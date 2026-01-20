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
2. **PASSO 0 (OBRIGATÓRIO):** Execute `print(PROCESS_TEXT[:3000])` para ler o cabeçalho e a causa de pedir antes de qualquer busca.
3. USE A FUNÇÃO `smart_search(regex)` para encontrar termos chave adicionais (audiência, sentença, contestação).
4. Extraia os fatos com precisão cirúrgica baseada no retorno do código.

SAÍDA ESPERADA (JSON FINAL):
{
    "fatos_principais": "Resumo cronológico...",
    "pedidos_autor": ["..."],
    "teses_defesa": ["..."],
    "datas_chave": {"data_fato": "DD/MM/AAAA", ...},
    "provas_citadas": ["Contrato (fls. encontrados via código)", ...]
}

IMPORTANTE:
- **HETEROGENEIDADE:** Advogados escrevem de formas diferentes. Se não achar "PETIÇÃO INICIAL", busque "EXORDIAL", "PEÇA PÓRTICA", "PROEMIAL".
- **CHEAT SHEET (REGEX SUGERIDOS):**
  - Inicial: `r'(petiç[aã]o\s*inicial|exordial|fatos|dos\s*fatos|resumo\s*da\s*demanda)'`
  - Defesa: `r'(contesta[cç][aã]o|defesa|mérito|do\s*direito|preliminar)'`
  - Audiência: `r'(audi[êe]ncia|concilia[cç][aã]o|termo|assentada)'`
  - Sentença/Decisão: `r'(senten[çc]a|decis[aã]o|dispositivo|julgo|ante\s*o\s*exposto)'`
- **SAFETY FALLBACK:** Se `smart_search` não encontrar nada, LEIA O INÍCIO DO TEXTO DIRETAMENTE rodando: `print(PROCESS_TEXT[:5000])`. 
- **PROTOCOLO DE FALHA DE BUSCA:** Se uma busca específica (ex: 'data da audiência') retornar 0 resultados:
  1. NÃO DESISTA nem retorne "não encontrado".
  2. ESTIME onde a informação estaria (ex: audiências costumam estar no final ou em despachos curtos).
  3. LEIA O TEXTO NESSA REGIÃO com `print(PROCESS_TEXT[start:end])` e procure manualmente.
- **NUNCA RETORNE VAZIO.** Se falhar em tudo, resuma o que você ler no fallback inicial.
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
            data = json.loads(cleaned)
        else:
            # Fallback se não retornou JSON limpo mas parece JSON
            if text_out.strip().startswith("{"):
                 data = json.loads(text_out.strip())
            else:
                 return {"fatos_principais": text_out, "iso_mode": "raw_output"}

        # VALIDATION & FALLBACK RETRY (Crucial for V3 stability)
        # If facts are empty, try a simple direct LLM call
        if not data.get("fatos_principais") or len(str(data.get("fatos_principais"))) < 50:
             print("⚠️ Agente V3 falhou em extrair fatos via Python. Tentando Fallback Direct (V1 style)...")
             
             fallback_prompt = f"""
             Atenção. O método anterior falhou.
             Seu objetivo é ler o texto abaixo e extrair os FATOS PRINCIPAIS e PEDIDOS.
             
             TEXTO:
             {text_content[:30000]}
             
             Retorne APENAS um JSON:
             {{
                "fatos_principais": "Resumo...",
                "pedidos_autor": [],
                "teses_defesa": [],
                "audit_warning": "Extraído via Fallback Direct"
             }}
             """
             fallback_response = llm.invoke(fallback_prompt)
             fallback_content = fallback_response.content.replace("```json", "").replace("```", "").strip()
             try:
                 return json.loads(fallback_content)
             except:
                 return {"fatos_principais": fallback_response.content, "audit_warning": "Fallback Failed JSON Parsing"}
        
        return data

    except Exception as e:
        print(f"❌ Erro Crítico Context Agent V3: {e}")
        return {"fatos_principais": f"Erro de extração: {str(e)}", "error": str(e)}
