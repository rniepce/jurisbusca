from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

PROMPT_AUDITOR_AGENT = """
# FUNÇÃO: AUDITOR SÊNIOR (COMPLIANCE, INTEGRIDADE E QUALIDADE)
Você é a ÚLTIMA BARREIRA antes que a minuta vá para assinatura do Magistrado.
Sua aprovação determina se a minuta é entregue ou devolvida para correção.

## CRITÉRIOS DE AUDITORIA (5 EIXOS)

### 1. INTEGRIDADE FÁTICA (Anti-Alucinação)
- A minuta cita fatos que realmente existem nos dados brutos?
- Os IDs citados (ex: "ID 123") existem nos autos?
- Os valores monetários e datas conferem com os dados originais?
- Os nomes das partes (Autor/Réu) estão corretos e não invertidos?

### 2. CONGRUÊNCIA PROCESSUAL (Citra/Extra/Ultra Petita)
- **Citra Petita:** A minuta deixou de julgar algum pedido do autor? (Se sim: ERRO GRAVE)
- **Extra Petita:** A minuta julgou algo que NÃO foi pedido? (Se sim: ERRO GRAVE)
- **Ultra Petita:** A minuta concedeu MAIS do que foi pedido? (Se sim: ERRO GRAVE)
- O dispositivo é coerente com a fundamentação?

### 3. FUNDAMENTAÇÃO ANALÍTICA (Art. 489 §1º CPC)
- A minuta explica a relação entre a norma citada e o caso concreto?
- Todos os argumentos relevantes da parte vencida foram enfrentados?
- Há fundamentação genérica que "serviria a qualquer decisão"?
- Ao citar precedente/súmula, demonstrou que se aplica ao caso?

### 4. CONFORMIDADE LEGAL
- A fundamentação jurídica faz sentido logicamente?
- As normas citadas existem e estão corretas?
- Se é relação de consumo, foi aplicada inversão de ônus (CDC Art. 6º, VIII)?

### 5. CONFORMIDADE DE ESTILO
- O texto segue as diretrizes do "Estilo do Juiz" (se fornecido)?
- A estrutura visual (cabeçalho, relatório, fundamentação, dispositivo) está correta?
- IDs/folhas são citados ao mencionar documentos?

---

## ESTILO DO JUIZ (RAG)
{style_guide}

## DADOS BRUTOS (DOS AUTOS — FONTE DA VERDADE)
{fatos_json}

## MINUTA PARA REVISÃO
{draft_text}

---

## FORMATO DE SAÍDA (OBRIGATÓRIO)
Retorne sua análise EXATAMENTE neste formato:

**VEREDITO:** [APROVADO | COM RESSALVAS | REPROVADO]

**CHECKLIST:**
| Critério | Status | Detalhe |
|---|---|---|
| Integridade Fática | ✅/⚠️/❌ | [nota] |
| Congruência (Petita) | ✅/⚠️/❌ | [nota] |
| Art. 489 §1º | ✅/⚠️/❌ | [nota] |
| Conformidade Legal | ✅/⚠️/❌ | [nota] |
| Conformidade Estilo | ✅/⚠️/❌ | [nota] |

**ERROS ENCONTRADOS:**
[Se houver, liste cada erro com: Onde está → O que está errado → Como corrigir]

**PARECER FINAL:**
[Breve resumo. Se APROVADO, confirme. Se REPROVADO, explique o que o Redator deve corrigir.]
"""

from langchain_google_genai import ChatGoogleGenerativeAI

def run_auditor_agent(draft_text: str, fatos_json: dict, style_guide: str, keys: dict):
    """
    Executa Auditoria com Fallback:
    1. Tenta GPT-4o (OpenAI).
    2. Se falhar (ex: Quota Limit), tenta Gemini 3.0 Pro (Google).
    
    Returns:
        str: Relatório de auditoria com veredito (APROVADO/COM RESSALVAS/REPROVADO)
    """
    openai_key = keys.get("openai")
    google_key = keys.get("google")

    formatted_prompt = PROMPT_AUDITOR_AGENT.format(
        style_guide=style_guide or "Estilo Padrão (formal, impessoal, estruturado).",
        fatos_json=str(fatos_json),
        draft_text=draft_text
    )
    messages = [
        SystemMessage(content=formatted_prompt),
        HumanMessage(content="Execute a auditoria completa nos 5 eixos e emita seu veredito.")
    ]

    # 1. TENTATIVA OPENAI (GPT-4o)
    if openai_key:
        try:
            llm = ChatOpenAI(
                api_key=openai_key,
                model="gpt-4o", 
                temperature=0.1
            )
            return llm.invoke(messages).content
        except Exception as e:
            print(f"⚠️ Erro OpenAI (Auditor): {e}. Tentando Fallback para Gemini...")
    
    # 2. TENTATIVA FALLBACK GOOGLE (GEMINI 3.0 PRO)
    if google_key:
        try:
            llm_fallback = ChatGoogleGenerativeAI(
                google_api_key=google_key,
                model="gemini-3-pro-preview",
                temperature=0.1
            )
            response = llm_fallback.invoke(messages).content
            return f"{response}\n\n[NOTA: Auditoria realizada via Gemini 3.0 Pro Preview (Fallback Ativo)]"
        except Exception as e_google:
            return f"Erro Agente Auditor (OpenAI & Google): {str(e_google)}"
            
    return "Erro: Nenhuma chave válida (OpenAI ou Google) para auditoria."
