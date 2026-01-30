
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

class PlanningEngine:
    """
    AGENTE DE PLANEJAMENTO (O Juiz Sênior).
    Gera um esqueleto lógico (Outline) antes da redação final.
    """
    
    def __init__(self, api_key: str, provider: str = "google"):
        self.api_key = api_key
        self.provider = provider
        
        if provider == "openai":
            # Modelo de Raciocínio (Reasoning) ou GPT-4o para planejar
            self.llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.1)
        else:
             # Gemini 3 Pro para planejamento (maior inteligência)
            self.llm = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=api_key, temperature=0.1)

    def generate_outline(self, case_summary: str) -> str:
        """
        Gera o plano de minutança (Esqueleto Lógico).
        """
        
        system_prompt = """
Você é um Juiz de Direito Sênior com 30 anos de carreira (Agente de Planejamento).
Sua função NÃO é escrever a sentença, mas sim DELINEAR A ESTRUTURA LÓGICA (Outline) que seu assessor deverá seguir.

# OBJETIVO
Ler o resumo do caso e criar um Esqueleto de Tópicos (Step-by-Step) para o julgamento.

# REGRAS RÍGIDAS
1. NÃO escreva parágrafos de texto corrido. Use apenas Tópicos (Bullet Points).
2. Ordene logicamente: Preliminares -> Prejudiciais -> Mérito (Fato A, Fato B) -> Danos -> Conclusão.
3. Para cada tópico, indique qual prova ou fato específico deve ser citado (ex: "Citar laudo fl. 40").
4. Identifique as Teses Jurídicas aplicáveis (Súmulas, Temas Repetitivos).

# SAÍDA ESPERADA (Exemplo)
## 1. RELATÓRIO
- Resumir pedido inicial (Dano Moral + Material).
- Citar contestação do Réu (Tese de Mero Aborrecimento).

## 2. FUNDAMENTAÇÃO
### 2.1 Preliminares
- Ilegitimidade Passiva: Rejeitar (Citar Teoria da Aparência).

### 2.2 Mérito
- Falha na Prestação de Serviço: Confirmada por Laudo Pericial (fl. X).
- Dano Moral: Ocorrência (Citar Súmula 387 STJ).
- Quantum: Fixar em R$ 5.000,00 (Critério Bifásico).

## 3. DISPOSITIVO
- Julgar Parcialmente Procedente.
"""
        
        human_msg = f"RESUMO DO CASO:\n{case_summary}\n\nGERAR ESQUELETO LÓGICO:"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_msg)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            return chain.invoke({})
        except Exception as e:
            return f"Erro ao gerar Outline: {e}"
