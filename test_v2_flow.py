import os
from dotenv import load_dotenv

# Carrega as chaves do .env
load_dotenv()

from v2_engine.orchestrator_v2 import run_hybrid_orchestration

def test_v2():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not anthropic_key:
        print("Erro: ANTHROPIC_API_KEY não encontrada no .env")
        return

    keys = {"anthropic": anthropic_key}
    
    # Texto mock de um processo
    mock_text = """
    EXCELENTÍSSIMO SENHOR JUIZ DE DIREITO DA 1ª VARA CÍVEL DA COMARCA DE BELO HORIZONTE/MG
    
    Ação de Indenização por Danos Morais e Materiais
    Autor: João da Silva
    Réu: Companhia Aérea Voe Bem S.A.
    
    DOS FATOS
    O autor adquiriu passagem aérea para o trecho BH - SP (ID 12345). No dia do embarque, o voo foi cancelado sem aviso prévio, fazendo o autor perder um compromisso importante de trabalho (ID 6789).
    
    DO DIREITO
    Aplica-se o Código de Defesa do Consumidor, com a inversão do ônus da prova. O dano moral é in re ipsa em casos de atraso/cancelamento de voo superior a 4 horas.
    
    DOS PEDIDOS
    Ante o exposto, requer a condenação da ré ao pagamento de R$ 10.000,00 a título de danos morais, bem como R$ 800,00 de danos materiais referentes à remarcação. Valor da causa: R$ 10.800,00.
    """
    
    mock_knowledge = """
    = Tema 999 =
    Não há ordem de suspensão para este tema no momento.
    """
    
    print("Iniciando a orquestração V2...")
    resultado = run_hybrid_orchestration(mock_text, keys, mock_knowledge)
    
    print("\n--- ETAPA 1: RELATÓRIO DE TRIAGEM ---")
    print(resultado.get("final_report", "Nenhum relatório gerado."))
    
    print("\n--- ETAPA 2: MINUTA (DRAFT) ---")
    print(resultado.get("draft_text", "Nenhuma minuta gerada."))
    
    print("\n--- ETAPA 3: DASHBOARD DE REVISÃO (QA) ---")
    print(resultado.get("auditor_dashboard", "Nenhum dashboard gerado."))
    
    print("\n--- LOGS DA EXECUÇÃO ---")
    for log in resultado.get("logs", []):
        print(log)

if __name__ == "__main__":
    test_v2()
