"""
Resolvedor central de modelos por (etapa, perfil de custo).

Todas as engines (flow_engine, V3, pipeline) perguntam a este módulo
"qual modelo uso na etapa X no perfil Y?" em vez de fixar nomes no código.
Trocar o perfil ativo re-roda o mesmo fluxo num tier de custo diferente,
sem editar grafo nem código de orquestração.

Os nomes retornados são strings que backend.get_llm() já entende:
  - gemini*                       -> Google Gemini
  - claude*                       -> Anthropic
  - deepseek* / grok* / kimi*     -> Azure AI Foundry
  - resto (gpt-5.x, etc.)         -> Azure OpenAI

Regra de ouro do perfil econômico: a economia vem dos passos leves
(gpt-5.4-mini). O passo de raciocínio — único que não decompõe e que usa o
tool loop do LegalREPL — fica forte (DeepSeek-V4-Pro) em ambos os perfis.
"""

DEFAULT_PROFILE = "premium"

PROFILES = {
    # Tier máximo de qualidade (comportamento atual da produção).
    "premium": {
        "reader":     "Kimi-K2.5",
        "reasoner":   "DeepSeek-V4-Pro",
        "formatter":  "gpt-5.3-chat",
        "classifier": "gpt-5.4-mini",
        "style":      "claude-sonnet-4-6",
        "default":    "gpt-5.3-chat",
    },
    # Tier econômico — só modelos baratos no Azure (opt-in, sem MacBook).
    "economico": {
        "reader":     "gpt-5.4-mini",
        "reasoner":   "DeepSeek-V4-Pro",   # forte SÓ no raciocínio
        "formatter":  "gpt-5.4-mini",
        "classifier": "gpt-5.4-mini",
        "style":      "gpt-5.4-mini",
        "default":    "gpt-5.4-mini",
    },
}


def resolve(stage: str, profile: str | None = None) -> str:
    """Retorna o nome do modelo para uma etapa dentro de um perfil.

    Cai para a chave 'default' do perfil se a etapa não existir, e para o
    perfil DEFAULT_PROFILE se o perfil informado for desconhecido/None.
    """
    p = PROFILES.get(profile or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])
    return p.get(stage, p["default"])


def list_profiles() -> list[str]:
    """Lista os perfis disponíveis (para popular a UI, por exemplo)."""
    return list(PROFILES.keys())
