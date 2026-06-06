"""flow_pricing.py — Tabela de preços (USD/1M tokens) por modelo.

Usada pelo flow_engine para calcular o custo aproximado de cada nó.
Valores em dólar por milhão de tokens (input/output).
"""

# {model_id: (input_per_1m, output_per_1m)}
# ⚠️ Valores marcados como ESTIMATIVA são aproximações de tier — ajuste com as
# tarifas reais do seu contrato Azure quando confirmar.
_PRICING_USD: dict[str, tuple[float, float]] = {
    # Azure OpenAI (preços públicos OpenAI; Azure costuma ser equivalente)
    "gpt-5.5":                (2.50, 20.00),  # ESTIMATIVA
    "gpt-5.4-pro":            (2.50, 20.00),  # ESTIMATIVA
    "gpt-5.4-mini":           (0.25,  2.00),
    "gpt-5.3-chat":           (1.25, 10.00),
    "gpt-5.2":                (1.25, 10.00),  # ESTIMATIVA
    "gpt-5.2-chat":           (1.25, 10.00),  # ESTIMATIVA
    "gpt-4.1-mini":           (0.40,  1.60),
    "gpt-4o":                 (2.50, 10.00),
    "gpt-4o-mini":            (0.15,  0.60),
    # Anthropic Claude
    "claude-sonnet-4-6":      (3.00, 15.00),
    "claude-sonnet-4-5":      (3.00, 15.00),
    "claude-haiku-4-5":       (1.00,  5.00),
    # Google Gemini
    "gemini-3.1-pro":         (1.25,  5.00),
    "gemini-2.5-pro":         (1.25,  5.00),
    "gemini-2.5-flash":       (0.075, 0.30),
    # DeepSeek (Azure AI Foundry) — tarifas padrão públicas (jun/2026); Azure Foundry
    # pode aplicar markup de ~20-35% sobre estas.
    "DeepSeek-V4-Pro":        (1.74,  3.48),
    "DeepSeek-V4-Flash":      (0.14,  0.28),
    "DeepSeek-V3.2-Speciale": (0.28,  0.42),
    "deepseek-chat":          (0.27,  1.10),
    "deepseek-reasoner":      (0.55,  2.19),
    # Grok (Azure AI Foundry) — Grok 4.3 (jun/2026)
    "grok-4.3":               (1.25,  2.50),
    # Kimi (Azure AI Foundry) — Moonshot (jun/2026)
    "Kimi-K2.5":              (0.60,  3.00),
    "Kimi-K2.6":              (0.95,  4.00),
    "kimi-k2.5":              (0.60,  3.00),
}


def get_pricing(model_id: str) -> tuple[float, float]:
    """Devolve (input_per_1m, output_per_1m) em USD. Fallback (0,0) se desconhecido.

    Busca insensível a maiúsculas/minúsculas (ex.: 'DeepSeek-V4-Pro' vs 'deepseek-...').
    """
    if not model_id:
        return (0.0, 0.0)
    if model_id in _PRICING_USD:
        return _PRICING_USD[model_id]
    mid = model_id.lower()
    # Busca exata e por prefixo, case-insensitive (ex.: claude-3-5-sonnet-...)
    for key, val in _PRICING_USD.items():
        kl = key.lower()
        if mid == kl or mid.startswith(kl) or kl.startswith(mid):
            return val
    return (0.0, 0.0)


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Custo aproximado em USD."""
    inp, out = get_pricing(model_id)
    return round((inp * input_tokens + out * output_tokens) / 1_000_000, 6)
