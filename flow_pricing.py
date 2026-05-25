"""flow_pricing.py — Tabela de preços (USD/1M tokens) por modelo.

Usada pelo flow_engine para calcular o custo aproximado de cada nó.
Valores em dólar por milhão de tokens (input/output).
"""

# {model_id: (input_per_1m, output_per_1m)}
_PRICING_USD: dict[str, tuple[float, float]] = {
    # Azure OpenAI (preços públicos OpenAI; Azure costuma ser equivalente)
    "gpt-5.3-chat":           (1.25, 10.00),
    "gpt-5.4-mini":           (0.25,  2.00),
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
    # DeepSeek
    "deepseek-chat":          (0.27,  1.10),
    "deepseek-reasoner":      (0.55,  2.19),
    # Kimi
    "kimi-k2.5":              (0.60,  2.50),
}


def get_pricing(model_id: str) -> tuple[float, float]:
    """Devolve (input_per_1m, output_per_1m) em USD. Fallback (0,0) se desconhecido."""
    if not model_id:
        return (0.0, 0.0)
    if model_id in _PRICING_USD:
        return _PRICING_USD[model_id]
    # Busca por prefixo (ex.: claude-3-5-sonnet-...)
    for key, val in _PRICING_USD.items():
        if model_id.startswith(key) or key.startswith(model_id):
            return val
    return (0.0, 0.0)


def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Custo aproximado em USD."""
    inp, out = get_pricing(model_id)
    return round((inp * input_tokens + out * output_tokens) / 1_000_000, 6)
