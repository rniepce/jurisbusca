"""
Knowledge Base Loader — carrega Arquivos A, B e C dos módulos JS compartilhados.
Usado pelos engines V2 (triage_agent) e V3 (orchestrator).
"""
import os
import re

_PROMPTS_DIR = os.path.join(
    os.path.dirname(__file__), "frontend", "src", "prompts"
)

def _extract_template_literal(filepath: str) -> str:
    """Extrai conteúdo entre backticks de um módulo JS (export const X = `...`;)."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    match = re.search(r"`([\s\S]*?)`", text)
    return match.group(1).strip() if match else ""


def load_knowledge_base() -> str:
    """Retorna a base de conhecimento completa (Arquivos A + B + C) como string."""
    parts = []
    for filename in [
        "arquivoASobrestamentos.js",
        "arquivoBSumulas.js",
        "arquivoCQualificados.js",
    ]:
        filepath = os.path.join(_PROMPTS_DIR, filename)
        if os.path.exists(filepath):
            content = _extract_template_literal(filepath)
            if content:
                parts.append(content)
    return "\n\n".join(parts)


# Cache singleton — carregado uma vez na inicialização
KNOWLEDGE_BASE = load_knowledge_base()
