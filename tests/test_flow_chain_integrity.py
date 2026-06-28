"""Testes de integridade da cadeia de agentes.

Prova *byte-a-byte* que o output do agente N entra inteiro no input do
agente N+1, sem perda, truncamento ou mutação. Cobre:

  - Cadeia longa (5 agentes em série) com conteúdos distintos;
  - Conteúdo realista (Markdown jurídico longo, acentuação, citações);
  - Conteúdo binário-suspeito (\\n, \\t, ", ', <, >, \\, {, });
  - Conteúdo extenso (~50 KB);
  - Confirmação cruzada: o que o LLM devolve é EXATAMENTE o que vira
    state[output_var] e EXATAMENTE o que aparece em [output_var] no
    user_msg do nó seguinte.

Rodar com:
    venv312/bin/python -m unittest tests.test_flow_chain_integrity -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _Resp:
    def __init__(self, c):
        self.content = c
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 10}


class ScriptedLLM:
    """LLM que devolve respostas pré-programadas em fila, e grava tudo."""

    def __init__(self, queue: list[str], registry: list[dict]):
        self.queue = queue
        self.registry = registry

    def invoke(self, messages):
        sys_c = ""
        user_c = ""
        for m in messages:
            t = type(m).__name__
            if t == "SystemMessage":
                sys_c = m.content
            elif t == "HumanMessage":
                user_c = m.content
        # responde na ordem da fila
        out = self.queue.pop(0) if self.queue else "(esgotado)"
        self.registry.append({
            "system": sys_c, "user": user_c, "out": out,
        })
        return _Resp(out)


def make_factory(queue: list[str], registry: list[dict]):
    def factory(model_name: str = "gpt-5.3-chat", **_):
        return ScriptedLLM(queue, registry)
    return factory


def edge(src: str, tgt: str, label: str = "", source_handle: str = "") -> dict:
    return {"id": f"e_{src}_{tgt}", "source": src, "target": tgt,
            "label": label, "sourceHandle": source_handle}


def node(node_id: str, ntype: str, **data) -> dict:
    return {"id": node_id, "type": ntype, "position": {"x": 0, "y": 0}, "data": data}


def run(config, input_text="", **kw):
    import flow_engine
    events = []
    for chunk in flow_engine.build_and_run_flow(config, input_text, **kw):
        line = chunk.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


# ─────────────────────────────────────────────────────────────────────
# Helpers de verificação byte-a-byte
# ─────────────────────────────────────────────────────────────────────

def assert_carries_exactly(test, downstream_user_msg: str, upstream_output_var: str,
                           upstream_output: str):
    """Verifica que [output_var]:\\n<conteúdo exato> aparece no user msg."""
    marker = f"[{upstream_output_var}]:\n{upstream_output}"
    test.assertIn(marker, downstream_user_msg,
                  f"\n\nFalha de integridade!\n"
                  f"Esperava encontrar (literal):\n{'─' * 60}\n{marker[:300]}...\n{'─' * 60}\n"
                  f"Encontrei no user msg:\n{'─' * 60}\n{downstream_user_msg[:600]}\n{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

class TestFiveAgentChain(unittest.TestCase):
    """A1 → A2 → A3 → A4 → A5: cada um produz algo único; cada um vê
    EXATAMENTE o que o anterior produziu."""

    def test_chain_is_byte_exact(self):
        outputs = [
            "## Etapa 1 — Triagem\nClassificação: cível.\nUrgência: alta.",
            "## Etapa 2 — Fatos\n- Parte autora: João Silva\n- Valor da causa: R$ 12.450,00",
            "## Etapa 3 — Direito aplicável\nArt. 422 CC; Art. 6º CDC; Súmula 297 STJ.",
            "## Etapa 4 — Dispositivo\nJULGO PROCEDENTE o pedido. Condeno o réu...",
            "## Etapa 5 — Revisão final\nMinuta revisada e pronta para assinatura.",
        ]
        config = {
            "nodes": [
                node("s", "start"),
                node("a1", "agent", label="Triagem",  prompt="p1", output_var="triagem"),
                node("a2", "agent", label="Fatos",    prompt="p2", output_var="fatos"),
                node("a3", "agent", label="Direito",  prompt="p3", output_var="direito"),
                node("a4", "agent", label="Dispositivo", prompt="p4", output_var="dispositivo"),
                node("a5", "agent", label="Revisão",  prompt="p5", output_var="revisao"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a1"), edge("a1", "a2"), edge("a2", "a3"),
                      edge("a3", "a4"), edge("a4", "a5"), edge("a5", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory(list(outputs), calls)):
            evs = run(config, "petição original integral")

        self.assertEqual(len(calls), 5, f"esperava 5 chamadas, vieram {len(calls)}")

        # Cada agente N ≥ 2 deve ter visto literalmente todas as saídas dos N-1 anteriores
        var_names = ["triagem", "fatos", "direito", "dispositivo", "revisao"]
        for i in range(1, 5):
            user_i = calls[i]["user"]
            for j in range(i):
                assert_carries_exactly(self, user_i, var_names[j], outputs[j])

        # E a saída final do fluxo é exatamente a do último agente
        final = [e for e in evs if e["event"] == "flow_done"][-1]
        self.assertEqual(final["output"], outputs[-1])

        # Print didático
        print("\n────── CADEIA DE 5 AGENTES ──────")
        for i, c in enumerate(calls, 1):
            preview = c["out"].split("\n")[0][:80]
            print(f"  [{i}] {var_names[i-1]:14s} → {preview!r}")
        print(f"  → flow_done.output == última saída: ✓")


class TestRichJudicialContent(unittest.TestCase):
    """Conteúdo realista de minuta — Markdown, acentuação, citações,
    quebras de linha e parágrafos longos."""

    def test_long_legal_text_passes_intact(self):
        long_minuta = (
            "# SENTENÇA\n\n"
            "**Processo nº** 0001234-56.2024.8.13.0024\n"
            "**Autor:** Maria Aparecida dos Santos\n"
            "**Réu:** Banco XYZ S.A.\n\n"
            "---\n\n"
            "## I. RELATÓRIO\n\n"
            "Trata-se de \"ação ordinária\" proposta por MARIA APARECIDA DOS SANTOS "
            "em face de BANCO XYZ S.A., na qual a parte autora alega: (i) cobrança "
            "indevida de tarifas; (ii) dano moral; e (iii) repetição de indébito.\n\n"
            "O valor da causa foi atribuído em R$ 25.430,87.\n\n"
            "## II. FUNDAMENTAÇÃO\n\n"
            "Conforme leciona Pontes de Miranda, 'a relação jurídica obrigacional...'\n\n"
            "Aplicam-se ao caso o art. 6º, VIII do CDC e o art. 42 do mesmo diploma.\n\n"
            "* Item 1 — provas documentais às fls. 12-45\n"
            "* Item 2 — depoimento da testemunha José da Silva\n"
            "* Item 3 — perícia contábil às fls. 87-103\n\n"
            "## III. DISPOSITIVO\n\n"
            "Ante o exposto, JULGO PARCIALMENTE PROCEDENTE o pedido para...\n\n"
            "P.R.I.\n\nBelo Horizonte/MG, " + ("x" * 200)
        )
        config = {
            "nodes": [
                node("s", "start"),
                node("redator", "agent", label="Redator", prompt="redija",
                     output_var="minuta"),
                node("revisor", "agent", label="Revisor", prompt="revise",
                     output_var="minuta_revisada"),
                node("e", "end"),
            ],
            "edges": [edge("s", "redator"), edge("redator", "revisor"), edge("revisor", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory(
            [long_minuta, long_minuta + "\n\n[REVISADO]"], calls,
        )):
            run(config, "petição")

        # O revisor precisa ter recebido a MINUTA EXATA do redator
        assert_carries_exactly(self, calls[1]["user"], "minuta", long_minuta)


class TestSpecialCharacters(unittest.TestCase):
    """JSON, aspas, escapes — tudo precisa passar sem ser tratado como controle."""

    def test_special_chars_survive(self):
        tricky = (
            'Resultado JSON:\n```json\n{"chave": "valor com \\\"aspas\\\" e \\n quebras"}\n```\n'
            "Tags: <tag attr='x'>conteúdo & mais</tag>\n"
            "Chaves: {{nao-deve-virar-placeholder}} e {var_avulsa}\n"
            "Markdown: **bold**, *itálico*, `código`, [link](http://x)\n"
            "Unicode: ✓ ✗ ⚖️ 📄 — “aspas curvas” — São Pãulo"
        )
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Producer", prompt="produza",
                     output_var="bruto"),
                node("b", "agent", label="Consumer", prompt="consuma",
                     output_var="processado"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory([tricky, "ok"], calls)):
            run(config, "x")

        assert_carries_exactly(self, calls[1]["user"], "bruto", tricky)


class TestLargePayload(unittest.TestCase):
    """50 KB de conteúdo — testa que não há truncamento silencioso."""

    def test_50kb_passes_intact(self):
        big = ("Parágrafo de exemplo com 100 caracteres exatos para testar payload grande no fluxo. "
               "X" * 8) * 500  # ~50 KB
        self.assertGreater(len(big), 45_000)

        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="pa", output_var="oa"),
                node("b", "agent", label="B", prompt="pb", output_var="ob"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory([big, "ok"], calls)):
            run(config, "x")

        # b deve ter recebido o conteúdo COMPLETO de a
        marker = f"[oa]:\n{big}"
        self.assertIn(marker, calls[1]["user"],
                      f"payload de {len(big)} chars não chegou inteiro "
                      f"(user_msg tem {len(calls[1]['user'])} chars)")


class TestOutputEqualsStateValue(unittest.TestCase):
    """O que o LLM devolve == state[output_var] == o que aparece no
    próximo agente. Triple-check."""

    def test_triple_check(self):
        produced = "FATO ÚNICO E IDENTIFICÁVEL [token-xyz-9871]"
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="pa", output_var="va"),
                node("b", "agent", label="B", prompt="pb", output_var="vb"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory([produced, "FINAL"], calls)):
            evs = run(config, "input")

        # 1) o LLM "respondeu" exatamente produced
        self.assertEqual(calls[0]["out"], produced)

        # 2) o flow_done.state preserva intacto
        final = [e for e in evs if e["event"] == "flow_done"][-1]
        self.assertEqual(final["state"]["va"], produced)

        # 3) o user msg do B contém exatamente produced
        self.assertIn(produced, calls[1]["user"])


class TestNoMutationAtEachHop(unittest.TestCase):
    """Confirma que NENHUM byte muda entre hops — calcula hash."""

    def test_hash_stability(self):
        import hashlib
        outs = [
            "TRIAGEM\n— alta urgência\n— matéria: cível\n— Σ τ ω " + "x" * 1000,
            "FATOS\n— testemunhas: 3\n— provas: " + "y" * 1000,
            "FINAL\n" + "z" * 500,
        ]
        hashes = [hashlib.sha256(o.encode("utf-8")).hexdigest() for o in outs]

        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="pa", output_var="va"),
                node("b", "agent", label="B", prompt="pb", output_var="vb"),
                node("c", "agent", label="C", prompt="pc", output_var="vc"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "c"), edge("c", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory(list(outs), calls)):
            evs = run(config, "input")

        # Para cada N>=1, o trecho equivalente a "[var_i]:\n<conteúdo>"
        # extraído do user msg deve bater no hash original.
        for i in range(1, 3):
            user_i = calls[i]["user"]
            for j in range(i):
                var = ["va", "vb", "vc"][j]
                expected = outs[j]
                # extrai apenas o bloco [var]:\n<...> até a próxima quebra dupla
                marker = f"[{var}]:\n"
                self.assertIn(marker, user_i, f"marcador {marker!r} sumiu")
                idx = user_i.index(marker) + len(marker)
                end = user_i.find("\n\n[", idx)
                if end == -1:
                    end = user_i.find("\n\nEntrada principal:", idx)
                if end == -1:
                    end = len(user_i)
                got = user_i[idx:end]
                got_hash = hashlib.sha256(got.encode("utf-8")).hexdigest()
                self.assertEqual(
                    got_hash, hashes[j],
                    f"\nConteúdo de '{var}' foi mutado entre hops!\n"
                    f"  esperado sha256: {hashes[j]}\n"
                    f"  recebido sha256: {got_hash}\n"
                    f"  primeiros 200 chars: {got[:200]!r}"
                )

        # Print didático
        print("\n────── INTEGRIDADE POR HASH ──────")
        for j, h in enumerate(hashes):
            print(f"  out {j+1} ({len(outs[j])} chars) sha256={h[:16]}... ✓")


if __name__ == "__main__":
    unittest.main(verbosity=2)
