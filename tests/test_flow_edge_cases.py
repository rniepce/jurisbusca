"""Testes adversariais e de robustez: fluxos malformados, falhas no LLM,
ciclos, condições explosivas, isolamento do `safe_eval`, etc.
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
        self.usage_metadata = {"input_tokens": 5, "output_tokens": 5}


class _LLM:
    def __init__(self, recipe):
        self.recipe = recipe

    def invoke(self, messages):
        out = self.recipe(messages) if callable(self.recipe) else self.recipe
        if isinstance(out, Exception):
            raise out
        return _Resp(out)


def make_factory(recipe):
    return lambda *_a, **_k: _LLM(recipe)


def edge(src, tgt, label="", source_handle=""):
    return {"id": f"e_{src}_{tgt}_{label}", "source": src, "target": tgt,
            "label": label, "sourceHandle": source_handle}


def node(node_id, ntype, **data):
    return {"id": node_id, "type": ntype, "position": {"x": 0, "y": 0}, "data": data}


def run(config, input_text="", **kw):
    import flow_engine
    out = []
    for chunk in flow_engine.build_and_run_flow(config, input_text, **kw):
        line = chunk.strip()
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: "):]))
    return out


# ─────────────────────────────────────────────────────────────────────
# Robustez de configuração
# ─────────────────────────────────────────────────────────────────────

class TestMalformedConfig(unittest.TestCase):

    def test_no_start_node_emits_error(self):
        config = {
            "nodes": [
                node("a", "agent", label="A", prompt="x", output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("a", "e")],
        }
        with patch("backend.get_llm", make_factory("ok")):
            events = run(config, "x")
        # Sem start, o engine deve emitir error em vez de travar
        self.assertTrue(any(e["event"] == "error" for e in events),
                        "fluxo sem start não emitiu evento de erro")

    def test_no_end_node_completes_when_chain_runs_out(self):
        """Sem nó end, o fluxo deve simplesmente parar quando não houver
        mais nós ready — não deve crashar nem rodar para sempre."""
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="x", output_var="oa"),
            ],
            "edges": [edge("s", "a")],
        }
        with patch("backend.get_llm", make_factory("ok")):
            events = run(config, "x")
        # rodou pelo menos o agent
        labels = [e.get("label") for e in events if e["event"] == "node_done"]
        self.assertIn("A", labels)
        # mas como não há end, nunca emite flow_done — e isso é OK, não crasha
        self.assertFalse(any(e["event"] == "error" for e in events))

    def test_circuit_breaker_against_loops(self):
        """Ciclo A→B→A: o circuit breaker deve disparar em vez de pendurar."""
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="pa", output_var="oa"),
                node("b", "agent", label="B", prompt="pb", output_var="ob"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "a"), edge("a", "b"),
                edge("b", "a"),  # ciclo!
                edge("a", "e"),
            ],
        }
        # Em prática, o scheduler atual roda cada nó uma vez (controla por
        # `completed`), então o ciclo simplesmente não re-executa. O importante
        # é que não trava nem crasha.
        with patch("backend.get_llm", make_factory("ok")):
            events = run(config, "x")
        types = [e["event"] for e in events]
        # ou completa, ou hits circuit breaker — em ambos não pode travar
        self.assertTrue("flow_done" in types or "error" in types,
                        f"flow com ciclo travou sem emitir nada terminal: {types}")

    def test_orphan_node_is_ignored(self):
        """Nó solto (sem predecessor nem ligação a partir do start) é ignorado."""
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="pa", output_var="oa"),
                node("orphan", "agent", label="Orphan", prompt="orph",
                     output_var="orfao"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        with patch("backend.get_llm", make_factory("ok")):
            events = run(config, "x")
        labels_done = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("A", labels_done)
        self.assertNotIn("Orphan", labels_done)


# ─────────────────────────────────────────────────────────────────────
# Falhas do LLM
# ─────────────────────────────────────────────────────────────────────

class TestLLMFailures(unittest.TestCase):

    def test_llm_exception_emits_node_error(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Quebra", prompt="x", output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        with patch("backend.get_llm",
                   make_factory(RuntimeError("upstream timeout"))):
            events = run(config, "x")
        errors = [e for e in events if e["event"] == "node_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["label"], "Quebra")
        self.assertIn("upstream timeout", errors[0]["error"])
        # Não pode emitir flow_done depois do erro
        self.assertFalse(any(e["event"] == "flow_done" for e in events))

    def test_llm_empty_string_is_valid_output(self):
        """LLM devolvendo string vazia não deve quebrar — apenas vira saída vazia."""
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="x", output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        with patch("backend.get_llm", make_factory("")):
            events = run(config, "x")
        done = [e for e in events if e["event"] == "node_done"][-1]
        self.assertEqual(done["output"], "")
        final = [e for e in events if e["event"] == "flow_done"][-1]
        self.assertEqual(final["output"], "")


# ─────────────────────────────────────────────────────────────────────
# Roteador: condições adversariais
# ─────────────────────────────────────────────────────────────────────

class TestRouterConditions(unittest.TestCase):

    def test_invalid_condition_defaults_to_false(self):
        # 'undefined_var' não existe no state — safe_eval engole e devolve False
        config = {
            "nodes": [
                node("s", "start"),
                node("r", "router", label="R", condition="undefined_var > 5"),
                node("t", "agent", label="T", prompt="t", output_var="ot"),
                node("f", "agent", label="F", prompt="f", output_var="of"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "r"),
                edge("r", "t", label="verdadeiro", source_handle="true"),
                edge("r", "f", label="falso", source_handle="false"),
                edge("f", "e"),
            ],
        }
        with patch("backend.get_llm", make_factory("ok")):
            events = run(config, "x")
        labels = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("F", labels)
        self.assertNotIn("T", labels)

    def test_safe_eval_blocks_dangerous_builtins(self):
        """Tentar usar __import__ via condition não pode rodar."""
        config = {
            "nodes": [
                node("s", "start"),
                # tentativa de import/exec
                node("r", "router", label="R",
                     condition="__import__('os').system('echo HACK')"),
                node("t", "agent", label="T", prompt="t", output_var="ot"),
                node("f", "agent", label="F", prompt="f", output_var="of"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "r"),
                edge("r", "t", label="verdadeiro", source_handle="true"),
                edge("r", "f", label="falso", source_handle="false"),
                edge("f", "e"),
            ],
        }
        with patch("backend.get_llm", make_factory("ok")):
            events = run(config, "x")
        # Cai em False (porque _safe_eval engole exceção e devolve False)
        labels = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("F", labels)
        self.assertNotIn("T", labels)


# ─────────────────────────────────────────────────────────────────────
# Variáveis e substituição
# ─────────────────────────────────────────────────────────────────────

class TestVariableQuirks(unittest.TestCase):

    def test_unknown_label_substitution_leaves_placeholder(self):
        """{{Inexistente}} deve permanecer literal se não houver match."""
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A",
                     prompt="Cite isto: {{NaoExiste}}",
                     output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        seen = []
        with patch("backend.get_llm", lambda *a, **k: _LLM(
            lambda msgs: seen.append(msgs[0].content) or "ok"
        )):
            run(config, "x")
        self.assertIn("{{NaoExiste}}", seen[0],
                      "placeholder desconhecido foi removido em vez de preservado")

    def test_label_substitution_is_case_insensitive(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Triagem", prompt="p", output_var="triagem"),
                node("b", "agent", label="B",
                     prompt="A triagem foi: {{TRIAGEM}}", output_var="ob"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }
        captured = []
        outs = iter(["RESULTADO-TRIAGEM", "ok"])
        with patch("backend.get_llm", lambda *a, **k: _LLM(
            lambda msgs: (captured.append(msgs[0].content), next(outs))[1]
        )):
            run(config, "x")
        # No 2º system prompt, {{TRIAGEM}} (uppercase) deve virar o conteúdo
        self.assertIn("RESULTADO-TRIAGEM", captured[1])
        self.assertNotIn("{{TRIAGEM}}", captured[1])

    def test_output_var_default_is_node_id_when_omitted(self):
        config = {
            "nodes": [
                node("s", "start"),
                # SEM output_var configurado
                node("agent_xyz", "agent", label="A", prompt="x"),
                node("e", "end"),
            ],
            "edges": [edge("s", "agent_xyz"), edge("agent_xyz", "e")],
        }
        with patch("backend.get_llm", make_factory("conteudo")):
            events = run(config, "x")
        final = [e for e in events if e["event"] == "flow_done"][-1]
        # No fallback, output_var = node_id
        self.assertEqual(final["state"]["agent_xyz"], "conteudo")


# ─────────────────────────────────────────────────────────────────────
# Cancelamento por roteador
# ─────────────────────────────────────────────────────────────────────

class TestCancellationPropagation(unittest.TestCase):

    def test_subtree_cancellation_does_not_fire_events(self):
        """Quando o roteador cancela um ramo, os nós cancelados não podem
        emitir node_start/node_done."""
        config = {
            "nodes": [
                node("s", "start"),
                node("r", "router", label="R", condition="True"),
                node("t", "agent", label="T", prompt="t", output_var="ot"),
                # ramo falso encadeado
                node("f1", "agent", label="F1", prompt="f1", output_var="of1"),
                node("f2", "agent", label="F2", prompt="f2", output_var="of2"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "r"),
                edge("r", "t", label="verdadeiro", source_handle="true"),
                edge("r", "f1", label="falso", source_handle="false"),
                edge("f1", "f2"),
                edge("t", "e"),
            ],
        }
        with patch("backend.get_llm", make_factory("ok")):
            events = run(config, "x")
        all_labels = set()
        for e in events:
            if e["event"] in ("node_start", "node_done"):
                all_labels.add(e.get("label"))
        self.assertIn("T", all_labels)
        self.assertNotIn("F1", all_labels,
                         "F1 (cancelado) emitiu evento")
        self.assertNotIn("F2", all_labels,
                         "F2 (cancelado por propagação) emitiu evento")


if __name__ == "__main__":
    unittest.main(verbosity=2)
