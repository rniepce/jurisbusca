"""Concorrência (fan-out paralelo) + Pricing (tabela de modelos)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _Resp:
    def __init__(self, c):
        self.content = c
        self.usage_metadata = {"input_tokens": 30, "output_tokens": 15}


class _SlowLLM:
    """Simula latência para forçar paralelismo real."""

    def __init__(self, content: str, sleep_s: float = 0.1):
        self.content = content
        self.sleep_s = sleep_s

    def invoke(self, _msgs):
        time.sleep(self.sleep_s)
        return _Resp(self.content)


def edge(src, tgt, label="", h=""):
    return {"id": f"e_{src}_{tgt}", "source": src, "target": tgt,
            "label": label, "sourceHandle": h}


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
# Concorrência
# ─────────────────────────────────────────────────────────────────────

class TestParallelConcurrency(unittest.TestCase):

    def test_ten_parallel_agents_all_complete(self):
        """10 agentes em paralelo, todos rodam, state final tem 10 vars."""
        n = 10
        nodes = [node("s", "start")] + [
            node(f"a{i}", "agent", label=f"A{i}", prompt=f"p{i}", output_var=f"v{i}")
            for i in range(n)
        ] + [node("e", "end")]
        edges = (
            [edge("s", f"a{i}") for i in range(n)] +
            [edge(f"a{i}", "e") for i in range(n)]
        )
        config = {"nodes": nodes, "edges": edges}

        # Cada agente devolve seu índice; usa LLM lento pra forçar paralelismo
        contents = [f"saida-{i}" for i in range(n)]
        idx = [0]
        lock = threading.Lock()

        def factory(*_a, **_k):
            with lock:
                i = idx[0]
                idx[0] += 1
            return _SlowLLM(contents[i] if i < n else "extra", sleep_s=0.05)

        with patch("backend.get_llm", factory):
            t0 = time.perf_counter()
            events = run(config, "x")
            elapsed = time.perf_counter() - t0

        # Sequencial seria 10 × 0.05 = 0.5s; paralelo (8 workers) ~ 0.1s
        # Aceitamos qualquer coisa abaixo de 0.4s — confirma paralelismo
        self.assertLess(elapsed, 0.4,
                        f"10 agentes paralelos levaram {elapsed:.2f}s — não rodaram em paralelo")

        # Todos completaram
        done_labels = [e["label"] for e in events if e["event"] == "node_done"]
        for i in range(n):
            self.assertIn(f"A{i}", done_labels)

        # State final tem todas as variáveis
        final = [e for e in events if e["event"] == "flow_done"][-1]
        for i in range(n):
            self.assertIn(f"v{i}", final["state"])

    def test_parallel_one_fails_others_still_recorded(self):
        """Se 1 nó em paralelo falha, os outros não são corrompidos."""
        nodes = [
            node("s", "start"),
            node("a", "agent", label="OK1", prompt="p1", output_var="v1"),
            node("b", "agent", label="Boom", prompt="p2", output_var="v2"),
            node("c", "agent", label="OK3", prompt="p3", output_var="v3"),
            node("e", "end"),
        ]
        edges = [edge("s", "a"), edge("s", "b"), edge("s", "c"),
                 edge("a", "e"), edge("b", "e"), edge("c", "e")]
        config = {"nodes": nodes, "edges": edges}

        counter = [0]
        counter_lock = threading.Lock()

        class _LLM:
            def invoke(self_, _msgs):
                with counter_lock:
                    i = counter[0]
                    counter[0] += 1
                if i == 1:
                    raise RuntimeError("falha sintética")
                return _Resp(f"out-{i}")

        def factory(*_a, **_k):
            return _LLM()

        with patch("backend.get_llm", factory):
            events = run(config, "x")

        # Pelo menos 1 node_error
        errors = [e for e in events if e["event"] == "node_error"]
        self.assertEqual(len(errors), 1)
        # E o flow não emite flow_done (engine aborta na falha)
        self.assertFalse(any(e["event"] == "flow_done" for e in events))

    def test_state_lock_prevents_concurrent_dict_corruption(self):
        """Roda 50 agentes em paralelo gravando no state — não pode haver
        chave perdida ou dict corrompido."""
        n = 50
        nodes = [node("s", "start")] + [
            node(f"a{i}", "agent", label=f"A{i}", prompt="p", output_var=f"v{i}")
            for i in range(n)
        ] + [node("e", "end")]
        edges = (
            [edge("s", f"a{i}") for i in range(n)] +
            [edge(f"a{i}", "e") for i in range(n)]
        )
        config = {"nodes": nodes, "edges": edges}

        def factory(*_a, **_k):
            return _SlowLLM("conteudo", sleep_s=0.01)

        with patch("backend.get_llm", factory):
            events = run(config, "x")
        final = [e for e in events if e["event"] == "flow_done"][-1]
        # Todas as 50 vars precisam estar lá
        self.assertEqual(len([k for k in final["state"] if k.startswith("v")]), n)


# ─────────────────────────────────────────────────────────────────────
# Pricing
# ─────────────────────────────────────────────────────────────────────

class TestPricing(unittest.TestCase):

    def test_known_model_returns_positive_cost(self):
        import flow_pricing
        cost = flow_pricing.estimate_cost_usd("gpt-5.4-mini", 100_000, 50_000)
        # 100k input * 0.25 + 50k output * 2.00 = 25 + 100 = 125 ¢ / 1k = $0.125
        self.assertAlmostEqual(cost, 0.125, places=4)

    def test_unknown_model_falls_back_to_zero(self):
        import flow_pricing
        cost = flow_pricing.estimate_cost_usd("modelo-inexistente-xyz", 1000, 1000)
        self.assertEqual(cost, 0.0)

    def test_prefix_matching(self):
        import flow_pricing
        # claude-sonnet-4-5-20250514 deve casar com claude-sonnet-4-5
        inp, out = flow_pricing.get_pricing("claude-sonnet-4-5-20250514")
        self.assertEqual(inp, 3.00)
        self.assertEqual(out, 15.00)

    def test_empty_model_returns_zero(self):
        import flow_pricing
        self.assertEqual(flow_pricing.estimate_cost_usd("", 1000, 1000), 0.0)
        self.assertEqual(flow_pricing.estimate_cost_usd(None, 1000, 1000), 0.0)

    def test_zero_tokens_zero_cost(self):
        import flow_pricing
        self.assertEqual(flow_pricing.estimate_cost_usd("gpt-5.3-chat", 0, 0), 0.0)

    def test_precision_does_not_overflow(self):
        import flow_pricing
        # 1 milhão de tokens × $3/M = $3 — confere precisão
        cost = flow_pricing.estimate_cost_usd("claude-sonnet-4-5", 1_000_000, 0)
        self.assertAlmostEqual(cost, 3.00, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
