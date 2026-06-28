"""Observabilidade: verifica que cada execução gera um registro
persistente correto em flow_runs + flow_run_events."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _Resp:
    def __init__(self, c, in_t=80, out_t=40):
        self.content = c
        self.usage_metadata = {"input_tokens": in_t, "output_tokens": out_t}


class _LLM:
    """Compartilha a mesma queue entre instâncias (uma queue por fluxo)."""
    def __init__(self, queue_ref: list):
        self._q = queue_ref

    def invoke(self, _msgs):
        out = self._q.pop(0) if self._q else "(esgotado)"
        if isinstance(out, Exception):
            raise out
        return _Resp(out)


def make_factory(queue):
    # mesma referência de lista entre todos os _LLM instanciados
    q = list(queue) if isinstance(queue, (list, tuple)) else [queue]
    return lambda *a, **k: _LLM(q)


def node(node_id, ntype, **data):
    return {"id": node_id, "type": ntype, "position": {"x": 0, "y": 0}, "data": data}


def edge(src, tgt, label="", h=""):
    return {"id": f"e_{src}_{tgt}", "source": src, "target": tgt,
            "label": label, "sourceHandle": h}


def _consume_stream(gen):
    """Consome um generator do api_server inteiro e devolve a lista de
    eventos parseados."""
    events = []
    for chunk in gen:
        line = chunk.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


class _BaseObsTest(unittest.TestCase):
    """Usa um HISTORY_DB temporário para isolar cada teste."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_history.db")
        self._prev_db = os.environ.get("HISTORY_DB_PATH")
        os.environ["HISTORY_DB_PATH"] = self.db_path
        # Forçar recriação do history_db apontando para o novo path
        import importlib
        import history_db
        importlib.reload(history_db)
        history_db.init_db()
        self.history_db = history_db
        # importa api_server (lazy — não precisa subir FastAPI)
        import api_server
        importlib.reload(api_server)
        self.api_server = api_server

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        if self._prev_db is None:
            os.environ.pop("HISTORY_DB_PATH", None)
        else:
            os.environ["HISTORY_DB_PATH"] = self._prev_db


class TestRunRecording(_BaseObsTest):

    def test_successful_run_writes_complete_record(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Etapa A",
                     prompt="x", output_var="oa", model="gpt-5.4-mini"),
                node("b", "agent", label="Etapa B",
                     prompt="y", output_var="ob", model="gpt-5.4-mini"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }
        with patch("backend.get_llm", make_factory(["saida-A", "saida-B"])):
            events = _consume_stream(
                self.api_server._observed_flow_stream(
                    user_id="u1", flow_id="flow-x", flow_name="Teste",
                    input_text="entrada", is_preview=False, config=config,
                )
            )

        # 1) Primeiro evento é run_started
        self.assertEqual(events[0]["event"], "run_started")
        run_id = events[0]["run_id"]
        self.assertTrue(run_id)

        # 2) Run gravado com totais corretos
        runs = self.history_db.list_flow_runs("u1")
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual(r["status"], "completed")
        self.assertEqual(r["flow_id"], "flow-x")
        self.assertEqual(r["flow_name"], "Teste")
        self.assertEqual(r["total_input_tokens"], 80 + 80)
        self.assertEqual(r["total_output_tokens"], 40 + 40)
        self.assertGreater(r["total_cost_usd"], 0)
        self.assertGreater(r["duration_ms"], 0)

        # 3) Detalhe contém eventos
        detail = self.history_db.get_flow_run(run_id, "u1")
        self.assertIsNotNone(detail)
        types = [e["event_type"] for e in detail["events"]]
        self.assertIn("node_start", types)
        self.assertIn("node_done", types)
        self.assertIn("flow_done", types)
        # final_output é a saída do nó conectado ao end (B)
        self.assertEqual(detail["final_output"], "saida-B")

    def test_failed_run_is_marked_error_and_finally_runs(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Boom", prompt="x", output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        with patch("backend.get_llm",
                   make_factory([RuntimeError("LLM caiu")])):
            _consume_stream(
                self.api_server._observed_flow_stream(
                    user_id="u1", flow_id="flow-err", flow_name="Falho",
                    input_text="x", is_preview=False, config=config,
                )
            )

        runs = self.history_db.list_flow_runs("u1")
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual(r["status"], "error")
        self.assertIn("LLM caiu", r["error"])
        self.assertIsNotNone(r["ended_at"])
        # mesmo com erro, duration foi calculada
        self.assertGreater(r["duration_ms"], 0)

    def test_hil_pause_marks_awaiting_human(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="x", output_var="oa"),
                node("h", "hil", label="Aprove"),
                node("b", "agent", label="B", prompt="y", output_var="ob"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "h"),
                      edge("h", "b"), edge("b", "e")],
        }
        with patch("backend.get_llm", make_factory(["RASCUNHO"])):
            _consume_stream(
                self.api_server._observed_flow_stream(
                    user_id="u1", flow_id="flow-hil", flow_name="HIL",
                    input_text="x", is_preview=False, config=config,
                )
            )

        runs = self.history_db.list_flow_runs("u1")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "awaiting_human")

    def test_user_isolation(self):
        """Run do usuário A não vaza para B."""
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="x", output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        with patch("backend.get_llm", make_factory(["x", "y"])):
            _consume_stream(self.api_server._observed_flow_stream(
                user_id="alice", flow_id="f", flow_name="A",
                input_text="i", is_preview=False, config=config))
            _consume_stream(self.api_server._observed_flow_stream(
                user_id="bob", flow_id="f", flow_name="A",
                input_text="i", is_preview=False, config=config))

        self.assertEqual(len(self.history_db.list_flow_runs("alice")), 1)
        self.assertEqual(len(self.history_db.list_flow_runs("bob")), 1)
        # Alice não pode acessar a run de Bob
        bob_run_id = self.history_db.list_flow_runs("bob")[0]["id"]
        self.assertIsNone(self.history_db.get_flow_run(bob_run_id, "alice"))

    def test_delete_run_removes_events(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="x", output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        with patch("backend.get_llm", make_factory(["x"])):
            events = _consume_stream(self.api_server._observed_flow_stream(
                user_id="u1", flow_id="f", flow_name="A",
                input_text="i", is_preview=False, config=config))
        run_id = events[0]["run_id"]

        self.assertTrue(self.history_db.delete_flow_run(run_id, "u1"))
        self.assertIsNone(self.history_db.get_flow_run(run_id, "u1"))
        # E os eventos foram deletados também
        with self.history_db.get_db() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM flow_run_events WHERE run_id = ?", (run_id,)
            )
            self.assertEqual(cur.fetchone()[0], 0)

    def test_metrics_accumulate_across_nodes(self):
        """Tokens totais = soma dos nós; custo > 0 quando preço conhecido."""
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="x", output_var="oa",
                     model="gpt-5.4-mini"),  # preço conhecido
                node("b", "agent", label="B", prompt="y", output_var="ob",
                     model="gpt-5.4-mini"),
                node("c", "agent", label="C", prompt="z", output_var="oc",
                     model="gpt-5.4-mini"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"),
                      edge("b", "c"), edge("c", "e")],
        }
        with patch("backend.get_llm", make_factory(["x", "y", "z"])):
            _consume_stream(self.api_server._observed_flow_stream(
                user_id="u1", flow_id="f", flow_name="A",
                input_text="i", is_preview=False, config=config))
        r = self.history_db.list_flow_runs("u1")[0]
        # 3 nós × 80 input × 40 output (default do _Resp)
        self.assertEqual(r["total_input_tokens"], 3 * 80)
        self.assertEqual(r["total_output_tokens"], 3 * 40)
        # custo > 0 (gpt-5.4-mini precificado)
        self.assertGreater(r["total_cost_usd"], 0)

    def test_preview_flag_persisted(self):
        config = {
            "nodes": [node("s", "start"),
                      node("a", "agent", label="A", prompt="x", output_var="oa"),
                      node("e", "end")],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        with patch("backend.get_llm", make_factory(["x"])):
            _consume_stream(self.api_server._observed_flow_stream(
                user_id="u1", flow_id=None, flow_name="(preview)",
                input_text="i", is_preview=True, config=config))
        r = self.history_db.list_flow_runs("u1")[0]
        self.assertEqual(r["is_preview"], 1)
        self.assertIsNone(r["flow_id"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
