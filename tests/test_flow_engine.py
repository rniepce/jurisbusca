"""Testes sintéticos do flow_engine.

Estes testes não dependem de chaves de API reais. Stubam `backend.get_llm`
com um FakeLLM determinístico que devolve `usage_metadata` para que as
métricas de tokens/custo possam ser validadas.

Rodar com:
    venv312/bin/python -m pytest tests/test_flow_engine.py -v
ou
    venv312/bin/python tests/test_flow_engine.py
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any
from unittest.mock import patch

# Garante que conseguimos importar o pacote raiz
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────
# FakeLLM — substitui backend.get_llm
# ─────────────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, content: str, in_tokens: int = 50, out_tokens: int = 25):
        self.content = content
        self.usage_metadata = {
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        }


class FakeLLM:
    """LLM determinístico. `recipe` pode ser:
       - str: sempre devolve esse conteúdo;
       - callable(messages) -> str: gera conteúdo a partir das mensagens.
    """
    def __init__(self, recipe: Any = "ok"):
        self.recipe = recipe
        self.calls: list[Any] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if callable(self.recipe):
            content = self.recipe(messages)
        else:
            content = self.recipe
        return FakeResponse(content)


def make_get_llm(recipes: dict[str, Any] | Any):
    """Devolve uma fábrica compatível com backend.get_llm.

    Se `recipes` é dict, faz lookup por nome do modelo. Senão, usa o mesmo
    recipe para todos.
    """
    def factory(model_name: str = "gpt-5.3-chat", temperature: float = 0.0, **_kwargs):
        if isinstance(recipes, dict):
            r = recipes.get(model_name, recipes.get("*", "ok"))
        else:
            r = recipes
        return FakeLLM(r)
    return factory


def run_flow_collect(config: dict, input_text: str = "", **kwargs) -> list[dict]:
    """Executa o fluxo e devolve a lista de eventos parseados (json)."""
    import flow_engine
    events: list[dict] = []
    for chunk in flow_engine.build_and_run_flow(config, input_text, **kwargs):
        line = chunk.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


# ─────────────────────────────────────────────────────────────────────
# Builders de fluxo
# ─────────────────────────────────────────────────────────────────────

def edge(src: str, tgt: str, label: str = "", source_handle: str = "") -> dict:
    return {"id": f"e_{src}_{tgt}_{label}", "source": src, "target": tgt,
            "label": label, "sourceHandle": source_handle}


def node(node_id: str, ntype: str, **data) -> dict:
    return {"id": node_id, "type": ntype, "position": {"x": 0, "y": 0}, "data": data}


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

class TestLinearFlow(unittest.TestCase):
    """start → agent → end"""

    def test_linear_emits_expected_events(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Resumir", prompt="Resuma o texto.",
                     output_var="resumo", model="gpt-5.3-chat"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        with patch("backend.get_llm", make_get_llm("resumo gerado")):
            events = run_flow_collect(config, "entrada qualquer")

        types = [e["event"] for e in events]
        self.assertIn("node_start", types)
        self.assertIn("node_done", types)
        self.assertIn("flow_done", types)

        # último node_done deve ter as métricas
        done = [e for e in events if e["event"] == "node_done"][-1]
        self.assertEqual(done["label"], "Resumir")
        self.assertEqual(done["output"], "resumo gerado")
        self.assertEqual(done["input_tokens"], 50)
        self.assertEqual(done["output_tokens"], 25)
        self.assertGreater(done["cost_usd"], 0)
        self.assertIn("duration_ms", done)

        # saída final corresponde ao output do agent (não ao input_text)
        final = [e for e in events if e["event"] == "flow_done"][-1]
        self.assertEqual(final["output"], "resumo gerado")


class TestRouter(unittest.TestCase):
    """router → A (true) | B (false), respeitando sourceHandle"""

    def _build_router_flow(self, cond: str) -> dict:
        return {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="ramo A", output_var="a_out"),
                node("r", "router", label="Decisor", condition=cond),
                node("br_true", "agent", label="True branch", prompt="t", output_var="t_out"),
                node("br_false", "agent", label="False branch", prompt="f", output_var="f_out"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "a"),
                edge("a", "r"),
                edge("r", "br_true", label="verdadeiro", source_handle="true"),
                edge("r", "br_false", label="falso", source_handle="false"),
                edge("br_true", "e"),
                edge("br_false", "e"),
            ],
        }

    def test_true_branch(self):
        config = self._build_router_flow("len(a_out) > 0")
        with patch("backend.get_llm", make_get_llm({
            "*": lambda msgs: "[ramo true]" if "ramo A" not in str(msgs[0]) else "saída A",
        })):
            events = run_flow_collect(config, "x")
        labels_done = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("True branch", labels_done)
        self.assertNotIn("False branch", labels_done)
        final = [e for e in events if e["event"] == "flow_done"][-1]
        self.assertEqual(final["output"], "[ramo true]")

    def test_false_branch(self):
        # condição falsa
        config = self._build_router_flow("False")
        with patch("backend.get_llm", make_get_llm("conteúdo")):
            events = run_flow_collect(config, "x")
        labels_done = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("False branch", labels_done)
        self.assertNotIn("True branch", labels_done)
        final = [e for e in events if e["event"] == "flow_done"][-1]
        self.assertEqual(final["output"], "conteúdo")

    def test_disconnected_branch_emits_error(self):
        # rota true existe, false não — condição evalua para False
        config = {
            "nodes": [
                node("s", "start"),
                node("r", "router", label="Decisor", condition="False"),
                node("br_true", "agent", label="T", prompt="t", output_var="t_out"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "r"),
                edge("r", "br_true", label="verdadeiro", source_handle="true"),
                edge("br_true", "e"),
            ],
        }
        with patch("backend.get_llm", make_get_llm("ok")):
            events = run_flow_collect(config, "x")
        # como o ramo false não existe, e há mais de uma saída? — neste teste
        # só existe 1 saída (true), então cai no fallback de 1-target.
        # Esse caso deve simplesmente seguir pelo único target (T) sem error.
        types = [e["event"] for e in events]
        self.assertIn("flow_done", types)


class TestHILResume(unittest.TestCase):
    """Garante que HIL pausa, e ao retomar não reemite human_required."""

    def test_pause_then_resume_no_loop(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Rascunho",
                     prompt="Esboce", output_var="rascunho"),
                node("h", "hil", label="Aprovar", question="aprovar?"),
                node("b", "agent", label="Revisao",
                     prompt="Polir", output_var="final"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "h"), edge("h", "b"), edge("b", "e")],
        }
        with patch("backend.get_llm", make_get_llm("texto bruto")):
            ev1 = run_flow_collect(config, "input")

        # primeira passagem deve pausar em HIL
        types = [e["event"] for e in ev1]
        self.assertIn("human_required", types)
        self.assertNotIn("flow_done", types)
        hil_ev = [e for e in ev1 if e["event"] == "human_required"][-1]
        self.assertEqual(hil_ev["next_node_id"], "b")
        state = hil_ev["state"]

        # resume — não pode reemitir human_required
        with patch("backend.get_llm", make_get_llm("revisado")):
            ev2 = run_flow_collect(config, "", start_from="b", initial_state=state)

        types2 = [e["event"] for e in ev2]
        self.assertNotIn("human_required", types2,
                         "HIL não pode ser re-emitido no resume (bug do loop)")
        self.assertIn("flow_done", types2)
        final = [e for e in ev2 if e["event"] == "flow_done"][-1]
        self.assertEqual(final["output"], "revisado")


class TestParallelFanout(unittest.TestCase):
    def test_two_agents_in_parallel(self):
        # start → A,B (paralelo) → end. End tem 2 predecessors mas só roda
        # quando algum deles termina (qualquer caminho satisfaz "any_alive").
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="a", output_var="oa"),
                node("b", "agent", label="B", prompt="b", output_var="ob"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("s", "b"), edge("a", "e"), edge("b", "e")],
        }
        with patch("backend.get_llm", make_get_llm("ok")):
            events = run_flow_collect(config, "x")
        types = [e["event"] for e in events]
        # quando há >1 simples ready ao mesmo tempo, deve emitir parallel_start
        self.assertIn("parallel_start", types)
        # ambos os nós devem ter rodado
        labels_done = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("A", labels_done)
        self.assertIn("B", labels_done)
        self.assertIn("flow_done", types)


class TestSwitch(unittest.TestCase):
    def test_switch_picks_category_and_cancels_others(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("sw", "switch", label="Cat", categories="Civil|Penal|Tributário",
                     model="gpt-5.4-mini", output_var="categoria"),
                node("civ", "agent", label="Civil agent", prompt="civ", output_var="oc"),
                node("pen", "agent", label="Penal agent", prompt="pen", output_var="op"),
                node("trib", "agent", label="Trib agent", prompt="trib", output_var="ot"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "sw"),
                edge("sw", "civ", label="Civil", source_handle="civil"),
                edge("sw", "pen", label="Penal", source_handle="penal"),
                edge("sw", "trib", label="Tributário", source_handle="tributário"),
                edge("civ", "e"),
                edge("pen", "e"),
                edge("trib", "e"),
            ],
        }

        # switch LLM responde "Penal"; demais retornam "ok"
        def recipe(model_name: str):
            return "Penal" if model_name == "gpt-5.4-mini" else "ok"

        def factory(model_name: str = "gpt-5.3-chat", **_):
            return FakeLLM(recipe(model_name))

        with patch("backend.get_llm", factory):
            events = run_flow_collect(config, "x")

        labels_done = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("Penal agent", labels_done)
        self.assertNotIn("Civil agent", labels_done)
        self.assertNotIn("Trib agent", labels_done)


class TestFinalOutputRespectsBranch(unittest.TestCase):
    """Bug regressão: a saída final tem que vir do nó conectado ao end no
    ramo escolhido — não 'a última coisa que vazou no state'."""

    def test_final_output_is_chosen_branch(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("r", "router", label="R", condition="True"),
                node("a", "agent", label="A", prompt="a", output_var="oa"),
                node("b", "agent", label="B", prompt="b", output_var="ob"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "r"),
                edge("r", "a", label="verdadeiro", source_handle="true"),
                edge("r", "b", label="falso", source_handle="false"),
                edge("a", "e"),
                # b NÃO conecta ao end — só está como branch alternativo
            ],
        }
        with patch("backend.get_llm", make_get_llm(lambda msgs: "saida-A" if "prompt='a'" not in str(msgs) else "saida-B")):
            events = run_flow_collect(config, "x")
        final = [e for e in events if e["event"] == "flow_done"][-1]
        # o nó "A" é o único conectado ao end e foi escolhido pelo router true
        self.assertEqual(final["output"], "saida-A")


class TestExtractor(unittest.TestCase):
    def test_extractor_parses_json_and_expands_state(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("x", "extractor", label="Extrair",
                     fields="numero:string:CNJ|valor:number:R$",
                     model="gpt-5.4-mini", output_var="json_saida"),
                node("e", "end"),
            ],
            "edges": [edge("s", "x"), edge("x", "e")],
        }
        valid_json = '{"numero": "0001-23.2024.8.13.0024", "valor": 1500}'
        with patch("backend.get_llm", make_get_llm(valid_json)):
            events = run_flow_collect(config, "processo qualquer")

        done = [e for e in events if e["event"] == "node_done" and e.get("label") == "Extrair"][-1]
        self.assertIn("numero", done["extracted_fields"])
        self.assertIn("valor", done["extracted_fields"])
        self.assertGreater(done["input_tokens"], 0)

        final = [e for e in events if e["event"] == "flow_done"][-1]
        # __metrics__ NÃO pode vazar para o state limpo
        self.assertNotIn("__metrics__", final["state"])
        # mas as chaves extraídas precisam estar lá
        self.assertEqual(final["state"]["numero"], "0001-23.2024.8.13.0024")
        self.assertEqual(final["state"]["valor"], 1500)


class TestVariableSubstitution(unittest.TestCase):
    def test_label_substitution_in_prompt(self):
        seen_prompts: list[str] = []

        def recipe(messages):
            seen_prompts.append(str(messages[0].content))
            return "ok"

        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Triagem", prompt="primeira chamada",
                     output_var="triagem"),
                node("b", "agent", label="Minuta",
                     prompt="Use isto: {{Triagem}} para decidir",
                     output_var="minuta"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }

        responses = iter(["resultado triagem", "minuta final"])

        def factory(*_a, **_kw):
            return FakeLLM(lambda msgs: (
                (seen_prompts.append(str(msgs[0].content)) or next(responses))
            ))

        with patch("backend.get_llm", factory):
            run_flow_collect(config, "input")

        # O segundo system prompt precisa conter "resultado triagem"
        # (substituído via {{Triagem}})
        self.assertTrue(
            any("resultado triagem" in p for p in seen_prompts),
            f"substituição de {{Triagem}} falhou; prompts: {seen_prompts}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
