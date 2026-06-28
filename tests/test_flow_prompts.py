"""Testes de prompt: o que cada agente *realmente* recebe e devolve.

Estes testes interceptam todas as chamadas do LLM em fluxos multi-agente
e validam:
  - System prompt do nó (substituições, knowledge injetada);
  - User message (estrutura "Entrada principal: ..." + contexto acumulado);
  - Que a saída do agente N chega ao agente N+1 no contexto;
  - Cadeias longas (3+ agentes) preservam todas as saídas anteriores;
  - O conteúdo é exatamente o produzido pelo nó anterior (sem perda/edição);
  - Variáveis @ via {{Label}} sobrevivem a múltiplos saltos.

Rodar com:
    venv312/bin/python -m unittest tests.test_flow_prompts -v
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


# ─────────────────────────────────────────────────────────────────────
# Captura de chamadas
# ─────────────────────────────────────────────────────────────────────

class CapturingLLM:
    """LLM que grava cada invocação para inspeção posterior."""

    def __init__(self, label: str, recipe: Any, registry: list[dict]):
        self.label = label
        self.recipe = recipe
        self.registry = registry

    def invoke(self, messages):
        # Extrai system e user
        sys_content = ""
        user_content = ""
        for m in messages:
            t = type(m).__name__
            if t == "SystemMessage":
                sys_content = m.content
            elif t == "HumanMessage":
                # último HumanMessage é o user_msg
                user_content = m.content
        record = {
            "label": self.label,
            "system": sys_content,
            "user": user_content,
            "raw_messages": messages,
        }
        self.registry.append(record)

        if callable(self.recipe):
            content = self.recipe(record)
        else:
            content = self.recipe

        class _Resp:
            def __init__(self, c):
                self.content = c
                self.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        return _Resp(content)


def make_factory(label_responses: dict, registry: list[dict]):
    """Devolve factory de get_llm que escolhe a resposta pelo prompt content.

    label_responses: {fragmento_no_system_prompt: resposta} — o primeiro que
    bater no system prompt é usado. Permite distinguir agentes por seu prompt.
    """
    def factory(model_name: str = "gpt-5.3-chat", **_):
        def recipe(record: dict):
            for fragment, resp in label_responses.items():
                if fragment in record["system"]:
                    return resp
            return "(resposta-default)"
        return CapturingLLM(model_name, recipe, registry)
    return factory


def edge(src: str, tgt: str, label: str = "", source_handle: str = "") -> dict:
    return {"id": f"e_{src}_{tgt}_{label}", "source": src, "target": tgt,
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
# Tests
# ─────────────────────────────────────────────────────────────────────

class TestSystemPromptDelivery(unittest.TestCase):
    """O system prompt configurado no nó tem que chegar 1:1 ao LLM."""

    def test_system_prompt_is_exactly_what_user_set(self):
        configured_prompt = (
            "Você é um magistrado especialista em direito penal. "
            "Analise as provas e emita parecer fundamentado."
        )
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Magistrado", prompt=configured_prompt,
                     output_var="parecer"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({"Magistrado": "ok"}, calls)):
            run(config, "Fatos do caso X")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["system"], configured_prompt)


class TestKnowledgeBaseInjection(unittest.TestCase):
    """Knowledge configurada no nó deve aparecer dentro do system prompt."""

    def test_knowledge_appears_in_system(self):
        cfg_prompt = "Use o material fornecido."
        knowledge = "DOC X — texto importante para o agente raciocinar"
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt=cfg_prompt,
                     knowledge=knowledge, output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({cfg_prompt: "ok"}, calls)):
            run(config, "input")

        self.assertIn("BASE DE CONHECIMENTO", calls[0]["system"])
        self.assertIn(knowledge, calls[0]["system"])
        # E o prompt original vem ANTES do bloco de knowledge
        self.assertTrue(calls[0]["system"].startswith(cfg_prompt))


class TestUserMessageStructure(unittest.TestCase):
    def test_first_agent_user_message_contains_input(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="proc", output_var="oa"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({"proc": "x"}, calls)):
            run(config, "TEXTO ORIGINAL DO PROCESSO")
        # primeiro agente: user = "Entrada principal:\nTEXTO ORIGINAL..."
        self.assertIn("Entrada principal:", calls[0]["user"])
        self.assertIn("TEXTO ORIGINAL DO PROCESSO", calls[0]["user"])


class TestOutputPropagation(unittest.TestCase):
    """Agente A → B: o que A produziu deve chegar em B."""

    def test_agent_b_receives_agent_a_output(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Triagem",
                     prompt="Faça a triagem do processo.", output_var="triagem"),
                node("b", "agent", label="Minuta",
                     prompt="Com base na triagem, esboce a decisão.",
                     output_var="minuta"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }
        agent_a_out = "TRIAGEM: ação procedente — fundamentos X, Y, Z."

        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({
            "triagem do processo": agent_a_out,
            "esboce a decisão": "MINUTA FINAL",
        }, calls)):
            run(config, "Petição inicial...")

        # Devem ter rodado dois agentes
        self.assertEqual(len(calls), 2)
        b_user = calls[1]["user"]
        # A user msg do agente B precisa conter EXATAMENTE a saída do A
        self.assertIn(agent_a_out, b_user,
                      f"Saída do agente A não chegou na user msg do B. Recebido:\n{b_user}")
        # E o nome da variável (triagem) deve aparecer como rótulo no contexto
        self.assertIn("[triagem]:", b_user)


class TestChainOfThree(unittest.TestCase):
    """A → B → C: C precisa ver as saídas de A e B."""

    def test_third_agent_has_all_prior_outputs(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A1", prompt="prompt-a", output_var="va"),
                node("b", "agent", label="B2", prompt="prompt-b", output_var="vb"),
                node("c", "agent", label="C3", prompt="prompt-c", output_var="vc"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "c"), edge("c", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({
            "prompt-a": "SAÍDA-A: análise prévia",
            "prompt-b": "SAÍDA-B: minuta intermediária",
            "prompt-c": "SAÍDA-C: revisão final",
        }, calls)):
            evs = run(config, "input inicial")

        self.assertEqual(len(calls), 3)

        # B vê A
        self.assertIn("SAÍDA-A: análise prévia", calls[1]["user"])

        # C vê A e B
        self.assertIn("SAÍDA-A: análise prévia", calls[2]["user"])
        self.assertIn("SAÍDA-B: minuta intermediária", calls[2]["user"])

        # ordenação: os rótulos devem aparecer com as chaves corretas
        self.assertIn("[va]:", calls[2]["user"])
        self.assertIn("[vb]:", calls[2]["user"])

        # final do flow é a saída do C
        final = [e for e in evs if e["event"] == "flow_done"][-1]
        self.assertEqual(final["output"], "SAÍDA-C: revisão final")


class TestTemplateLabelSubstitution(unittest.TestCase):
    """`{{Label}}` deve ser substituído pelo conteúdo gerado, mesmo a 2 saltos."""

    def test_double_hop_substitution(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Origem", prompt="gere X",
                     output_var="origem"),
                node("b", "agent", label="Meio", prompt="passe adiante",
                     output_var="meio"),
                node("c", "agent", label="Final",
                     prompt="Cite literalmente: {{Origem}} — e analise {{Meio}}.",
                     output_var="fim"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "c"), edge("c", "e")],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({
            "gere X": "TEXTO-ORIGEM",
            "passe adiante": "TEXTO-MEIO",
            "Cite literalmente": "RESULTADO",  # vai bater no system substituído também
        }, calls)):
            run(config, "input")

        c_system = calls[2]["system"]
        self.assertIn("TEXTO-ORIGEM", c_system,
                      f"Substituição {{{{Origem}}}} falhou. System: {c_system!r}")
        self.assertIn("TEXTO-MEIO", c_system,
                      f"Substituição {{{{Meio}}}} falhou. System: {c_system!r}")
        # E o placeholder original não pode ter sobrado
        self.assertNotIn("{{Origem}}", c_system)
        self.assertNotIn("{{Meio}}", c_system)


class TestNoLeakageOfMetaVars(unittest.TestCase):
    """Chaves __user_id, __token etc. não podem vazar no user_msg."""

    def test_meta_keys_filtered_from_context(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="p", output_var="oa"),
                node("b", "agent", label="B", prompt="p2", output_var="ob"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }
        calls: list[dict] = []
        initial = {"__user_id": "u-123", "__token": "secret-token", "input_text": "x"}
        with patch("backend.get_llm", make_factory({"p": "OA", "p2": "OB"}, calls)):
            run(config, "x", initial_state=initial)

        b_user = calls[1]["user"]
        self.assertNotIn("__user_id", b_user)
        self.assertNotIn("u-123", b_user)
        self.assertNotIn("secret-token", b_user)


class TestParallelOutputsBothReachDownstream(unittest.TestCase):
    """A,B em paralelo → C: C precisa ver A E B."""

    def test_downstream_sees_both_parallel_outputs(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="prompt-a", output_var="va"),
                node("b", "agent", label="B", prompt="prompt-b", output_var="vb"),
                node("c", "agent", label="C", prompt="prompt-c", output_var="vc"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "a"), edge("s", "b"),
                edge("a", "c"), edge("b", "c"),
                edge("c", "e"),
            ],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({
            "prompt-a": "OUT-A",
            "prompt-b": "OUT-B",
            "prompt-c": "OUT-C",
        }, calls)):
            run(config, "input")

        # 3 chamadas: A, B, C (ordem pode variar entre A e B)
        c_call = next((c for c in calls if "prompt-c" in c["system"]), None)
        self.assertIsNotNone(c_call, "agente C não foi chamado")
        self.assertIn("OUT-A", c_call["user"])
        self.assertIn("OUT-B", c_call["user"])


class TestInputTextStaysOriginal(unittest.TestCase):
    """O `input_text` não pode ser mutado por agentes intermediários — eles
    salvam em sua própria output_var. C ainda deve ver o input original."""

    def test_input_text_survives_chain(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="pa", output_var="oa"),
                node("b", "agent", label="B", prompt="pb", output_var="ob"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "b"), edge("b", "e")],
        }
        original_input = "PETIÇÃO ORIGINAL DO PROCESSO 0001-23"
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({"pa": "X", "pb": "Y"}, calls)):
            run(config, original_input)

        # ambos os agentes devem ter a entrada principal intacta
        for c in calls:
            self.assertIn(original_input, c["user"],
                          f"input_text foi alterado para {c['user']!r}")


class TestResumeKeepsContext(unittest.TestCase):
    """Após HIL, o agente seguinte ainda precisa ver as saídas de antes do HIL."""

    def test_post_hil_agent_sees_pre_hil_output(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Rascunho",
                     prompt="esboce", output_var="rascunho"),
                node("h", "hil", label="Aprovar"),
                node("b", "agent", label="Polir",
                     prompt="finalize", output_var="final"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "h"),
                      edge("h", "b"), edge("b", "e")],
        }
        # 1ª passada: pausa em HIL
        calls1: list[dict] = []
        with patch("backend.get_llm", make_factory({"esboce": "RASCUNHO-IA"}, calls1)):
            ev1 = run(config, "petição")
        hil_ev = [e for e in ev1 if e["event"] == "human_required"][-1]
        state = hil_ev["state"]
        self.assertIn("rascunho", state)
        self.assertEqual(state["rascunho"], "RASCUNHO-IA")

        # 2ª passada: resume — o agente "Polir" precisa receber o rascunho
        calls2: list[dict] = []
        with patch("backend.get_llm", make_factory({"finalize": "FINAL-IA"}, calls2)):
            run(config, "", start_from="b", initial_state=state)
        self.assertEqual(len(calls2), 1)
        b_user = calls2[0]["user"]
        self.assertIn("RASCUNHO-IA", b_user,
                      f"Após HIL, o agente seguinte não viu o rascunho. user_msg={b_user}")


class TestRouterDoesNotPolluteDownstream(unittest.TestCase):
    """Se o roteador escolhe ramo true, o ramo false não pode aparecer no
    contexto de quem vem depois."""

    def test_only_chosen_branch_in_context(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="A", prompt="pa", output_var="oa"),
                node("r", "router", label="R", condition="True"),
                node("br_t", "agent", label="T", prompt="pt", output_var="vt"),
                node("br_f", "agent", label="F", prompt="pf", output_var="vf"),
                node("merge", "agent", label="Merge", prompt="merge",
                     output_var="om"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "a"), edge("a", "r"),
                edge("r", "br_t", label="verdadeiro", source_handle="true"),
                edge("r", "br_f", label="falso", source_handle="false"),
                edge("br_t", "merge"),
                edge("merge", "e"),
            ],
        }
        calls: list[dict] = []
        with patch("backend.get_llm", make_factory({
            "pa": "OUT-A",
            "pt": "OUT-T",  # ramo true
            "pf": "OUT-F",  # NÃO deve rodar
            "merge": "OUT-MERGE",
        }, calls)):
            run(config, "x")

        # Falso NÃO deve ter sido chamado
        self.assertFalse(any("pf" in c["system"] for c in calls),
                         "ramo falso foi chamado embora condição seja True")
        # Merge não deve ver OUT-F (pois nunca rodou)
        merge_call = next(c for c in calls if "merge" in c["system"])
        self.assertNotIn("OUT-F", merge_call["user"])
        self.assertIn("OUT-T", merge_call["user"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
