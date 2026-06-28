"""Nós especiais — extractor (retries, tipos), switch (fallback), docx (gera
arquivo), HIL (user_input editado sobrescreve estado)."""
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
    def __init__(self, c):
        self.content = c
        self.usage_metadata = {"input_tokens": 20, "output_tokens": 10}


class _LLM:
    def __init__(self, queue):
        self.queue = list(queue) if not isinstance(queue, str) else None
        self.fixed = queue if isinstance(queue, str) else None

    def invoke(self, _messages):
        if self.queue is not None:
            return _Resp(self.queue.pop(0) if self.queue else "(esgotado)")
        return _Resp(self.fixed)


def make_factory(queue):
    return lambda *a, **k: _LLM(queue)


def edge(src, tgt, label="", source_handle=""):
    return {"id": f"e_{src}_{tgt}", "source": src, "target": tgt,
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
# EXTRACTOR
# ─────────────────────────────────────────────────────────────────────

class TestExtractorVariants(unittest.TestCase):

    def _config(self, fields="numero:string:CNJ|valor:number:R$|partes:array:lista"):
        return {
            "nodes": [
                node("s", "start"),
                node("x", "extractor", label="Extrair", fields=fields,
                     model="gpt-5.4-mini", output_var="dados"),
                node("e", "end"),
            ],
            "edges": [edge("s", "x"), edge("x", "e")],
        }

    def test_all_field_types(self):
        valid = json.dumps({
            "string_f": "abc", "int_f": 42, "num_f": 3.14,
            "bool_f": True, "arr_f": [1, 2, 3],
            "obj_f": {"k": "v"},
        })
        fields = "|".join([
            "string_f:string:s",
            "int_f:integer:i",
            "num_f:number:n",
            "bool_f:boolean:b",
            "arr_f:array:a",
            "obj_f:object:o",
        ])
        with patch("backend.get_llm", make_factory([valid])):
            events = run(self._config(fields=fields), "x")
        final = [e for e in events if e["event"] == "flow_done"][-1]
        st = final["state"]
        self.assertEqual(st["string_f"], "abc")
        self.assertEqual(st["int_f"], 42)
        self.assertEqual(st["num_f"], 3.14)
        # arrays/objects são serializados como JSON string
        self.assertEqual(json.loads(st["arr_f"]), [1, 2, 3])
        self.assertEqual(json.loads(st["obj_f"]), {"k": "v"})

    def test_retries_on_invalid_json_then_succeeds(self):
        bad = "isto não é JSON"
        bad2 = '{"numero": "x"}'  # falta 'valor' e 'partes'
        good = json.dumps({
            "numero": "0001-23.2024.8.13.0024",
            "valor": 1500.5,
            "partes": ["A", "B"],
        })
        with patch("backend.get_llm", make_factory([bad, bad2, good])):
            events = run(self._config(), "x")
        done = [e for e in events if e["event"] == "node_done" and e.get("label") == "Extrair"][-1]
        # extracted_fields inclui as chaves extraídas + a versão JSON formatada
        # sob output_var e output_var_json (estrutura interna do extractor).
        self.assertIn("numero", done["extracted_fields"])
        self.assertIn("valor", done["extracted_fields"])
        self.assertIn("partes", done["extracted_fields"])
        final = [e for e in events if e["event"] == "flow_done"][-1]
        self.assertEqual(final["state"]["numero"], "0001-23.2024.8.13.0024")

    def test_3_attempts_then_fails(self):
        with patch("backend.get_llm", make_factory(["lixo", "lixo", "lixo"])):
            events = run(self._config(), "x")
        errors = [e for e in events if e["event"] == "node_error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("3 tentativas", errors[0]["error"])

    def test_strips_markdown_json_fence(self):
        wrapped = '```json\n{"numero": "X", "valor": 1, "partes": []}\n```'
        with patch("backend.get_llm", make_factory([wrapped])):
            events = run(self._config(), "x")
        self.assertTrue(any(e["event"] == "flow_done" for e in events))


# ─────────────────────────────────────────────────────────────────────
# SWITCH
# ─────────────────────────────────────────────────────────────────────

class TestSwitchBehavior(unittest.TestCase):

    def _config(self):
        return {
            "nodes": [
                node("s", "start"),
                node("sw", "switch", label="Cat",
                     categories="Civil|Penal|Tributário",
                     model="gpt-5.4-mini", output_var="cat"),
                node("civ", "agent", label="Civ", prompt="c", output_var="oc"),
                node("pen", "agent", label="Pen", prompt="p", output_var="op"),
                node("trib", "agent", label="Trib", prompt="t", output_var="ot"),
                node("e", "end"),
            ],
            "edges": [
                edge("s", "sw"),
                edge("sw", "civ", label="Civil", source_handle="civil"),
                edge("sw", "pen", label="Penal", source_handle="penal"),
                edge("sw", "trib", label="Tributário", source_handle="tributário"),
                edge("civ", "e"), edge("pen", "e"), edge("trib", "e"),
            ],
        }

    def test_mixed_case_response_still_matches(self):
        # LLM responde "penal" minúsculo
        with patch("backend.get_llm", make_factory(["penal", "ok-pen"])):
            events = run(self._config(), "x")
        labels = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("Pen", labels)

    def test_response_with_extra_text_still_matches(self):
        # LLM responde com texto extra ("A categoria é: Tributário, justificativa: ...")
        verbose = "A categoria que melhor classifica é Tributário, pois envolve impostos."
        with patch("backend.get_llm", make_factory([verbose, "ok-trib"])):
            events = run(self._config(), "x")
        labels = [e["label"] for e in events if e["event"] == "node_done"]
        self.assertIn("Trib", labels)
        self.assertNotIn("Civ", labels)
        self.assertNotIn("Pen", labels)

    def test_no_match_falls_back_to_first_category(self):
        # LLM responde algo que não bate em nenhuma categoria
        with patch("backend.get_llm", make_factory(["XYZ inesperado", "ok-civ"])):
            events = run(self._config(), "x")
        labels = [e["label"] for e in events if e["event"] == "node_done"]
        # fallback é a primeira categoria (Civil)
        self.assertIn("Civ", labels)


# ─────────────────────────────────────────────────────────────────────
# DOCX
# ─────────────────────────────────────────────────────────────────────

class TestDocxNode(unittest.TestCase):

    def test_docx_creates_file_on_disk(self):
        config = {
            "nodes": [
                node("s", "start"),
                node("a", "agent", label="Redator", prompt="redija",
                     output_var="minuta"),
                node("d", "docx", label="Gerar DOCX",
                     filename="teste_unitario.docx"),
                node("e", "end"),
            ],
            "edges": [edge("s", "a"), edge("a", "d"), edge("d", "e")],
        }
        markdown = (
            "# Sentença\n## Relatório\n"
            "Texto em **negrito** e itálico aqui.\n"
            "- Item 1\n- Item 2\n"
        )
        with patch("backend.get_llm", make_factory([markdown])):
            events = run(config, "x")
        done_docx = [e for e in events if e["event"] == "node_done"
                     and e.get("label") == "Gerar DOCX"]
        self.assertEqual(len(done_docx), 1)
        url = done_docx[0]["download_url"]
        self.assertTrue(url.startswith("/api/flows/docx-download/"))
        # localiza o arquivo no tmpdir
        filename = url.rsplit("/", 1)[-1]
        path = os.path.join(tempfile.gettempdir(), "jurisbusca_docx", filename)
        self.assertTrue(os.path.exists(path), f"docx não foi escrito em {path}")
        self.assertGreater(os.path.getsize(path), 1000,
                           "arquivo docx vazio/quebrado")


# ─────────────────────────────────────────────────────────────────────
# HIL — comportamento adicional (user_input edit)
# ─────────────────────────────────────────────────────────────────────

class TestHILUserInputEdit(unittest.TestCase):
    """Mesmo cenário do teste de resume, mas o usuário edita o rascunho:
    a versão editada precisa sobrescrever a última variável."""

    def test_edited_content_replaces_last_var_on_resume(self):
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
        captured_b_input = []

        # 1ª passada
        with patch("backend.get_llm", make_factory(["RASCUNHO-IA"])):
            ev1 = run(config, "petição")
        hil = [e for e in ev1 if e["event"] == "human_required"][-1]
        state = hil["state"]

        # USUÁRIO EDITA: substitui a última variável não-meta
        edited = "RASCUNHO-EDITADO-POR-HUMANO"
        last_key = None
        for k in state:
            if k != "input_text" and not k.startswith("__"):
                last_key = k
        self.assertIsNotNone(last_key)
        state[last_key] = edited  # simula o que o api_server faz com user_input

        class _LLMCap:
            def invoke(self, msgs):
                captured_b_input.append(msgs[-1].content)
                return _Resp("FINAL-IA")

        with patch("backend.get_llm", lambda *a, **k: _LLMCap()):
            run(config, "", start_from="b", initial_state=state)

        self.assertEqual(len(captured_b_input), 1)
        self.assertIn(edited, captured_b_input[0],
                      "agente após HIL não recebeu a versão editada pelo humano")
        self.assertNotIn("RASCUNHO-IA", captured_b_input[0],
                         "rascunho original (não editado) vazou para o próximo agente")


if __name__ == "__main__":
    unittest.main(verbosity=2)
