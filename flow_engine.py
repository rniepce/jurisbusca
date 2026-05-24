"""
flow_engine.py — Executes user-defined agent flows defined as JSON.

Suporta os seguintes tipos de nó:
  • start, end
  • agent  — chama LLM com prompt
  • router — bifurcação true/false via condição Python
  • switch — classificador multi-saída via LLM
  • hil    — Human-in-the-loop (pausa e emite evento, encerra stream)
  • docx   — converte markdown da entrada em arquivo .docx
  • juris  — RAG sobre acordãos do TJMG
  • modelo — RAG sobre templates do usuário
  • estilo — aplica style dossier do usuário

Substituições de variáveis: `{{Label do Nó}}` no prompt é substituído pela saída
do nó cujo label bate (case-insensitive).
"""

import json
import os
import re
import base64
import uuid
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator


_SAFE_BUILTINS = {
    "len": len, "int": int, "float": float, "str": str, "bool": bool,
    "abs": abs, "min": min, "max": max, "round": round,
}


def _safe_eval(condition: str, state: dict) -> bool:
    try:
        return bool(eval(condition, {"__builtins__": _SAFE_BUILTINS}, state))  # noqa: S307
    except Exception:
        return False


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _build_label_map(nodes_list: list) -> dict[str, str]:
    """Retorna {label_normalizado: output_var} para substituição em prompts."""
    m: dict[str, str] = {}
    for n in nodes_list:
        cfg = n.get("data", n.get("config", {}))
        label = (cfg.get("label") or "").strip().lower()
        ovar = cfg.get("output_var") or n.get("id")
        if label and ovar:
            m[label] = ovar
    return m


def _substitute_vars(text: str, state: dict, label_map: dict[str, str]) -> str:
    """Substitui {{Label do Nó}} pelo conteúdo armazenado em state[output_var]."""
    if not text or "{{" not in text:
        return text

    def replace(match: re.Match) -> str:
        key = match.group(1).strip().lower()
        ovar = label_map.get(key)
        if ovar and ovar in state:
            return str(state[ovar])
        return match.group(0)

    return re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", replace, text)


def _build_user_message(state: dict, input_text: str) -> str:
    """Constrói a mensagem do usuário com input principal + contexto de nós anteriores."""
    context = "\n\n".join(
        f"[{k}]:\n{v}" for k, v in state.items()
        if k != "input_text" and not k.startswith("__")
    )
    user_msg = f"Entrada principal:\n{state.get('input_text', input_text)}"
    if context:
        user_msg = f"{context}\n\n{user_msg}"
    return user_msg


# ─────────────────────────────────────────────────────────────────────
# NÓS ESPECIAIS — handlers individuais
# ─────────────────────────────────────────────────────────────────────

def _run_agent(node, cfg, state, input_text, label_map):
    """Agente LLM padrão. Retorna (output_str, output_var) ou levanta exceção."""
    import backend as be
    from langchain_core.messages import SystemMessage, HumanMessage

    model = cfg.get("model") or "gpt-5.3-chat"
    prompt = cfg.get("prompt", "")
    knowledge = cfg.get("knowledge", "")
    output_var = cfg.get("output_var") or node["id"]

    # Substitui {{Label}} pelos valores reais
    prompt = _substitute_vars(prompt, state, label_map)

    llm = be.get_llm(model, temperature=0.3)

    full_prompt = prompt
    if knowledge:
        full_prompt = (
            (prompt + "\n\n" if prompt else "")
            + "BASE DE CONHECIMENTO (use como referência para responder):\n"
            + knowledge[:80_000]
        )

    msgs = []
    if full_prompt:
        msgs.append(SystemMessage(content=full_prompt))
    msgs.append(HumanMessage(content=_build_user_message(state, input_text)))

    response = llm.invoke(msgs)
    result = response.content if hasattr(response, "content") else str(response)
    return result, output_var


def _run_switch(node, cfg, state, input_text, label_map):
    """Classificador LLM multi-saída. Retorna a categoria escolhida."""
    import backend as be
    from langchain_core.messages import SystemMessage, HumanMessage

    model = cfg.get("model") or "gpt-5.4-mini"
    raw_cats = cfg.get("categories", "")
    categories = [c.strip() for c in raw_cats.split("|") if c.strip()]
    if not categories:
        raise ValueError("Switch sem categorias definidas.")

    llm = be.get_llm(model, temperature=0.0)
    cats_str = "\n".join(f"- {c}" for c in categories)
    sys_msg = SystemMessage(content=(
        "Você é um classificador. Analise a entrada e responda APENAS com "
        "exatamente um item da lista de categorias abaixo, sem nenhum texto extra:\n\n"
        f"{cats_str}"
    ))
    user_msg = HumanMessage(content=_build_user_message(state, input_text))
    response = llm.invoke([sys_msg, user_msg])
    raw = (response.content if hasattr(response, "content") else str(response)).strip()

    # Encontra a categoria que melhor bate (case-insensitive, prefix)
    chosen = next((c for c in categories if c.lower() in raw.lower()), categories[0])
    return chosen


def _run_docx(node, cfg, state, input_text, label_map):
    """Converte o markdown da última saída em .docx, salva em /tmp e devolve URL."""
    from docx import Document  # python-docx
    from docx.shared import Pt

    # Pega o último conteúdo gerado (output do nó anterior)
    last_output = ""
    for k, v in list(state.items())[::-1]:
        if k != "input_text" and not k.startswith("__"):
            last_output = str(v)
            break
    if not last_output:
        last_output = state.get("input_text", input_text)

    filename = cfg.get("filename") or "documento.docx"
    if not filename.endswith(".docx"):
        filename += ".docx"

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    for raw_line in last_output.split("\n"):
        line = raw_line.rstrip()
        if not line.strip():
            doc.add_paragraph("")
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith(("- ", "* ")):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        else:
            p = doc.add_paragraph()
            # negrito simples **texto**
            parts = re.split(r"(\*\*[^*]+\*\*)", line)
            for part in parts:
                if part.startswith("**") and part.endswith("**"):
                    r = p.add_run(part[2:-2])
                    r.bold = True
                else:
                    p.add_run(part)

    out_dir = os.path.join(tempfile.gettempdir(), "jurisbusca_docx")
    os.makedirs(out_dir, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    out_path = os.path.join(out_dir, f"{file_id}__{filename}")
    doc.save(out_path)

    url = f"/api/flows/docx-download/{file_id}__{filename}"
    return f"📄 **Documento gerado:** [{filename}]({url})", "docx_url", url


def _run_juris(node, cfg, state, input_text, label_map):
    """Busca semântica em acordãos TJMG via jurisprudence_search."""
    query = (cfg.get("query") or "").strip()
    query = _substitute_vars(query, state, label_map)
    if not query:
        query = state.get("input_text", input_text)[:500]
    top_k = int(cfg.get("top_k") or 5)

    results = []
    try:
        import jurisprudence_search as js
        if not js.is_available():
            return "(banco de jurisprudência indisponível)", cfg.get("output_var") or node["id"]
        # tenta semantic_search; cai pra search FTS se embeddings indisponíveis
        try:
            hits = js.semantic_search(query, limit=top_k)
        except Exception:
            hits = js.search(query, limit=top_k)
        for i, h in enumerate(hits or []):
            num = h.get("numero_processo") or h.get("id") or "—"
            tipo = h.get("tipo_recurso") or "—"
            ementa = h.get("ementa") or h.get("text") or ""
            results.append(f"### Acordão {i+1} — {tipo} {num}\n{str(ementa)[:1200]}")
    except Exception as exc:
        results = [f"(erro ao buscar jurisprudência: {exc})"]

    text = "\n\n".join(results) if results else "(nenhum resultado)"
    return text, cfg.get("output_var") or node["id"]


def _run_modelo(node, cfg, state, input_text, label_map):
    """Busca semântica nos templates do usuário via backend.search_templates."""
    import backend as be

    query = (cfg.get("query") or "").strip()
    query = _substitute_vars(query, state, label_map)
    if not query:
        query = state.get("input_text", input_text)[:500]
    top_k = int(cfg.get("top_k") or 3)
    user_id = state.get("__user_id", "default")
    token = state.get("__token", "")

    try:
        hits = be.search_templates(user_id, query, k=top_k, token=token)
        if not hits:
            return "(nenhum modelo encontrado)", cfg.get("output_var") or node["id"]
        text = "\n\n---\n\n".join(
            f"### Modelo {i+1}: {h.get('filename') or h.get('name') or 'sem nome'}\n{(h.get('text') or h.get('content') or '')[:1500]}"
            for i, h in enumerate(hits)
        )
    except Exception as exc:
        text = f"(erro ao buscar modelos: {exc})"

    return text, cfg.get("output_var") or node["id"]


def _run_estilo(node, cfg, state, input_text, label_map):
    """Reescreve o último texto aplicando o style dossier do usuário."""
    import backend as be
    from langchain_core.messages import SystemMessage, HumanMessage

    style_dossier = state.get("__style_dossier", "")
    last_output = ""
    for k, v in list(state.items())[::-1]:
        if k != "input_text" and not k.startswith("__"):
            last_output = str(v)
            break
    if not last_output:
        last_output = state.get("input_text", input_text)

    if not style_dossier:
        return last_output + "\n\n*(style dossier não configurado — texto retornado sem ajuste)*", cfg.get("output_var") or node["id"]

    llm = be.get_llm("claude-sonnet-4-6", temperature=0.4)
    sys_msg = SystemMessage(content=(
        "Você é um editor de texto jurídico. Reescreva o texto abaixo aplicando "
        "rigorosamente o estilo de escrita descrito a seguir, sem alterar o conteúdo "
        "factual ou os dispositivos legais citados.\n\n"
        f"ESTILO ALVO:\n{style_dossier[:20_000]}"
    ))
    response = llm.invoke([sys_msg, HumanMessage(content=last_output)])
    result = response.content if hasattr(response, "content") else str(response)
    return result, cfg.get("output_var") or node["id"]


# ─────────────────────────────────────────────────────────────────────
# Worker thread-safe para nós "simples" (executáveis em paralelo)
# ─────────────────────────────────────────────────────────────────────

SIMPLE_NODE_TYPES = {"agent", "juris", "modelo", "estilo", "docx"}

_SPECIAL_HANDLERS = {
    "juris": _run_juris,
    "modelo": _run_modelo,
    "estilo": _run_estilo,
}


def _execute_simple_node(node, state_snapshot, input_text, label_map):
    """Executa um nó simples e retorna (eventos, state_updates).

    Roda em qualquer thread. Recebe um snapshot do state (cópia) e devolve só
    as variáveis novas que devem ser mergidas no state global.
    """
    node_id = node["id"]
    ntype = node["type"]
    cfg = node.get("data", node.get("config", {}))
    label = cfg.get("label") or node_id

    events: list[dict] = [{"event": "node_start", "node_id": node_id, "label": label}]
    updates: dict = {}

    try:
        if ntype == "agent":
            output, ovar = _run_agent(node, cfg, state_snapshot, input_text, label_map)
            updates[ovar] = output
            events.append({"event": "node_done", "node_id": node_id, "label": label,
                           "output": output, "output_var": ovar})

        elif ntype == "docx":
            output, ovar, url = _run_docx(node, cfg, state_snapshot, input_text, label_map)
            updates[ovar] = url
            updates["__docx_url"] = url
            events.append({"event": "node_done", "node_id": node_id, "label": label,
                           "output": output, "output_var": ovar, "download_url": url})

        elif ntype in _SPECIAL_HANDLERS:
            output, ovar = _SPECIAL_HANDLERS[ntype](node, cfg, state_snapshot, input_text, label_map)
            updates[ovar] = output
            events.append({"event": "node_done", "node_id": node_id, "label": label,
                           "output": output, "output_var": ovar})

    except Exception as exc:
        events.append({"event": "node_error", "node_id": node_id, "label": label, "error": str(exc)})
        updates["__error__"] = str(exc)

    return events, updates


# ─────────────────────────────────────────────────────────────────────
# MAIN: build_and_run_flow
# ─────────────────────────────────────────────────────────────────────

def build_and_run_flow(
    flow_config: dict,
    input_text: str,
    *,
    start_from: str | None = None,
    initial_state: dict | None = None,
) -> Generator[str, None, None]:
    """Executa o fluxo com scheduler topológico (suporta fan-out paralelo)."""
    nodes_list = flow_config.get("nodes", [])
    nodes = {n["id"]: n for n in nodes_list}
    edges = flow_config.get("edges", [])
    label_map = _build_label_map(nodes_list)

    # adjacency: source -> list of (target, label, source_handle)
    adj: dict[str, list[tuple[str, str, str]]] = {}
    # predecessors: target -> list of (source, source_handle)
    predecessors: dict[str, list[tuple[str, str]]] = {n["id"]: [] for n in nodes_list}
    for e in edges:
        src, tgt = e["source"], e["target"]
        h = e.get("sourceHandle", "") or ""
        adj.setdefault(src, []).append((tgt, e.get("label", ""), h))
        predecessors[tgt].append((src, h))

    state: dict = dict(initial_state or {})
    state.setdefault("input_text", input_text)
    state_lock = threading.Lock()

    completed: set[str] = set()
    cancelled: set[str] = set()

    # Pre-marca como completed: tudo na initial_state (resume) que já tem chave igual a output_var
    if start_from and initial_state:
        for n in nodes_list:
            ovar = (n.get("data") or {}).get("output_var") or n["id"]
            if ovar in state and n["id"] != start_from:
                completed.add(n["id"])

    # determina start
    start_node = next((n for n in nodes_list if n["type"] == "start"), None)
    start_id = start_from or (start_node["id"] if start_node else None)
    if not start_id:
        yield _sse({"event": "error", "message": "Nenhum nó de início encontrado no fluxo."})
        return

    # Para resume, podemos precisar pular o start
    if start_from:
        # marca start como completed para liberar successors
        if start_node:
            completed.add(start_node["id"])

    def _cancel_subtree(skipped_targets: list[str]):
        """Propaga cancelamento: nós cujos predecessors todos viraram cancelled."""
        stack = list(skipped_targets)
        while stack:
            n = stack.pop()
            if n in cancelled or n in completed:
                continue
            preds = predecessors.get(n, [])
            if preds and all(p in cancelled or (p, _) in [] for p, _ in preds):
                # se todos os predecessors estão cancelled, cancela este também
                if all(p in cancelled for p, _ in preds):
                    cancelled.add(n)
                    for tgt, _l, _h in adj.get(n, []):
                        stack.append(tgt)

    def _ready_nodes() -> list[str]:
        out: list[str] = []
        for n_id in nodes:
            if n_id in completed or n_id in cancelled:
                continue
            preds = predecessors[n_id]
            if not preds:
                if n_id == start_id and start_id not in completed:
                    out.append(n_id)
                continue
            all_decided = all(p in completed or p in cancelled for p, _ in preds)
            any_alive = any(p in completed for p, _ in preds)
            if all_decided and any_alive:
                out.append(n_id)
        return out

    def _emit_final():
        # pega o último output não-meta
        final_output = state.get("input_text", input_text)
        for k, v in state.items():
            if k != "input_text" and not k.startswith("__"):
                final_output = v
        return _sse({"event": "flow_done", "output": final_output, "state": state})

    # ── Loop principal ──────────────────────────────────────────────
    safety = 0
    while True:
        safety += 1
        if safety > 200:  # circuit breaker
            yield _sse({"event": "error", "message": "Loop overflow no fluxo."})
            return

        ready = _ready_nodes()
        if not ready:
            break

        # processa nós "control flow" um a um (não dá pra paralelizar decisões)
        control_ids = [r for r in ready if nodes[r]["type"] in ("start", "end", "router", "switch", "hil")]
        simple_ids = [r for r in ready if nodes[r]["type"] in SIMPLE_NODE_TYPES]

        # ── 1. START / END / ROUTER / SWITCH / HIL — sequencial ────
        if control_ids:
            current_id = control_ids[0]
            node = nodes[current_id]
            ntype = node["type"]
            cfg = node.get("data", node.get("config", {}))
            label = cfg.get("label") or current_id

            if ntype == "start":
                completed.add(current_id)
                continue

            if ntype == "end":
                yield _emit_final()
                return

            if ntype == "router":
                condition = cfg.get("condition", "True")
                yield _sse({"event": "node_start", "node_id": current_id, "label": label})
                branch = "true" if _safe_eval(condition, state) else "false"
                yield _sse({"event": "node_done", "node_id": current_id, "label": label, "branch": branch})
                targets = adj.get(current_id, [])
                chosen_target = next(
                    (tgt for tgt, lbl, _h in targets if lbl == branch or (lbl == "" and branch == "true")),
                    targets[0][0] if targets else None,
                )
                # cancela os outros
                for tgt, _l, _h in targets:
                    if tgt != chosen_target:
                        cancelled.add(tgt)
                        _cancel_subtree([tgt])
                completed.add(current_id)
                continue

            if ntype == "switch":
                yield _sse({"event": "node_start", "node_id": current_id, "label": label})
                try:
                    chosen = _run_switch(node, cfg, state, input_text, label_map)
                    state[cfg.get("output_var") or current_id] = chosen
                except Exception as exc:
                    yield _sse({"event": "node_error", "node_id": current_id, "label": label, "error": str(exc)})
                    return
                yield _sse({"event": "node_done", "node_id": current_id, "label": label,
                            "branch": chosen, "output": chosen})
                targets = adj.get(current_id, [])
                chosen_target = next(
                    (tgt for tgt, _l, h in targets if h.lower() == chosen.lower()),
                    targets[0][0] if targets else None,
                )
                for tgt, _l, _h in targets:
                    if tgt != chosen_target:
                        cancelled.add(tgt)
                        _cancel_subtree([tgt])
                completed.add(current_id)
                continue

            if ntype == "hil":
                yield _sse({"event": "node_start", "node_id": current_id, "label": label})
                last_output = state.get("input_text", input_text)
                for k, v in state.items():
                    if k != "input_text" and not k.startswith("__"):
                        last_output = v
                targets = adj.get(current_id, [])
                next_id = targets[0][0] if targets else None
                yield _sse({
                    "event": "human_required",
                    "node_id": current_id,
                    "label": label,
                    "question": cfg.get("question") or "Aprove para continuar.",
                    "content": last_output,
                    "next_node_id": next_id,
                    "state": {k: v for k, v in state.items() if not k.startswith("__")},
                })
                return

        # ── 2. nós simples — paralelo se mais de 1 ─────────────────
        elif simple_ids:
            # snapshot atual do state (cada thread recebe sua cópia)
            with state_lock:
                snapshot = dict(state)

            if len(simple_ids) == 1:
                # single — roda inline (sem overhead de thread)
                events, updates = _execute_simple_node(
                    nodes[simple_ids[0]], snapshot, input_text, label_map
                )
                for ev in events:
                    yield _sse(ev)
                if "__error__" in updates:
                    return
                with state_lock:
                    state.update(updates)
                completed.add(simple_ids[0])
            else:
                # FAN-OUT paralelo
                yield _sse({
                    "event": "parallel_start",
                    "node_ids": simple_ids,
                    "labels": [nodes[n].get("data", {}).get("label") or n for n in simple_ids],
                })
                with ThreadPoolExecutor(max_workers=min(len(simple_ids), 8)) as pool:
                    futures = {
                        pool.submit(_execute_simple_node, nodes[n_id], snapshot, input_text, label_map): n_id
                        for n_id in simple_ids
                    }
                    for fut in as_completed(futures):
                        n_id = futures[fut]
                        events, updates = fut.result()
                        for ev in events:
                            yield _sse(ev)
                        if "__error__" in updates:
                            return
                        with state_lock:
                            state.update(updates)
                        completed.add(n_id)
                yield _sse({"event": "parallel_done"})
        else:
            # ready vazio depois das filtragens — algo errado
            break

    yield _emit_final()
