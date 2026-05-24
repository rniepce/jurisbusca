"""
flow_engine.py — Executes user-defined agent flows defined as JSON.

Iterates through nodes in topological order, calling LLMs for agent nodes
and evaluating conditions for router nodes. Yields SSE-formatted event strings.
"""

import json
import os
from typing import Generator


_SAFE_BUILTINS = {
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
}


def _safe_eval(condition: str, state: dict) -> bool:
    try:
        return bool(eval(condition, {"__builtins__": _SAFE_BUILTINS}, state))  # noqa: S307
    except Exception:
        return False


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_and_run_flow(flow_config: dict, input_text: str) -> Generator[str, None, None]:
    """Execute a flow config and yield SSE event strings."""
    nodes = {n["id"]: n for n in flow_config.get("nodes", [])}
    edges = flow_config.get("edges", [])

    # adjacency: node_id -> list of (target_id, edge_label)
    adj: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        src = edge["source"]
        adj.setdefault(src, []).append((edge["target"], edge.get("label", "")))

    state: dict = {"input_text": input_text}

    start = next((n for n in flow_config.get("nodes", []) if n["type"] == "start"), None)
    if not start:
        yield _sse({"event": "error", "message": "Nenhum nó de início encontrado no fluxo."})
        return

    current_id: str | None = start["id"]
    visited: set[str] = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = nodes.get(current_id)
        if not node:
            break

        ntype = node["type"]

        if ntype == "start":
            targets = adj.get(current_id, [])
            current_id = targets[0][0] if targets else None

        elif ntype == "agent":
            cfg = node.get("data", node.get("config", {}))
            label = cfg.get("label") or node["id"]
            model = cfg.get("model") or "gpt-5.3-chat"
            prompt = cfg.get("prompt", "")
            output_var = cfg.get("output_var") or "output"
            knowledge = cfg.get("knowledge", "")

            yield _sse({"event": "node_start", "node_id": current_id, "label": label})

            try:
                import backend as be
                from langchain_core.messages import SystemMessage, HumanMessage

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

                context = "\n\n".join(
                    f"[{k}]:\n{v}" for k, v in state.items() if k != "input_text"
                )
                user_msg = f"Entrada principal:\n{state.get('input_text', input_text)}"
                if context:
                    user_msg = f"{context}\n\n{user_msg}"
                msgs.append(HumanMessage(content=user_msg))

                response = llm.invoke(msgs)
                result = response.content if hasattr(response, "content") else str(response)
                state[output_var] = result

                yield _sse({
                    "event": "node_done",
                    "node_id": current_id,
                    "label": label,
                    "output_var": output_var,
                    "output": result,
                })

            except Exception as exc:
                yield _sse({"event": "node_error", "node_id": current_id, "label": label, "error": str(exc)})
                break

            targets = adj.get(current_id, [])
            current_id = targets[0][0] if targets else None

        elif ntype == "router":
            cfg = node.get("data", node.get("config", {}))
            condition = cfg.get("condition", "True")
            label = cfg.get("label") or "Roteador"

            yield _sse({"event": "node_start", "node_id": current_id, "label": label})

            branch = "true" if _safe_eval(condition, state) else "false"

            yield _sse({"event": "node_done", "node_id": current_id, "label": label, "branch": branch})

            targets = adj.get(current_id, [])
            matched = next((tgt for tgt, lbl in targets if lbl == branch), None)
            if not matched and targets:
                matched = targets[0][0]
            current_id = matched

        elif ntype == "end":
            # Collect the last agent output as the final result
            final_output = state.get("input_text", input_text)
            for k, v in state.items():
                if k != "input_text":
                    final_output = v

            yield _sse({"event": "flow_done", "output": final_output, "state": state})
            return

        else:
            targets = adj.get(current_id, [])
            current_id = targets[0][0] if targets else None

    yield _sse({"event": "flow_done", "output": state.get("output", input_text), "state": state})
