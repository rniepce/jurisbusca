"""Deep Research over a single uploaded process.

Pipeline (each step yields SSE events to the client):
    1. chunk the uploaded text (HybridSemanticChunker)
    2. embed into an ephemeral in-memory Chroma collection
    3. ask the model to plan 8-12 investigation questions
    4. for each question: semantic-search top-K chunks + LLM synthesis with citations
    5. consolidate everything into a structured markdown dossier

Designed to handle large processes (200+ pages) without blowing the context window,
since each question only retrieves the K most relevant chunks instead of the full text.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Iterator

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

import backend as be

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.3-chat"
TOP_K = 5                  # chunks retrieved per question
MAX_QUESTIONS = 12
PLANNER_EXCERPT_CHARS = 8000   # how much of the text we show the planner


# ── Prompts ────────────────────────────────────────────────────────────────

PROMPT_PLANNER = """Você é um magistrado experiente investigando um processo judicial complexo \
antes de redigir qualquer decisão.

Sua tarefa: gerar uma lista de **8 a 12 perguntas de investigação** que cobrirão todos \
os aspectos relevantes deste processo. As perguntas devem ser específicas, ordenadas do \
geral para o específico, e cobrir:

- Identificação processual (partes, classe, valor, fase)
- Histórico processual (citação, contestação, audiências, perícias)
- Pedidos e causa de pedir
- Defesa (preliminares e mérito)
- Provas produzidas
- Questões controvertidas pendentes
- Aplicação de jurisprudência vinculante ou súmulas
- Possíveis preliminares de ordem pública (prescrição, decadência, ilegitimidade, AJG)
- Pontos críticos para a decisão (risco de embargos, atenção especial)

Devolva APENAS um JSON no formato:
{"questions": ["pergunta 1?", "pergunta 2?", ...]}

Não inclua explicações fora do JSON."""


PROMPT_RESEARCH = """Você é um analista jurídico investigando um processo. A pergunta abaixo \
precisa ser respondida com base **estritamente** nos trechos do processo fornecidos.

PERGUNTA:
{question}

TRECHOS RECUPERADOS DO PROCESSO:
{chunks}

Instruções:
- Responda em 1-3 parágrafos objetivos.
- Cite o trecho de onde a informação foi extraída usando [Trecho N].
- Se a informação NÃO está nos trechos, responda "Não foi possível localizar no processo." (não invente).
- Mantenha linguagem técnica jurídica."""


PROMPT_DOSSIER = """Você recebeu um conjunto de perguntas e respostas sobre um processo judicial.
Consolide tudo num **dossiê estruturado** em markdown que servirá de base para o magistrado \
decidir o processo.

ESTRUTURA OBRIGATÓRIA:

# Dossiê de Análise Processual

## 1. Identificação
(partes, classe, valor da causa, fase atual)

## 2. Síntese dos fatos
(narrativa concisa)

## 3. Histórico processual
(fases relevantes)

## 4. Pedidos
(lista)

## 5. Defesa
(preliminares + mérito)

## 6. Provas produzidas
(o que foi efetivamente juntado/produzido)

## 7. Questões controvertidas
(o que ainda precisa ser decidido)

## 8. Preliminares de ordem pública
(prescrição, decadência, ilegitimidade, AJG — se aplicáveis)

## 9. Jurisprudência aplicável
(se identificada nos autos)

## 10. Pontos críticos para a decisão
(risco de embargos, atenção especial, recomendações)

---

PERGUNTAS E RESPOSTAS COLETADAS:
{qa_pairs}

Regras:
- Preserve as citações [Trecho N] das respostas individuais para manter a rastreabilidade.
- Se uma seção não tiver informação suficiente, escreva "Não localizado nos autos analisados."
- Seja conciso mas completo."""


# ── Helpers ────────────────────────────────────────────────────────────────

def _event(event_type: str, **data) -> str:
    """Format an SSE event line (single-line JSON to keep parsing trivial)."""
    payload = {"event": event_type, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _parse_questions(plan_text: str) -> list[str]:
    """Extract a question list from the planner output, tolerating loose formats."""
    json_match = re.search(r'\{[\s\S]*"questions"[\s\S]*\}', plan_text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            qs = data.get("questions", [])
            if isinstance(qs, list):
                return [q.strip() for q in qs if isinstance(q, str) and q.strip()][:MAX_QUESTIONS]
        except json.JSONDecodeError:
            pass
    # Fallback: bullet/numbered lines ending in '?'
    questions = []
    for line in plan_text.splitlines():
        m = re.match(r'^\s*(?:\d+[\.\)]|[-*])\s*(.+\?)\s*$', line)
        if m:
            questions.append(m.group(1).strip())
    return questions[:MAX_QUESTIONS]


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_deep_research(
    text: str,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    top_k: int = TOP_K,
) -> Iterator[str]:
    """Run the deep research pipeline. Yields SSE-formatted progress events.

    Events:
        phase           — high-level phase ("chunking", "embedding", "planning", "synthesizing")
        plan            — emitted once with the full question list
        question_start  — beginning of one research question
        question_done   — answer for one research question
        question_error  — non-fatal failure on a single question
        done            — final dossier ready
        error           — fatal failure
    """
    try:
        # ── 1. Chunk ──
        yield _event("phase", phase="chunking", message="Dividindo o processo em trechos semânticos...")
        from chunking import HybridSemanticChunker
        chunker = HybridSemanticChunker(api_key=api_key, provider="azure")
        chunks = chunker.split_text(text)
        for i, doc in enumerate(chunks, start=1):
            doc.metadata["chunk_id"] = i
        logger.info("deep-research: %d chunks generated", len(chunks))

        # ── 2. Embed into ephemeral Chroma ──
        yield _event("phase", phase="embedding", message=f"Indexando {len(chunks)} trechos...")
        embedding_fn = be.get_embedding_function(api_key=api_key)
        collection_name = f"deep_research_{uuid.uuid4().hex[:10]}"
        vector_store = Chroma(
            embedding_function=embedding_fn,
            collection_name=collection_name,
        )
        # Batch to be polite with Azure rate limits
        BATCH = 32
        for i in range(0, len(chunks), BATCH):
            vector_store.add_documents(chunks[i:i + BATCH])

        # ── 3. Plan investigation questions ──
        yield _event("phase", phase="planning", message="Planejando perguntas de investigação...")
        llm = be.get_llm(model_name=model_name, temperature=0.3, api_key=api_key)
        plan_response = llm.invoke([
            SystemMessage(content=PROMPT_PLANNER),
            HumanMessage(content=f"Início do processo (primeiros {PLANNER_EXCERPT_CHARS} caracteres):\n\n{text[:PLANNER_EXCERPT_CHARS]}"),
        ])
        plan_text = be.safe_content(plan_response)
        questions = _parse_questions(plan_text)
        if not questions:
            yield _event("error", message="Falha ao gerar plano de pesquisa. O modelo não retornou perguntas no formato esperado.")
            return
        logger.info("deep-research: %d questions planned", len(questions))
        yield _event("plan", questions=questions)

        # ── 4. Research loop ──
        qa_pairs = []
        for idx, question in enumerate(questions):
            yield _event("question_start", index=idx, total=len(questions), question=question)
            try:
                relevant = vector_store.similarity_search(question, k=top_k)
                chunks_text = "\n\n".join(
                    f"[Trecho {d.metadata.get('chunk_id', '?')}]\n{d.page_content.strip()}"
                    for d in relevant
                )
                research_resp = llm.invoke([
                    SystemMessage(content=PROMPT_RESEARCH.format(question=question, chunks=chunks_text)),
                    HumanMessage(content="Analise e responda."),
                ])
                answer = be.safe_content(research_resp)
                qa_pairs.append({"question": question, "answer": answer})
                yield _event("question_done", index=idx, answer=answer)
            except Exception as exc:
                logger.exception("deep-research: failed on question %d", idx)
                qa_pairs.append({"question": question, "answer": f"Erro ao processar: {exc}"})
                yield _event("question_error", index=idx, message=str(exc))

        # ── 5. Synthesize final dossier ──
        yield _event("phase", phase="synthesizing", message="Consolidando dossiê final...")
        qa_text = "\n\n".join(
            f"### {qa['question']}\n{qa['answer']}" for qa in qa_pairs
        )
        dossier_resp = llm.invoke([
            SystemMessage(content=PROMPT_DOSSIER.format(qa_pairs=qa_text)),
            HumanMessage(content="Gere o dossiê consolidado."),
        ])
        dossier = be.safe_content(dossier_resp)

        yield _event("done", dossier=dossier, qa_pairs=qa_pairs, chunks_indexed=len(chunks))

    except Exception as exc:
        logger.exception("deep-research: fatal error")
        yield _event("error", message=str(exc))
