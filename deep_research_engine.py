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

DEFAULT_MODEL = "gpt-5.3-chat"  # reasoning model — auto-activates reasoning_effort=high
TOP_K = 8                       # chunks retrieved per question (raised for richer evidence)
MAX_QUESTIONS = 12
PLANNER_EXCERPT_CHARS = 12000   # how much of the text we show the planner
MAX_OUTPUT_TOKENS_RESEARCH = 4000   # per-question answer budget
MAX_OUTPUT_TOKENS_DOSSIER = 16000   # final dossier budget (large prose synthesis)


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


PROMPT_RESEARCH = """Você é um analista jurídico sênior investigando profundamente um processo \
judicial. A pergunta abaixo exige análise minuciosa, exaustiva e rigorosa, baseada \
**estritamente** nos trechos do processo fornecidos.

PERGUNTA SOB INVESTIGAÇÃO:
{question}

TRECHOS RECUPERADOS DO PROCESSO:
{chunks}

INSTRUÇÕES DE EXECUÇÃO (cumpra todas):

1. **Profundidade**: produza uma resposta extensa e detalhada — múltiplos parágrafos, \
nunca menos de 4. Quando houver várias dimensões na pergunta, separe em subtópicos com \
títulos `###`.

2. **Raciocínio explícito**: antes de concluir, explicite o raciocínio jurídico passo a passo. \
Demonstre como cada elemento dos trechos sustenta sua conclusão.

3. **Citações rastreáveis**: toda afirmação factual deve vir acompanhada de citação \
`[Trecho N]`. Se uma mesma informação aparece em múltiplos trechos, cite todos.

4. **Análise crítica**: não se limite a descrever — analise. Aponte:
   - Contradições internas nos autos, se houver
   - Lacunas probatórias relevantes
   - Implicações para a decisão final
   - Riscos processuais identificados

5. **Fundamentação legal**: quando aplicável, identifique os dispositivos legais (CPC, \
CC, leis especiais, súmulas, temas vinculantes) que se conectam aos fatos descritos.

6. **Rigor**: se a informação NÃO está nos trechos, escreva explicitamente \
"Não foi possível localizar nos trechos analisados" — NUNCA invente nem extrapole.

7. **Linguagem**: técnica jurídica, precisa, sem floreios desnecessários, mas sem \
cortes que sacrifiquem a completude."""


PROMPT_DOSSIER = """Você é um magistrado experiente redigindo um **relatório de análise \
processual completo e minucioso** sobre um processo judicial. Você recebeu o resultado \
de uma investigação aprofundada (perguntas e respostas detalhadas) e precisa consolidar \
tudo em um relatório técnico extenso que servirá de base para sua decisão.

⚠️ ESTE NÃO É UM RESUMO. É um relatório completo, denso, com todas as nuances. \
Aproveite TODO o conteúdo das respostas — não suprima informações relevantes.

ESTRUTURA OBRIGATÓRIA DO RELATÓRIO (markdown):

# Relatório de Análise Processual

## I. Identificação do Processo
Partes (qualificação completa, se disponível), classe processual, valor da causa, \
fase atual, juízo competente. Não apenas listar — contextualize.

## II. Histórico Processual Detalhado
Narrativa cronológica das fases do processo: distribuição, citação, contestação, \
réplica, audiências, perícias, decisões interlocutórias relevantes. Inclua datas \
quando disponíveis e o conteúdo essencial de cada ato.

## III. Síntese dos Fatos
Narrativa detalhada dos fatos da causa, com todas as versões (autor e réu) e \
contextualização suficiente para compreender a controvérsia. Mínimo 3-5 parágrafos.

## IV. Pedidos e Causa de Pedir
Análise individualizada de cada pedido formulado (principal e sucessivos), \
identificando a causa de pedir próxima e remota, a natureza da tutela pretendida \
(declaratória, condenatória, constitutiva, mandamental, executiva), e os valores \
ou obrigações específicas pleiteados.

## V. Defesa Apresentada
### V.1. Preliminares
Análise individualizada de cada preliminar suscitada, com avaliação de procedência.

### V.2. Mérito
Teses defensivas, fatos impeditivos/modificativos/extintivos invocados, fundamentos \
jurídicos da defesa.

## VI. Provas Produzidas
Inventário detalhado: documentos (com identificação), testemunhal (depoimentos \
relevantes), pericial (conclusões do laudo), depoimentos pessoais. Para cada conjunto \
probatório, indique o que demonstra ou não demonstra.

## VII. Questões Controvertidas
Enumeração precisa dos pontos ainda pendentes de decisão, com análise sobre quais \
exigem prova e quais comportam julgamento imediato.

## VIII. Preliminares e Matérias de Ordem Pública
Análise (mesmo que para descartar) de:
- Prescrição/decadência
- Legitimidade ativa e passiva
- Interesse processual
- Competência
- Litispendência/coisa julgada
- Assistência judiciária gratuita
- Citação válida / revelia
- Outras nulidades processuais

## IX. Jurisprudência e Precedentes Aplicáveis
Súmulas, temas de recursos repetitivos (STJ), temas de repercussão geral (STF), \
enunciados, e jurisprudência citada nos autos ou claramente aplicável ao caso. \
Analise como cada precedente se ajusta (ou se distingue) dos fatos.

## X. Análise Crítica e Pontos Críticos para a Decisão
Esta é a seção mais importante. Discorra extensamente sobre:
- Pontos fortes e fracos da pretensão autoral
- Pontos fortes e fracos da defesa
- Lacunas probatórias e seu impacto no julgamento
- Contradições nos autos
- Riscos de embargos de declaração / recurso
- Recomendações específicas para a redação da decisão
- Questões de prova dinâmica do ônus, se aplicável

## XI. Sugestão de Encaminhamento
Indicação fundamentada do próximo ato adequado: saneador, sentença, despacho de mero \
expediente, designação de audiência, conversão em diligência, etc. — com justificativa.

---

PERGUNTAS E RESPOSTAS DA INVESTIGAÇÃO:
{qa_pairs}

REGRAS ABSOLUTAS:
- ⚠️ EXTENSÃO: o relatório deve ser **completo e minucioso**. Não economize palavras \
em seções com conteúdo substantivo. Mínimo desejado: 2.500 palavras (mais é melhor \
quando há matéria nos autos).
- ⚠️ Preserve TODAS as citações `[Trecho N]` das respostas originais — rastreabilidade \
é obrigatória.
- ⚠️ Para informações não localizadas, escreva "Não localizado nos autos analisados." \
e prossiga (não pule a seção).
- ⚠️ NÃO seja repetitivo: integre as respostas das perguntas individuais; não copie-as \
em bloco.
- ⚠️ Linguagem técnica, formal, precisa — adequada a uma decisão judicial.
- ⚠️ Quando apropriado, use listas, subseções (`###`) e ênfase (`**negrito**`) para \
facilitar a leitura."""


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
        # Planner: low-ish temperature, no special budget needed (output is short JSON).
        planner_llm = be.get_llm(model_name=model_name, temperature=0.3, api_key=api_key)
        # Research LLM: high output budget so each answer can be exhaustive.
        # On GPT-5.3 this also auto-enables reasoning_effort=high (see backend.get_llm).
        research_llm = be.get_llm(
            model_name=model_name,
            temperature=0.5,
            api_key=api_key,
            max_tokens=MAX_OUTPUT_TOKENS_RESEARCH,
        )
        # Dossier LLM: largest output budget — final synthesis is long-form prose.
        dossier_llm = be.get_llm(
            model_name=model_name,
            temperature=0.5,
            api_key=api_key,
            max_tokens=MAX_OUTPUT_TOKENS_DOSSIER,
        )
        plan_response = planner_llm.invoke([
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
                research_resp = research_llm.invoke([
                    SystemMessage(content=PROMPT_RESEARCH.format(question=question, chunks=chunks_text)),
                    HumanMessage(content="Realize a análise minuciosa solicitada."),
                ])
                answer = be.safe_content(research_resp)
                qa_pairs.append({"question": question, "answer": answer})
                yield _event("question_done", index=idx, answer=answer)
            except Exception as exc:
                logger.exception("deep-research: failed on question %d", idx)
                qa_pairs.append({"question": question, "answer": f"Erro ao processar: {exc}"})
                yield _event("question_error", index=idx, message=str(exc))

        # ── 5. Synthesize final dossier ──
        yield _event("phase", phase="synthesizing", message="Consolidando relatório completo (high-effort)...")
        qa_text = "\n\n".join(
            f"### {qa['question']}\n{qa['answer']}" for qa in qa_pairs
        )
        dossier_resp = dossier_llm.invoke([
            SystemMessage(content=PROMPT_DOSSIER.format(qa_pairs=qa_text)),
            HumanMessage(content="Redija o relatório completo e minucioso conforme especificado."),
        ])
        dossier = be.safe_content(dossier_resp)

        yield _event("done", dossier=dossier, qa_pairs=qa_pairs, chunks_indexed=len(chunks))

    except Exception as exc:
        logger.exception("deep-research: fatal error")
        yield _event("error", message=str(exc))
