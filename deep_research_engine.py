"""Deep Research over a single uploaded process — high-effort, long-form mode.

Pipeline (each step yields SSE events to the client):
    1. chunk the uploaded text (HybridSemanticChunker)
    2. embed into an ephemeral in-memory Chroma collection
    3. plan a broad set of investigation questions (18-22)
    4. research loop: each question -> top-K similarity + LLM synthesis with citations
    5. synthesize the final report section-by-section (12 sections, each in its own
       LLM call with all Q&A as context) — this is what lets the report grow to
       20+ pages: we are NOT bound by a single call's output cap.
    6. compose an executive summary from the assembled report

Designed for large/complex processes (hundreds of pages). Total runtime can reach
10-15 minutes; the SSE stream emits granular progress to keep the UI live.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Iterator, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

import backend as be

logger = logging.getLogger(__name__)

# ── Tuning knobs ───────────────────────────────────────────────────────────
DEFAULT_MODEL = "gpt-5.3-chat"           # auto-activates reasoning_effort=high in backend.get_llm
TOP_K = 12                                # chunks retrieved per investigation question
MAX_QUESTIONS = 22
PLANNER_EXCERPT_CHARS = 18000             # how much of the text we show the planner
MAX_OUTPUT_TOKENS_RESEARCH = 8000         # per-question answer budget
MAX_OUTPUT_TOKENS_SECTION = 8000          # per-section synthesis budget
MAX_OUTPUT_TOKENS_SUMMARY = 4000          # executive summary budget
# A trim limit so we don't blow context when feeding all Q&A into each section call.
# 22 questions × 8k tokens ≈ 175k tokens is too much; trim each answer to ~3k chars
# (~750 tokens) when used as section context. The full text is still preserved in
# the qa_pairs payload returned to the UI.
QA_TRIM_FOR_SYNTHESIS = 3000


# ── Report skeleton (12 sections, generated independently) ────────────────
REPORT_SECTIONS: list[dict] = [
    {
        "id": "identificacao",
        "title": "I. Identificação do Processo",
        "guidance": (
            "Faça uma análise contextualizada e completa: partes (qualificação detalhada quando disponível), "
            "classe processual, valor da causa e seu significado para a controvérsia, juízo competente, "
            "fase atual, número do processo se disponível. Não apenas liste — contextualize cada elemento."
        ),
    },
    {
        "id": "historico",
        "title": "II. Histórico Processual Detalhado",
        "guidance": (
            "Narrativa cronológica exaustiva de todos os atos processuais relevantes: distribuição, "
            "despachos iniciais, citação, contestação, réplica, decisões interlocutórias, audiências realizadas, "
            "perícias designadas/produzidas, recursos interpostos, conversões em diligência. Inclua datas e "
            "o conteúdo essencial de cada ato. Esta seção deve permitir ao leitor reconstituir todo o trâmite."
        ),
    },
    {
        "id": "fatos",
        "title": "III. Síntese dos Fatos",
        "guidance": (
            "Narrativa detalhada dos fatos da causa em ordem cronológica, contemplando: a versão do autor, "
            "a versão do réu, fatos incontroversos vs. controvertidos, contexto fático relevante (relação "
            "jurídica subjacente, antecedentes, circunstâncias). Mínimo 4-6 parágrafos densos."
        ),
    },
    {
        "id": "pedidos",
        "title": "IV. Pedidos e Causa de Pedir",
        "guidance": (
            "Análise individualizada de cada pedido (principal, sucessivos, cumulativos, alternativos): "
            "(a) identificação precisa do pedido; (b) natureza da tutela pretendida (declaratória, "
            "constitutiva, condenatória, mandamental, executiva); (c) causa de pedir próxima (fatos) e "
            "remota (fundamentos jurídicos); (d) valores ou obrigações específicas pleiteados; "
            "(e) eventual tutela de urgência ou evidência postulada."
        ),
    },
    {
        "id": "defesa",
        "title": "V. Defesa Apresentada",
        "guidance": (
            "Análise individualizada da contestação e demais peças defensivas, dividida em:\n"
            "### V.1. Preliminares\nCada preliminar suscitada com avaliação técnica de procedência.\n"
            "### V.2. Mérito\nTeses defensivas, fatos impeditivos/modificativos/extintivos invocados, "
            "fundamentos jurídicos da defesa, eventual reconvenção ou pedido contraposto, alegações sobre prescrição/decadência."
        ),
    },
    {
        "id": "provas",
        "title": "VI. Provas Produzidas e Análise Probatória",
        "guidance": (
            "Inventário detalhado de todo o material probatório constante dos autos:\n"
            "- **Documental**: identificar documentos juntados por cada parte, fls./IDs quando possível, "
            "e o fato que cada documento demonstra ou não demonstra.\n"
            "- **Testemunhal**: depoimentos relevantes, pontos centrais de cada testemunha.\n"
            "- **Pericial**: existência, conclusões do laudo, eventuais quesitos não respondidos, impugnações.\n"
            "- **Depoimentos pessoais e confissões** se houver.\n"
            "Em seguida, faça **análise probatória crítica**: o ônus está sendo cumprido? Há provas suficientes para cada fato controvertido?"
        ),
    },
    {
        "id": "controvertidas",
        "title": "VII. Questões Controvertidas",
        "guidance": (
            "Enumeração precisa dos pontos controvertidos ainda pendentes de decisão. Para cada um: "
            "(a) descrição da controvérsia; (b) elementos probatórios já existentes; (c) elementos "
            "ainda necessários, se houver; (d) se a questão é de fato, de direito ou mista; "
            "(e) se comporta julgamento imediato ou exige instrução."
        ),
    },
    {
        "id": "preliminares",
        "title": "VIII. Preliminares e Matérias de Ordem Pública",
        "guidance": (
            "Análise individualizada (mesmo que seja para descartar) de:\n"
            "- Prescrição e decadência\n"
            "- Legitimidade ativa e passiva (incluindo casos de litisconsórcio necessário)\n"
            "- Interesse processual (binômio necessidade-adequação)\n"
            "- Competência (em razão da matéria, valor, pessoa, território; absoluta/relativa)\n"
            "- Litispendência, coisa julgada, conexão, continência\n"
            "- Capacidade postulatória e regularidade da representação\n"
            "- Assistência judiciária gratuita (deferimento/indeferimento, impugnações)\n"
            "- Validade da citação, eventual revelia e seus efeitos\n"
            "- Outras nulidades processuais identificadas\n"
            "Para cada item, indicar se há providência saneadora pendente."
        ),
    },
    {
        "id": "jurisprudencia",
        "title": "IX. Jurisprudência e Precedentes Aplicáveis",
        "guidance": (
            "Identifique e analise:\n"
            "- Súmulas (STF, STJ, TST, tribunais regionais) aplicáveis\n"
            "- Temas de recursos repetitivos (STJ — Temas 1, 2, ..., 1000+)\n"
            "- Temas de repercussão geral (STF)\n"
            "- IRDRs (CPC 976) eventualmente aplicáveis\n"
            "- Enunciados de jornadas, conselhos, FONAJE\n"
            "- Jurisprudência citada nas peças das partes\n"
            "Para cada precedente, analise: (a) tese fixada; (b) aplicabilidade aos fatos do caso "
            "(subsumibilidade); (c) eventual distinção (distinguishing) ou superação (overruling)."
        ),
    },
    {
        "id": "analise_critica",
        "title": "X. Análise Crítica e Pontos Críticos para a Decisão",
        "guidance": (
            "Esta é a seção mais densa do relatório. Discorra extensamente sobre:\n"
            "- **Pontos fortes da pretensão autoral**: o que mais favorece o acolhimento\n"
            "- **Pontos fracos da pretensão autoral**: vulnerabilidades, lacunas\n"
            "- **Pontos fortes da defesa**: teses mais convincentes\n"
            "- **Pontos fracos da defesa**: fragilidades, contradições\n"
            "- **Lacunas probatórias** e seu impacto provável no julgamento\n"
            "- **Contradições internas** nos autos (entre peças, entre documentos e alegações)\n"
            "- **Distribuição dinâmica do ônus da prova** (CPC 373 §1º) — cabível? justificada?\n"
            "- **Riscos de embargos de declaração** por omissão/contradição\n"
            "- **Riscos recursais** específicos do caso\n"
            "- **Pontos de atenção especial** que o magistrado deve observar ao decidir\n"
            "Mínimo 6-8 parágrafos substantivos."
        ),
    },
    {
        "id": "encaminhamento",
        "title": "XI. Sugestão de Encaminhamento",
        "guidance": (
            "Indicação fundamentada e específica do próximo ato processual adequado, escolhendo entre:\n"
            "- Saneador (com pontos a sanear)\n"
            "- Sentença (com tese principal sugerida)\n"
            "- Decisão (tutela de urgência, etc.)\n"
            "- Despacho de mero expediente\n"
            "- Designação de audiência (com objetivos específicos)\n"
            "- Conversão em diligência (com providências específicas)\n"
            "- Intimação para alegações finais\n"
            "Justifique a escolha com base na análise das seções anteriores e indique, se aplicável, "
            "o tipo de minuta a ser elaborada e seus pontos essenciais."
        ),
    },
]


# ── Prompts ────────────────────────────────────────────────────────────────

PROMPT_PLANNER = """Você é um magistrado experiente investigando um processo judicial complexo \
antes de redigir um relatório de análise aprofundada e a decisão final.

Sua tarefa: gerar **{max_questions} perguntas de investigação** que cubram exaustivamente \
todos os aspectos do processo. As perguntas devem ser específicas, granulares, e cobrir todas \
as dimensões relevantes:

- Identificação processual (partes, classe, valor, fase, juízo)
- Histórico processual completo (citação, contestação, audiências, perícias, decisões interlocutórias)
- Pedidos (principais, sucessivos, cumulativos) e causa de pedir
- Defesa: preliminares específicas e mérito (cada tese defensiva)
- Provas documentais, testemunhais e periciais
- Questões controvertidas pendentes
- Preliminares de ordem pública (prescrição, decadência, ilegitimidade, interesse, competência, AJG)
- Aplicação de jurisprudência vinculante, súmulas, temas repetitivos
- Pontos críticos para a decisão e riscos recursais
- Lacunas probatórias e contradições

Devolva APENAS um JSON no formato:
{{"questions": ["pergunta 1?", "pergunta 2?", ...]}}

Não inclua explicações fora do JSON. Gere {max_questions} perguntas precisas e independentes."""


PROMPT_RESEARCH = """Você é um analista jurídico sênior investigando profundamente um processo \
judicial. A pergunta abaixo exige análise minuciosa, exaustiva e rigorosa, baseada \
**estritamente** nos trechos do processo fornecidos.

PERGUNTA SOB INVESTIGAÇÃO:
{question}

TRECHOS RECUPERADOS DO PROCESSO:
{chunks}

INSTRUÇÕES DE EXECUÇÃO (cumpra todas):

1. **Profundidade**: produza resposta longa e densa — mínimo 5 parágrafos substantivos. \
Quando houver múltiplas dimensões na pergunta, organize com subtópicos `###`.

2. **Raciocínio explícito**: antes de concluir, explicite o raciocínio jurídico passo a passo. \
Demonstre como cada elemento dos trechos sustenta a conclusão.

3. **Citações rastreáveis**: toda afirmação factual deve vir com citação `[Trecho N]`. \
Se uma informação aparece em múltiplos trechos, cite todos os relevantes.

4. **Análise crítica**: não se limite a descrever. Aponte:
   - Contradições internas nos autos, se houver
   - Lacunas probatórias relevantes
   - Implicações para a decisão final
   - Riscos processuais identificados
   - Pontos de atenção que o magistrado deve observar

5. **Fundamentação legal**: identifique dispositivos legais (CPC, CC, CTN, CLT, leis especiais), \
súmulas e temas vinculantes (STF, STJ) que se conectam aos fatos.

6. **Rigor**: se a informação NÃO está nos trechos, escreva explicitamente \
"Não foi possível localizar nos trechos analisados" — NUNCA invente nem extrapole.

7. **Linguagem**: técnica jurídica, precisa, formal — adequada a um relatório de análise \
processual que servirá de base para uma decisão judicial.

⚠️ Esta pergunta faz parte de uma investigação maior. Sua resposta será integrada a um \
relatório de 20+ páginas. Seja completo: não economize palavras quando há matéria nos autos."""


PROMPT_SECTION = """Você é um magistrado experiente redigindo uma **seção específica** de um \
relatório de análise processual extenso e minucioso. Este relatório tem 20+ páginas e é \
composto por 11 seções independentes; você está redigindo apenas a seção abaixo.

═══════════════════════════════════════════════════════════════════════════
SEÇÃO A REDIGIR:
{section_title}

DIRETRIZ DE CONTEÚDO:
{section_guidance}
═══════════════════════════════════════════════════════════════════════════

CONTEXTO DA INVESTIGAÇÃO (perguntas e respostas analíticas sobre o processo):
{qa_pairs}

═══════════════════════════════════════════════════════════════════════════

REGRAS ABSOLUTAS:

⚠️ **Extensão**: esta seção deve ter no mínimo **6 parágrafos substantivos** (idealmente \
entre 8 e 15 parágrafos quando há matéria nos autos). Não se preocupe com tamanho — \
seja completo.

⚠️ **Cabeçalho**: comece com o título exato da seção em markdown:
`## {section_title}`

⚠️ **Citações**: preserve TODAS as citações `[Trecho N]` provenientes do contexto da \
investigação. Rastreabilidade é obrigatória.

⚠️ **Conteúdo único**: gere APENAS o conteúdo desta seção. NÃO inclua outras seções do \
relatório. NÃO inclua resumo executivo, conclusões gerais ou índice.

⚠️ **Quando não houver informação**: se o processo não fornece base para algum aspecto \
desta seção, escreva explicitamente "Não localizado nos autos analisados quanto a [X]." \
e prossiga com o que existe.

⚠️ **Subseções**: use `###` para criar subseções dentro desta seção quando isso ajudar \
a organizar análises distintas.

⚠️ **Linguagem**: técnica jurídica, formal, precisa. Adequada a uma peça oficial."""


PROMPT_EXECUTIVE_SUMMARY = """Você é um magistrado experiente redigindo o **Resumo Executivo** \
de um relatório de análise processual extenso (20+ páginas, 11 seções) que já foi totalmente \
elaborado.

═══════════════════════════════════════════════════════════════════════════
RELATÓRIO COMPLETO (use como única fonte):
{full_report}
═══════════════════════════════════════════════════════════════════════════

Sua tarefa: escrever o Resumo Executivo que abrirá o relatório. Ele deve:

1. **Comprimento**: 1-2 páginas (~5-8 parágrafos densos).
2. **Conteúdo**: destacar (a) controvérsia central; (b) principais questões a decidir; \
(c) pontos críticos identificados; (d) eventual preliminar que comprometa o mérito; \
(e) recomendação de encaminhamento. Sem listar todas as seções.
3. **Independência**: deve ser legível por si só — alguém que leia só o Resumo Executivo \
deve sair com uma compreensão essencial do processo.
4. **Formato**: começar com `## Resumo Executivo` (markdown).
5. **Sem citações `[Trecho N]`** — este é o resumo de mais alto nível, abstrai o material \
bruto.

NÃO inclua o restante do relatório. Apenas o Resumo Executivo."""


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


def _format_qa_for_synthesis(qa_pairs: list[dict]) -> str:
    """Trim each Q&A pair so the full block fits in a section synthesis call.

    The full untrimmed text remains in the qa_pairs returned to the UI.
    """
    parts: list[str] = []
    for i, qa in enumerate(qa_pairs, start=1):
        q = qa.get("question", "")
        a = qa.get("answer", "") or ""
        if len(a) > QA_TRIM_FOR_SYNTHESIS:
            a = a[:QA_TRIM_FOR_SYNTHESIS] + "\n[...resposta truncada para síntese; veja completa na UI]"
        parts.append(f"### {i}. {q}\n{a}")
    return "\n\n".join(parts)


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_deep_research(
    text: str,
    api_key: str,
    model_name: str = DEFAULT_MODEL,
    top_k: int = TOP_K,
) -> Iterator[str]:
    """Run the deep research pipeline. Yields SSE-formatted progress events.

    Events emitted:
        phase           — high-level phase ("chunking", "embedding", "planning",
                          "researching", "synthesizing", "summarizing")
        plan            — emitted once with the full question list
        question_start  — beginning of one research question
        question_done   — answer for one research question
        question_error  — non-fatal failure on a single question
        section_start   — beginning of a report section
        section_done    — content of a report section ready
        section_error   — non-fatal failure on a single section
        done            — final report ready (executive summary + sections concatenated)
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
        yield _event("phase", phase="embedding", message=f"Indexando {len(chunks)} trechos no vector store efêmero...")
        embedding_fn = be.get_embedding_function(api_key=api_key)
        collection_name = f"deep_research_{uuid.uuid4().hex[:10]}"
        vector_store = Chroma(
            embedding_function=embedding_fn,
            collection_name=collection_name,
        )
        BATCH = 32
        for i in range(0, len(chunks), BATCH):
            vector_store.add_documents(chunks[i:i + BATCH])

        # ── 3. Prepare LLMs ──
        # Planner: short JSON output, no special budget.
        planner_llm = be.get_llm(model_name=model_name, temperature=0.3, api_key=api_key)
        # Research LLM: high output for exhaustive per-question answers.
        research_llm = be.get_llm(
            model_name=model_name,
            temperature=0.5,
            api_key=api_key,
            max_tokens=MAX_OUTPUT_TOKENS_RESEARCH,
        )
        # Section LLM: each section is a substantial standalone piece.
        section_llm = be.get_llm(
            model_name=model_name,
            temperature=0.5,
            api_key=api_key,
            max_tokens=MAX_OUTPUT_TOKENS_SECTION,
        )
        # Summary LLM: tighter, ~1-2 page abstractive summary.
        summary_llm = be.get_llm(
            model_name=model_name,
            temperature=0.4,
            api_key=api_key,
            max_tokens=MAX_OUTPUT_TOKENS_SUMMARY,
        )

        # ── 4. Plan investigation questions ──
        yield _event("phase", phase="planning", message="Planejando perguntas de investigação...")
        plan_response = planner_llm.invoke([
            SystemMessage(content=PROMPT_PLANNER.format(max_questions=MAX_QUESTIONS)),
            HumanMessage(content=(
                f"Início do processo (primeiros {PLANNER_EXCERPT_CHARS} caracteres):\n\n"
                f"{text[:PLANNER_EXCERPT_CHARS]}"
            )),
        ])
        plan_text = be.safe_content(plan_response)
        questions = _parse_questions(plan_text)
        if not questions:
            yield _event("error", message="Falha ao gerar plano de pesquisa. O modelo não retornou perguntas no formato esperado.")
            return
        logger.info("deep-research: %d questions planned", len(questions))
        yield _event("plan", questions=questions)

        # ── 5. Research loop ──
        yield _event("phase", phase="researching", message=f"Investigando {len(questions)} pontos do processo...")
        qa_pairs: list[dict] = []
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

        # ── 6. Section-by-section synthesis ──
        yield _event(
            "phase",
            phase="synthesizing",
            message=f"Redigindo o relatório seção por seção ({len(REPORT_SECTIONS)} seções)...",
        )
        qa_block = _format_qa_for_synthesis(qa_pairs)
        sections: list[dict] = []
        for s_idx, section_def in enumerate(REPORT_SECTIONS):
            yield _event(
                "section_start",
                index=s_idx,
                total=len(REPORT_SECTIONS),
                title=section_def["title"],
                section_id=section_def["id"],
            )
            try:
                section_resp = section_llm.invoke([
                    SystemMessage(content=PROMPT_SECTION.format(
                        section_title=section_def["title"],
                        section_guidance=section_def["guidance"],
                        qa_pairs=qa_block,
                    )),
                    HumanMessage(content=f"Redija a seção '{section_def['title']}' agora."),
                ])
                content = be.safe_content(section_resp).strip()
                sections.append({
                    "id": section_def["id"],
                    "title": section_def["title"],
                    "content": content,
                })
                yield _event("section_done", index=s_idx, title=section_def["title"], section_id=section_def["id"], content=content)
            except Exception as exc:
                logger.exception("deep-research: failed on section %s", section_def["id"])
                err_content = f"## {section_def['title']}\n\n_Erro ao gerar esta seção: {exc}_"
                sections.append({
                    "id": section_def["id"],
                    "title": section_def["title"],
                    "content": err_content,
                })
                yield _event("section_error", index=s_idx, title=section_def["title"], section_id=section_def["id"], message=str(exc))

        # Concatenate sections into a working report (without exec summary yet)
        sections_concat = "\n\n".join(s["content"] for s in sections)

        # ── 7. Executive summary (depends on the assembled report) ──
        yield _event("phase", phase="summarizing", message="Compondo o resumo executivo a partir do relatório...")
        try:
            summary_resp = summary_llm.invoke([
                SystemMessage(content=PROMPT_EXECUTIVE_SUMMARY.format(full_report=sections_concat)),
                HumanMessage(content="Redija o Resumo Executivo agora."),
            ])
            executive_summary = be.safe_content(summary_resp).strip()
        except Exception as exc:
            logger.exception("deep-research: executive summary failed")
            executive_summary = (
                "## Resumo Executivo\n\n_Não foi possível gerar o resumo executivo "
                f"automaticamente ({exc}). O relatório completo está disponível abaixo._"
            )

        # ── 8. Assemble final report ──
        final_report = (
            f"# Relatório de Análise Processual\n\n"
            f"{executive_summary}\n\n"
            f"---\n\n"
            f"{sections_concat}"
        )

        yield _event(
            "done",
            dossier=final_report,
            executive_summary=executive_summary,
            sections=sections,
            qa_pairs=qa_pairs,
            chunks_indexed=len(chunks),
        )

    except Exception as exc:
        logger.exception("deep-research: fatal error")
        yield _event("error", message=str(exc))
