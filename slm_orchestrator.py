"""
SLM Orchestrator — Pipeline de 5 etapas para processamento de peças jurídicas.

Pipeline: Router → Extrator → Jurista → Redator → Auditor
Usa 3 modelos base com LoRA adapter swapping (futuro) para caber em 36GB M3 Max.

Memória: ~13GB para os 3 modelos + ~2-4GB KV cache = ~17GB total.
"""

import json
import time
import traceback
from typing import Optional

import slm_engine
from prompts_slm import (
    SLM_ROUTER_PROMPT,
    SLM_EXTRATOR_PROMPT,
    SLM_JURISTA_PROMPT,
    SLM_REDATOR_PROMPT,
    SLM_AUDITOR_PROMPT,
)

# ── RAG: Jurisprudência ──────────────────────────────────────────────────────
try:
    import jurisprudence_search
    HAS_JURIS = jurisprudence_search.is_available()
except ImportError:
    HAS_JURIS = False
    jurisprudence_search = None

# ── Style Dossier (auto-loaded from magistrate's templates) ──────────────────
import os
import random

_STYLE_DOSSIER = None
_GOLDEN_SAMPLES = {}  # tipo -> [texto, texto, ...]
_STYLE_DIR = os.path.join(os.path.dirname(__file__), "slm_training", "data")


def _load_style_dossier():
    """Carrega o dossiê de estilo e golden samples dos modelos de decisão."""
    global _STYLE_DOSSIER, _GOLDEN_SAMPLES

    dossier_path = os.path.join(_STYLE_DIR, "style_dossier.json")
    if os.path.exists(dossier_path):
        with open(dossier_path, "r", encoding="utf-8") as f:
            _STYLE_DOSSIER = json.load(f)
        print(f"📋 Style dossier carregado ({_STYLE_DOSSIER.get('total_modelos', 0)} modelos)")

    # Carregar golden samples do dataset do redator
    redator_train = os.path.join(_STYLE_DIR, "redator", "train.jsonl")
    if os.path.exists(redator_train):
        with open(redator_train, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                msg_user = entry["messages"][1]["content"]
                msg_assistant = entry["messages"][2]["content"]

                # Classificar por tipo baseado no prompt do user
                for tipo in ["sentenca", "saneamento", "tutela", "homologacao", "embargos"]:
                    if tipo in msg_user.lower():
                        if tipo not in _GOLDEN_SAMPLES:
                            _GOLDEN_SAMPLES[tipo] = []
                        _GOLDEN_SAMPLES[tipo].append(msg_assistant)
                        break
                else:
                    if "geral" not in _GOLDEN_SAMPLES:
                        _GOLDEN_SAMPLES["geral"] = []
                    _GOLDEN_SAMPLES["geral"].append(msg_assistant)

        total_samples = sum(len(v) for v in _GOLDEN_SAMPLES.values())
        print(f"📝 Golden samples carregados: {total_samples} minutas por tipo: {list(_GOLDEN_SAMPLES.keys())}")


def _get_style_prompt(tipo: str = "sentenca") -> str:
    """Gera um prompt de estilo baseado no dossiê + golden sample."""
    parts = []

    if _STYLE_DOSSIER:
        conectivos = _STYLE_DOSSIER.get("conectivos_frequentes", {})
        top_3 = list(conectivos.keys())[:3]
        parts.append(f"ESTILO DO MAGISTRADO (4ª Vara Cível BH):")
        parts.append(f"- Conectivos preferidos: {', '.join(top_3)}")
        parts.append(f"- Tamanho médio de decisões: ~{_STYLE_DOSSIER.get('char_medio', 9000)} caracteres")
        aberturas = _STYLE_DOSSIER.get("aberturas_comuns", [])
        if aberturas:
            parts.append(f"- Abertura típica: \"{aberturas[0]}...\"")

    # Golden sample do mesmo tipo
    samples = _GOLDEN_SAMPLES.get(tipo, _GOLDEN_SAMPLES.get("geral", []))
    if samples:
        sample = random.choice(samples)
        # Pegar só os primeiros 1500 chars como exemplo
        parts.append(f"\nEXEMPLO DE MINUTA DO MAGISTRADO (use como referência de estilo):")
        parts.append(sample[:1500])

    return "\n".join(parts) if parts else ""


# Carregar na importação
_load_style_dossier()


def _safe_json_parse(text: str) -> dict:
    """Tenta extrair JSON de uma resposta do modelo (pode conter texto extra)."""
    # Tenta parse direto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Procura JSON no meio do texto
    import re
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Se não conseguiu parsear, retorna o texto bruto como fallback
    return {"raw_response": text, "_parse_error": True}


class SLMOrchestrator:
    """
    Pipeline sequencial de 5 SLMs para processamento jurídico.

    Etapa 1: Router (Qwen 2.5 1.5B) — classifica tipo processual
    Etapa 2: Extrator (Gemma 3 4B) — extrai fatos estruturados
    Etapa 3: Jurista (Mistral-Nemo 12B) — análise de mérito
    Etapa 4: Redator (Mistral-Nemo 12B) — redige minuta
    Etapa 5: Auditor (Gemma 3 4B) — verifica qualidade
    """

    def __init__(self, load_all: bool = False):
        """
        Args:
            load_all: Se True, carrega todos os modelos na inicialização.
                      Se False (padrão), usa lazy loading por etapa.
        """
        self._initialized = False
        if load_all:
            self._preload_models()

    def _preload_models(self):
        """Pré-carrega os 3 modelos base (~13GB total)."""
        print("🔄 Pré-carregando 3 modelos base para pipeline SLM...")
        start = time.time()
        slm_engine.ensure_loaded("router")
        slm_engine.ensure_loaded("small")
        slm_engine.ensure_loaded("main")
        elapsed = time.time() - start
        print(f"✅ 3 modelos carregados em {elapsed:.1f}s")
        self._initialized = True

    def _step_router(self, processo_text: str) -> dict:
        """Etapa 1: Classifica o tipo processual."""
        # Usa apenas os primeiros 2000 chars para classificação rápida
        truncated = processo_text[:2000]

        response = slm_engine.generate(
            prompt=truncated,
            system_prompt=SLM_ROUTER_PROMPT,
            model_key="router",
            max_tokens=200,
        )

        return _safe_json_parse(response)

    def _step_extrator(self, processo_text: str) -> dict:
        """Etapa 2: Extrai fatos estruturados do processo."""
        response = slm_engine.generate(
            prompt=processo_text,
            system_prompt=SLM_EXTRATOR_PROMPT,
            model_key="small",
            max_tokens=1024,
        )

        return _safe_json_parse(response)

    def _step_jurista(self, fatos: dict, rota: dict, jurisprudencia_ctx: str = "") -> str:
        """Etapa 3: Análise de mérito jurídica."""
        # Monta o prompt com os insumos
        prompt = SLM_JURISTA_PROMPT.format(
            fatos_json=json.dumps(fatos, ensure_ascii=False, indent=2),
            tipo_processo=rota.get("tipo", "sentenca"),
            jurisprudencia=jurisprudencia_ctx or "Nenhuma jurisprudência disponível.",
        )

        response = slm_engine.generate(
            prompt="Analise os fatos e produza o raciocínio jurídico conforme as instruções.",
            system_prompt=prompt,
            model_key="main",
            max_tokens=4096,
        )

        return response

    def _step_redator(self, analise_juridica: str, estilo: str = "") -> str:
        """Etapa 4: Redigir minuta final."""
        prompt = SLM_REDATOR_PROMPT.format(
            analise_juridica=analise_juridica,
            estilo=estilo or "Sem modelo de estilo disponível. Use linguagem jurídica padrão.",
        )

        response = slm_engine.generate(
            prompt="Redija a sentença/decisão conforme o raciocínio do Juiz.",
            system_prompt=prompt,
            model_key="main",
            max_tokens=4096,
        )

        return response

    def _step_auditor(self, minuta: str, fatos: dict, processo_text: str) -> dict:
        """Etapa 5: Auditoria tripla (fática + eficiência + jurídica)."""
        prompt = SLM_AUDITOR_PROMPT.format(
            fatos_originais=json.dumps(fatos, ensure_ascii=False, indent=2)
                if isinstance(fatos, dict)
                else str(fatos),
            minuta=minuta,
            pedidos=json.dumps(fatos.get("pedidos", []), ensure_ascii=False)
                if isinstance(fatos, dict)
                else "N/A",
        )

        response = slm_engine.generate(
            prompt="Audite a minuta conforme o checklist.",
            system_prompt=prompt,
            model_key="small",
            max_tokens=1024,
        )

        return _safe_json_parse(response)

    def _get_jurisprudencia_context(self, causa_de_pedir: str) -> str:
        """Busca jurisprudência relevante via RAG (se disponível)."""
        if not HAS_JURIS or not causa_de_pedir:
            return ""

        try:
            results = jurisprudence_search.search(
                query=causa_de_pedir,
                page_size=3,
            )
            if results.get("results"):
                contexto_parts = []
                for r in results["results"][:3]:
                    ementa = r.get("ementa", "")[:500]
                    contexto_parts.append(
                        f"- **{r.get('numero_processo', 'N/A')}** ({r.get('ano', 'N/A')}): {ementa}"
                    )
                return "\n".join(contexto_parts)
        except Exception as e:
            print(f"⚠️ Erro ao buscar jurisprudência: {e}")

        return ""

    def run_pipeline(
        self,
        processo_text: str,
        estilo: str = "",
        max_audit_rounds: int = 1,
        callback=None,
    ) -> dict:
        """
        Executa o pipeline completo de 5 etapas.

        Args:
            processo_text: Texto completo do processo
            estilo: Dossiê de estilo do magistrado (opcional)
            max_audit_rounds: Máximo de ciclos correção-reauditoria
            callback: Função callback(etapa, msg) para progress updates

        Returns:
            dict com: minuta, audit, rota, fatos, analise, timing
        """
        timing = {}
        total_start = time.time()

        def _notify(etapa: str, msg: str):
            print(f"  📌 [{etapa}] {msg}")
            if callback:
                callback(etapa, msg)

        # ── ETAPA 1: Roteamento ──────────────────────────────────────────
        _notify("Router", "Classificando tipo processual...")
        t0 = time.time()
        rota = self._step_router(processo_text)
        timing["router"] = round(time.time() - t0, 1)
        _notify("Router", f"Tipo: {rota.get('tipo', '?')} | Matéria: {rota.get('materia', '?')}")

        # ── ETAPA 2: Extração de Fatos ───────────────────────────────────
        _notify("Extrator", "Extraindo fatos estruturados...")
        t0 = time.time()
        fatos = self._step_extrator(processo_text)
        timing["extrator"] = round(time.time() - t0, 1)
        _notify("Extrator", f"Autor: {fatos.get('autor', '?')} | Réu: {fatos.get('reu', '?')}")

        # ── ETAPA 3: Análise Jurídica (com RAG) ─────────────────────────
        _notify("Jurista", "Analisando mérito jurídico...")
        t0 = time.time()

        # Buscar jurisprudência relevante
        causa = fatos.get("causa_de_pedir", "") if isinstance(fatos, dict) else ""
        juris_ctx = self._get_jurisprudencia_context(causa)
        if juris_ctx:
            _notify("Jurista", f"RAG: {len(juris_ctx)} chars de jurisprudência encontrados")

        analise = self._step_jurista(fatos, rota, juris_ctx)
        timing["jurista"] = round(time.time() - t0, 1)

        # ── ETAPA 4: Redação da Minuta ───────────────────────────────────
        _notify("Redator", "Redigindo minuta...")
        t0 = time.time()

        # Auto-inject style from magistrate's templates if not provided
        tipo_rota = rota.get("tipo", "sentenca") if isinstance(rota, dict) else "sentenca"
        effective_estilo = estilo or _get_style_prompt(tipo_rota)
        if effective_estilo and not estilo:
            _notify("Redator", f"Estilo do magistrado carregado ({len(effective_estilo)} chars)")

        minuta = self._step_redator(analise, effective_estilo)
        timing["redator"] = round(time.time() - t0, 1)

        # ── ETAPA 5: Auditoria ───────────────────────────────────────────
        _notify("Auditor", "Auditando minuta...")
        t0 = time.time()
        audit = self._step_auditor(minuta, fatos, processo_text)
        timing["auditor"] = round(time.time() - t0, 1)

        # ── Ciclo de correção (se reprovado) ─────────────────────────────
        audit_round = 0
        while not audit.get("aprovado", True) and audit_round < max_audit_rounds:
            audit_round += 1
            _notify("Redator", f"Corrigindo minuta (rodada {audit_round})...")
            erros = audit.get("resumo", "") or json.dumps(audit.get("fatico", {}).get("erros", []))

            # Resubmete ao redator com os erros
            correction_prompt = f"""A minuta foi REPROVADA pela auditoria. Corrija os seguintes problemas:
{erros}

Minuta original:
{minuta}

Reescreva a minuta corrigida:"""

            minuta = slm_engine.generate(
                prompt=correction_prompt,
                system_prompt=SLM_REDATOR_PROMPT.format(
                    analise_juridica=analise,
                    estilo=estilo or "Linguagem jurídica padrão.",
                ),
                model_key="main",
                max_tokens=4096,
            )

            # Re-auditar
            _notify("Auditor", f"Re-auditando (rodada {audit_round})...")
            audit = self._step_auditor(minuta, fatos, processo_text)

        timing["total"] = round(time.time() - total_start, 1)

        result = {
            "minuta": minuta,
            "audit": audit,
            "rota": rota,
            "fatos": fatos,
            "analise": analise,
            "jurisprudencia_contexto": juris_ctx,
            "timing": timing,
            "model_info": {
                "router": slm_engine.MODELS["router"]["name"],
                "extrator": slm_engine.MODELS["small"]["name"],
                "jurista": slm_engine.MODELS["main"]["name"],
                "redator": slm_engine.MODELS["main"]["name"],
                "auditor": slm_engine.MODELS["small"]["name"],
            },
        }

        print(f"\n🏁 Pipeline completo em {timing['total']}s")
        print(f"   Router: {timing.get('router', '?')}s | Extrator: {timing.get('extrator', '?')}s | "
              f"Jurista: {timing.get('jurista', '?')}s | Redator: {timing.get('redator', '?')}s | "
              f"Auditor: {timing.get('auditor', '?')}s")

        return result
