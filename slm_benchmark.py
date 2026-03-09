"""
SLM Benchmark — Comparação automatizada: Pipeline SLM Local vs GPT-5.2.

Uso:
    python slm_benchmark.py --input data/test_cases/ --output benchmark_results.md

Mede: latência, qualidade da extração, completude da minuta.
"""

import json
import os
import time
import argparse
from pathlib import Path


def benchmark_slm_pipeline(processo_text: str) -> dict:
    """Executa o pipeline SLM local e mede métricas."""
    from slm_orchestrator import SLMOrchestrator

    orch = SLMOrchestrator()
    start = time.time()
    result = orch.run_pipeline(processo_text)
    elapsed = time.time() - start

    return {
        "engine": "SLM Pipeline (Local)",
        "latency_total": round(elapsed, 1),
        "timing": result.get("timing", {}),
        "minuta_length": len(result.get("minuta", "")),
        "audit_score": result.get("audit", {}).get("score", "N/A"),
        "audit_approved": result.get("audit", {}).get("aprovado", None),
        "fatos_extracted": len(result.get("fatos", {})),
        "rota": result.get("rota", {}),
        "minuta_preview": result.get("minuta", "")[:300],
        "model_info": result.get("model_info", {}),
    }


def benchmark_gpt52(processo_text: str) -> dict:
    """Executa GPT-5.2 (chat simples) e mede métricas."""
    try:
        import backend as be
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = be.get_llm(model_name="gpt-5.2-chat", temperature=0.3)
        messages = [
            SystemMessage(content=(
                "Você é um Magistrado de alto nível. Leia o processo abaixo e produza:\n"
                "1. Resumo dos fatos e pedidos\n"
                "2. Fundamentação jurídica\n"
                "3. Dispositivo (decisão final)\n"
                "Formate como uma minuta de decisão/sentença completa."
            )),
            HumanMessage(content=f"AUTOS DO PROCESSO:\n{processo_text}"),
        ]

        start = time.time()
        response = llm.invoke(messages)
        elapsed = time.time() - start

        content = be.safe_content(response)
        return {
            "engine": "GPT-5.2 (Azure API)",
            "latency_total": round(elapsed, 1),
            "minuta_length": len(content),
            "minuta_preview": content[:300],
        }
    except Exception as e:
        return {
            "engine": "GPT-5.2 (Azure API)",
            "error": str(e),
            "latency_total": 0,
        }


def generate_report(results: list, output_path: str):
    """Gera relatório markdown de benchmark."""
    report = ["# 📊 Benchmark: SLM Pipeline Local vs GPT-5.2\n"]
    report.append(f"Data: {time.strftime('%Y-%m-%d %H:%M')}\n")
    report.append(f"Processos testados: {len(results)}\n\n")

    # Tabela resumo
    report.append("## Resultados\n")
    report.append("| # | Engine | Latência | Tam. Minuta | Audit Score |\n")
    report.append("|---|--------|----------|-------------|-------------|\n")

    for i, r in enumerate(results):
        for engine_result in [r.get("slm", {}), r.get("gpt52", {})]:
            if not engine_result:
                continue
            report.append(
                f"| {i+1} | {engine_result.get('engine', '?')} | "
                f"{engine_result.get('latency_total', '?')}s | "
                f"{engine_result.get('minuta_length', '?')} chars | "
                f"{engine_result.get('audit_score', 'N/A')} |\n"
            )

    # Médias
    slm_latencies = [r["slm"]["latency_total"] for r in results if "slm" in r and "latency_total" in r["slm"]]
    gpt_latencies = [r["gpt52"]["latency_total"] for r in results if "gpt52" in r and "latency_total" in r["gpt52"] and not r["gpt52"].get("error")]

    if slm_latencies:
        report.append(f"\n**Latência média SLM Local:** {sum(slm_latencies)/len(slm_latencies):.1f}s\n")
    if gpt_latencies:
        report.append(f"**Latência média GPT-5.2:** {sum(gpt_latencies)/len(gpt_latencies):.1f}s\n")

    report.append("\n## Detalhes por Processo\n")
    for i, r in enumerate(results):
        report.append(f"\n### Processo {i+1}: {r.get('filename', 'N/A')}\n")
        if "slm" in r:
            slm = r["slm"]
            report.append(f"**SLM Local:** {slm.get('latency_total', '?')}s | "
                         f"Score: {slm.get('audit_score', 'N/A')} | "
                         f"Rota: {slm.get('rota', {}).get('tipo', '?')}\n")
            if slm.get("timing"):
                t = slm["timing"]
                report.append(f"  ⏱ Router: {t.get('router', '?')}s | Extrator: {t.get('extrator', '?')}s | "
                             f"Jurista: {t.get('jurista', '?')}s | Redator: {t.get('redator', '?')}s | "
                             f"Auditor: {t.get('auditor', '?')}s\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(report))
    print(f"\n📊 Relatório salvo em {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SLM Pipeline vs GPT-5.2")
    parser.add_argument("--input", required=True, help="Pasta com processos de teste (TXT/JSON)")
    parser.add_argument("--output", default="benchmark_results.md", help="Arquivo de saída")
    parser.add_argument("--skip-gpt", action="store_true", help="Pular teste GPT-5.2 (só testar SLM)")
    args = parser.parse_args()

    test_files = sorted(Path(args.input).glob("*.txt")) + sorted(Path(args.input).glob("*.json"))
    if not test_files:
        print(f"❌ Nenhum arquivo de teste em {args.input}")
        return

    print(f"📂 {len(test_files)} processos de teste encontrados\n")

    results = []
    for f in test_files:
        print(f"\n{'='*60}")
        print(f"🔍 Processo: {f.name}")

        if f.suffix == ".json":
            with open(f) as fh:
                data = json.load(fh)
                processo_text = data.get("processo", data.get("texto", json.dumps(data)))
        else:
            with open(f) as fh:
                processo_text = fh.read()

        result = {"filename": f.name}

        # SLM Pipeline
        print("  🧠 Executando SLM Pipeline...")
        result["slm"] = benchmark_slm_pipeline(processo_text)
        print(f"  ✅ SLM: {result['slm']['latency_total']}s")

        # GPT-5.2
        if not args.skip_gpt:
            print("  ☁️ Executando GPT-5.2...")
            result["gpt52"] = benchmark_gpt52(processo_text)
            if result["gpt52"].get("error"):
                print(f"  ⚠️ GPT-5.2: {result['gpt52']['error'][:100]}")
            else:
                print(f"  ✅ GPT-5.2: {result['gpt52']['latency_total']}s")

        results.append(result)

    generate_report(results, args.output)


if __name__ == "__main__":
    main()
