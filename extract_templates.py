"""
Extrai texto dos modelos de decisão do magistrado e gera datasets de fine-tuning.

Uso:
    python extract_templates.py

Lê os .docx de 'minutas 4ª vara 2' e:
1. Extrai texto puro de cada documento
2. Classifica por tipo (sentença, saneador, tutela, etc.)
3. Gera dataset ChatML para fine-tuning do Redator e Router
4. Gera um dossiê de estilo consolidado
"""

import json
import os
import re
from pathlib import Path

try:
    import docx
except ImportError:
    print("pip install python-docx")
    exit(1)


TEMPLATE_DIR = Path("/Users/danielabueno/Downloads/minutas 4ª vara  2")
OUTPUT_DIR = Path("/Users/danielabueno/Downloads/jurisbusca/slm_training/data")


def extract_docx_text(filepath: str) -> str:
    """Extrai texto puro de um DOCX."""
    doc = docx.Document(filepath)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def classify_template(filename: str, text: str) -> dict:
    """Classifica o tipo da minuta pelo nome do arquivo e conteúdo."""
    fname = filename.lower()

    # Classificação por nome do arquivo
    if any(x in fname for x in ["saneador", "saneamento"]):
        tipo = "saneamento"
    elif any(x in fname for x in ["tutela", "liminar"]):
        tipo = "tutela_urgencia"
    elif any(x in fname for x in ["homologação", "homologacao", "desistência", "renuncia", "transação"]):
        tipo = "homologacao"
    elif any(x in fname for x in ["embargos de declaração"]):
        tipo = "embargos_declaracao"
    elif any(x in fname for x in ["embargos à execução", "embargos monitórios"]):
        tipo = "sentenca"
    elif any(x in fname for x in ["extinção"]):
        tipo = "sentenca"
    elif any(x in fname for x in ["abandono"]):
        tipo = "despacho"
    elif "baaf" in fname:
        tipo = "sentenca"
    else:
        tipo = "sentenca"

    # Detectar matéria
    if any(x in fname for x in ["trânsito", "transito", "acidente"]):
        materia = "civel"
    elif any(x in fname for x in ["negativação", "negativacao", "serasa"]):
        materia = "consumidor"
    elif any(x in fname for x in ["emprestimo", "empréstimo", "revisional", "financiamento"]):
        materia = "consumidor"
    elif any(x in fname for x in ["companhia aérea", "voo"]):
        materia = "consumidor"
    elif any(x in fname for x in ["associação", "desconto"]):
        materia = "consumidor"
    elif any(x in fname for x in ["imóvel", "imobiliário", "construtivo", "posse", "despejo", "locação"]):
        materia = "civel"
    elif any(x in fname for x in ["monitória", "cobrança"]):
        materia = "civel"
    elif any(x in fname for x in ["vale"]):
        materia = "civel"
    elif any(x in fname for x in ["seguro"]):
        materia = "consumidor"
    else:
        materia = "civel"

    # Detectar resultado
    if any(x in fname for x in ["improcedente"]):
        resultado = "improcedente"
    elif any(x in fname for x in ["procedente", "acolhimento"]):
        resultado = "procedente"
    elif any(x in fname for x in ["parcial"]):
        resultado = "parcialmente_procedente"
    elif any(x in fname for x in ["deferimento"]):
        resultado = "deferido"
    elif any(x in fname for x in ["indeferimento", "indefere"]):
        resultado = "indeferido"
    elif any(x in fname for x in ["não acolhimento"]):
        resultado = "nao_acolhido"
    elif any(x in fname for x in ["revelia"]):
        resultado = "procedente"
    elif any(x in fname for x in ["extinção"]):
        resultado = "extinto"
    else:
        resultado = "n/a"

    return {
        "tipo": tipo,
        "materia": materia,
        "resultado": resultado,
    }


def extract_sections(text: str) -> dict:
    """Tenta extrair seções da minuta (relatório, fundamentação, dispositivo)."""
    sections = {"relatorio": "", "fundamentacao": "", "dispositivo": "", "full_text": text}

    # Padrões comuns de seção
    lines = text.split("\n")
    current_section = "relatorio"

    for line in lines:
        line_lower = line.lower().strip()

        if any(x in line_lower for x in ["fundamentação", "fundamentacao", "fundamenta", "do direito", "mérito"]):
            current_section = "fundamentacao"
        elif any(x in line_lower for x in ["dispositivo", "ante o exposto", "diante do exposto",
                                            "julgo", "isto posto", "posto isso", "ex positis"]):
            current_section = "dispositivo"

        sections[current_section] += line + "\n"

    return sections


def main():
    if not TEMPLATE_DIR.exists():
        print(f"❌ Diretório não encontrado: {TEMPLATE_DIR}")
        return

    docx_files = sorted(TEMPLATE_DIR.glob("*.docx"))
    print(f"📂 Encontrados {len(docx_files)} modelos de decisão\n")

    # Extrair todos
    templates = []
    type_counts = {}

    for f in docx_files:
        try:
            text = extract_docx_text(str(f))
            if len(text) < 100:
                print(f"  ⚠️ Pulando {f.name} (muito curto: {len(text)} chars)")
                continue

            classification = classify_template(f.name, text)
            sections = extract_sections(text)

            template = {
                "filename": f.name,
                "classification": classification,
                "text": text,
                "sections": sections,
                "char_count": len(text),
            }
            templates.append(template)

            tipo = classification["tipo"]
            type_counts[tipo] = type_counts.get(tipo, 0) + 1

        except Exception as e:
            print(f"  ❌ Erro em {f.name}: {e}")

    print(f"\n✅ Extraídos {len(templates)} modelos")
    print(f"📊 Por tipo: {json.dumps(type_counts, ensure_ascii=False)}")

    # ── Dataset 1: Router (classificação) ────────────────────────────────
    router_data = []
    for t in templates:
        router_data.append({
            "messages": [
                {"role": "system", "content": "Você é um classificador processual. Classifique o tipo de processo judicial. Responda APENAS com JSON: {\"tipo\": \"sentenca|saneamento|despacho|tutela_urgencia|homologacao\", \"confianca\": 1.0, \"materia\": \"civel|consumidor|familia|fazenda_publica|outro\"}"},
                {"role": "user", "content": t["text"][:2000]},
                {"role": "assistant", "content": json.dumps({
                    "tipo": t["classification"]["tipo"],
                    "confianca": 1.0,
                    "materia": t["classification"]["materia"],
                }, ensure_ascii=False)},
            ]
        })

    router_dir = OUTPUT_DIR / "router"
    os.makedirs(router_dir, exist_ok=True)

    # Split 80/10/10
    import random
    random.seed(42)
    random.shuffle(router_data)
    n = len(router_data)
    split1 = int(n * 0.8)
    split2 = int(n * 0.9)

    for subset, name in [(router_data[:split1], "train"), (router_data[split1:split2], "valid"), (router_data[split2:], "test")]:
        path = router_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in subset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  💾 Router {name}: {len(subset)} exemplos → {path}")

    # ── Dataset 2: Redator (estilo de minuta) ────────────────────────────
    redator_data = []
    for t in templates:
        tipo = t["classification"]["tipo"]
        materia = t["classification"]["materia"]
        resultado = t["classification"]["resultado"]

        # O "input" do redator seria o raciocínio jurídico resumido
        # O "output" é a minuta real do magistrado
        instrucao = (
            f"Redija uma {tipo.replace('_', ' ')} de matéria {materia}, "
            f"com resultado {resultado}. "
            f"Tema: {t['filename'].replace('.docx', '')}."
        )

        redator_data.append({
            "messages": [
                {"role": "system", "content": "Você é um ghostwriter judicial. Transforme o raciocínio do Juiz em sentença/decisão formal pronta para assinatura. Use linguagem jurídica culta brasileira. Siga o estilo do magistrado da 4ª Vara Cível de Belo Horizonte."},
                {"role": "user", "content": instrucao},
                {"role": "assistant", "content": t["text"]},
            ]
        })

    redator_dir = OUTPUT_DIR / "redator"
    os.makedirs(redator_dir, exist_ok=True)

    random.shuffle(redator_data)
    n = len(redator_data)
    split1 = int(n * 0.8)
    split2 = int(n * 0.9)

    for subset, name in [(redator_data[:split1], "train"), (redator_data[split1:split2], "valid"), (redator_data[split2:], "test")]:
        path = redator_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in subset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  💾 Redator {name}: {len(subset)} exemplos → {path}")

    # ── Dossiê de Estilo ─────────────────────────────────────────────────
    print("\n📝 Analisando padrões de estilo...")

    # Analisar padrões de abertura
    aberturas = []
    for t in templates:
        first_lines = t["text"][:200]
        aberturas.append(first_lines)

    # Contar conectivos mais usados
    all_text = " ".join(t["text"] for t in templates)
    conectivos = {}
    for c in ["Ante o exposto", "Diante do exposto", "Posto isso", "Isto posto",
              "Ex positis", "Vistos etc", "VISTOS", "Fundamento e decido",
              "Relatório dispensado", "Sem mais delongas",
              "JULGO PROCEDENTE", "JULGO IMPROCEDENTE", "JULGO PARCIALMENTE"]:
        count = all_text.lower().count(c.lower())
        if count > 0:
            conectivos[c] = count

    dossie = {
        "total_modelos": len(templates),
        "tipos": type_counts,
        "conectivos_frequentes": dict(sorted(conectivos.items(), key=lambda x: -x[1])),
        "char_medio": sum(t["char_count"] for t in templates) // len(templates),
        "aberturas_comuns": list(set(a[:50] for a in aberturas))[:10],
    }

    dossie_path = OUTPUT_DIR / "style_dossier.json"
    with open(dossie_path, "w", encoding="utf-8") as f:
        json.dump(dossie, f, ensure_ascii=False, indent=2)
    print(f"  💾 Dossiê de estilo: {dossie_path}")
    print(f"     Conectivos: {json.dumps(conectivos, ensure_ascii=False)}")

    print(f"\n🏁 Concluído! {len(templates)} modelos processados.")
    print(f"   Router: {len(router_data)} exemplos")
    print(f"   Redator: {len(redator_data)} exemplos")


if __name__ == "__main__":
    main()
