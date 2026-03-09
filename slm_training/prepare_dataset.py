"""
Dataset Preparation — Converte dados jurídicos brutos para ChatML JSONL.

Uso:
    python slm_training/prepare_dataset.py --role extrator --input data/raw/ --output data/extrator/

Formatos de entrada suportados:
    - JSON: {"processo": "...", "anotacao": {...}}
    - TXT: texto bruto de processo (sem anotação — gera template para anotar)
"""

import json
import os
import random
import argparse
from pathlib import Path

# ── Prompts de sistema (importados para embedding no dataset) ────────────────
SYSTEM_PROMPTS = {
    "router": "Você é um classificador processual. Classifique o tipo de processo judicial. Responda APENAS com JSON: {\"tipo\": \"sentenca|saneamento|despacho|tutela_urgencia|homologacao\", \"confianca\": 0.0-1.0, \"materia\": \"civel|consumidor|familia|fazenda_publica|outro\"}",
    "extrator": "Você é um extrator de informações processuais. Extraia os dados do processo e responda APENAS com JSON estruturado contendo: autor, reu, acao, pedidos, valor_causa, causa_de_pedir, datas_chave, pontos_controvertidos.",
    "jurista": "Você é um Juiz de Direito de Vara Cível do TJMG. Analise os fatos extraídos e produza raciocínio jurídico usando silogismo judicial. Use APENAS lei federal (CPC, CC, CDC) e Súmulas STJ/STF.",
    "redator": "Você é um ghostwriter judicial. Transforme o raciocínio do Juiz em sentença/decisão formal pronta para assinatura. Use linguagem jurídica culta brasileira.",
    "auditor": "Você é um auditor judicial. Verifique a minuta contra os fatos originais em 3 dimensões: integridade fática, eficiência (Prov. 355), e congruência jurídica. Responda em JSON.",
}


def create_chatml_entry(system: str, user: str, assistant: str) -> dict:
    """Cria uma entrada no formato ChatML para fine-tuning."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def split_dataset(entries: list, train_ratio=0.8, valid_ratio=0.1):
    """Split 80/10/10."""
    random.shuffle(entries)
    n = len(entries)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return entries[:train_end], entries[train_end:valid_end], entries[valid_end:]


def save_jsonl(entries: list, path: str):
    """Salva lista de dicts como JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  💾 Salvou {len(entries)} exemplos em {path}")


def load_raw_data(input_dir: str) -> list[dict]:
    """Carrega dados brutos de uma pasta."""
    entries = []
    input_path = Path(input_dir)

    for f in sorted(input_path.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                entries.extend(data)
            else:
                entries.append(data)

    for f in sorted(input_path.glob("*.jsonl")):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    print(f"📂 Carregados {len(entries)} exemplos de {input_dir}")
    return entries


def prepare_router_dataset(entries: list) -> list[dict]:
    """Converte dados para formato de treino do Router."""
    chatml = []
    for e in entries:
        processo = e.get("processo", e.get("texto", ""))
        anotacao = e.get("anotacao", e.get("classificacao", {}))
        if not processo or not anotacao:
            continue
        chatml.append(create_chatml_entry(
            system=SYSTEM_PROMPTS["router"],
            user=processo[:2000],  # Router usa apenas primeiros 2K chars
            assistant=json.dumps(anotacao, ensure_ascii=False),
        ))
    return chatml


def prepare_extrator_dataset(entries: list) -> list[dict]:
    """Converte dados para formato de treino do Extrator."""
    chatml = []
    for e in entries:
        processo = e.get("processo", e.get("texto", ""))
        fatos = e.get("fatos", e.get("anotacao", {}))
        if not processo or not fatos:
            continue
        chatml.append(create_chatml_entry(
            system=SYSTEM_PROMPTS["extrator"],
            user=processo,
            assistant=json.dumps(fatos, ensure_ascii=False),
        ))
    return chatml


def prepare_jurista_dataset(entries: list) -> list[dict]:
    """Converte dados para formato de treino do Jurista."""
    chatml = []
    for e in entries:
        fatos = e.get("fatos", {})
        analise = e.get("analise", e.get("fundamentacao", ""))
        if not fatos or not analise:
            continue
        chatml.append(create_chatml_entry(
            system=SYSTEM_PROMPTS["jurista"],
            user=json.dumps(fatos, ensure_ascii=False),
            assistant=analise,
        ))
    return chatml


def prepare_redator_dataset(entries: list) -> list[dict]:
    """Converte dados para formato de treino do Redator."""
    chatml = []
    for e in entries:
        analise = e.get("analise", e.get("fundamentacao", ""))
        minuta = e.get("minuta", e.get("sentenca", ""))
        if not analise or not minuta:
            continue
        chatml.append(create_chatml_entry(
            system=SYSTEM_PROMPTS["redator"],
            user=analise,
            assistant=minuta,
        ))
    return chatml


def prepare_auditor_dataset(entries: list) -> list[dict]:
    """Converte dados para formato de treino do Auditor."""
    chatml = []
    for e in entries:
        fatos = e.get("fatos", {})
        minuta = e.get("minuta", "")
        audit = e.get("auditoria", e.get("audit", {}))
        if not minuta or not audit:
            continue
        user_text = f"FATOS:\n{json.dumps(fatos, ensure_ascii=False)}\n\nMINUTA:\n{minuta}"
        chatml.append(create_chatml_entry(
            system=SYSTEM_PROMPTS["auditor"],
            user=user_text,
            assistant=json.dumps(audit, ensure_ascii=False),
        ))
    return chatml


PREPARERS = {
    "router": prepare_router_dataset,
    "extrator": prepare_extrator_dataset,
    "jurista": prepare_jurista_dataset,
    "redator": prepare_redator_dataset,
    "auditor": prepare_auditor_dataset,
}


def generate_annotation_template(processo_text: str, role: str) -> dict:
    """Gera um template de anotação para um processo (para o magistrado preencher)."""
    templates = {
        "router": {"tipo": "sentenca", "confianca": 1.0, "materia": "civel"},
        "extrator": {
            "autor": "PREENCHER", "reu": "PREENCHER",
            "acao": "PREENCHER", "pedidos": ["PREENCHER"],
            "valor_causa": "R$ 0,00", "causa_de_pedir": "PREENCHER",
        },
        "jurista": {"analise": "PREENCHER COM FUNDAMENTAÇÃO"},
        "redator": {"minuta": "PREENCHER COM MINUTA ASSINADA"},
        "auditor": {"aprovado": True, "score": 100, "erros": []},
    }
    return {
        "processo": processo_text[:5000],
        "anotacao": templates.get(role, {}),
        "_instrucao": f"Preencha o campo 'anotacao' com os dados corretos para treinar o {role}.",
    }


def main():
    parser = argparse.ArgumentParser(description="Preparar dataset ChatML para fine-tuning SLM")
    parser.add_argument("--role", required=True, choices=list(PREPARERS.keys()),
                        help="Papel do SLM (router, extrator, jurista, redator, auditor)")
    parser.add_argument("--input", required=True, help="Pasta com dados brutos (JSON/JSONL)")
    parser.add_argument("--output", required=True, help="Pasta de saída para train/valid/test.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Seed para split aleatório")
    args = parser.parse_args()

    random.seed(args.seed)

    entries = load_raw_data(args.input)
    if not entries:
        print("❌ Nenhum dado encontrado. Gerando templates de anotação...")
        # Se não tem dados, gera templates para o magistrado anotar
        template = generate_annotation_template("COLE O TEXTO DO PROCESSO AQUI", args.role)
        template_path = os.path.join(args.output, f"template_{args.role}.json")
        os.makedirs(args.output, exist_ok=True)
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump([template], f, ensure_ascii=False, indent=2)
        print(f"📝 Template salvo em {template_path} — Peça ao magistrado para preencher.")
        return

    preparer = PREPARERS[args.role]
    chatml_entries = preparer(entries)
    print(f"✅ Convertidos {len(chatml_entries)} exemplos para ChatML")

    if len(chatml_entries) < 3:
        print("⚠️ Poucos exemplos. Salvando tudo como train.jsonl")
        save_jsonl(chatml_entries, os.path.join(args.output, "train.jsonl"))
        return

    train, valid, test = split_dataset(chatml_entries)
    save_jsonl(train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(valid, os.path.join(args.output, "valid.jsonl"))
    save_jsonl(test, os.path.join(args.output, "test.jsonl"))

    print(f"\n📊 Split: {len(train)} train / {len(valid)} valid / {len(test)} test")


if __name__ == "__main__":
    main()
