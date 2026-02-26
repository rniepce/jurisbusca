"""
ETL: Indexa os TXTs de jurisprudência do TJMG em SQLite + FTS5.

Uso:
    python jurisprudence_indexer.py --source ~/Downloads/jurisprudencia --output data/jurisprudencia.db

Estrutura esperada:
    jurisprudencia/
    ├── 2020/01/*.txt
    ├── 2020/02/*.txt
    ...
    └── 2026/02/*.txt

Cada arquivo: {num_processo}_{AAAA-MM-DD}.txt
"""

import os
import re
import sys
import glob
import time
import sqlite3
import argparse


def extract_metadata(filepath: str) -> dict:
    """Extrai metadados do caminho e nome do arquivo."""
    basename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(basename)[0]

    # Parse: {numero_processo}_{AAAA-MM-DD}
    parts = name_no_ext.rsplit("_", 1)
    numero_processo = parts[0] if len(parts) >= 1 else name_no_ext
    data_publicacao = parts[1] if len(parts) >= 2 else ""

    # Ano e mês da estrutura de pastas  (.../2024/06/file.txt)
    path_parts = filepath.replace("\\", "/").split("/")
    ano = ""
    mes = ""
    for i, p in enumerate(path_parts):
        if re.match(r"^20\d{2}$", p):
            ano = p
            if i + 1 < len(path_parts) and re.match(r"^\d{2}$", path_parts[i + 1]):
                mes = path_parts[i + 1]
            break

    return {
        "numero_processo": numero_processo,
        "data_publicacao": data_publicacao,
        "ano": int(ano) if ano else 0,
        "mes": int(mes) if mes else 0,
    }


def extract_ementa(text: str) -> str:
    """
    Extrai a ementa do texto do acórdão.
    Padrões comuns:
      - Começa com "EMENTA:" e vai até a linha do tipo de recurso + número
      - Ou as primeiras linhas antes de "COMARCA DE"
    """
    lines = text.strip().split("\n")

    # Pula tags tipo <CABB...> no início
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("<"):
            start_idx = i
            break

    # Busca o bloco da ementa
    ementa_lines = []
    in_ementa = False

    for i in range(start_idx, min(len(lines), 60)):  # Olha só as 60 primeiras linhas
        line = lines[i].strip()

        if not in_ementa:
            if line.upper().startswith("EMENTA:") or line.upper().startswith("EMENTA -"):
                in_ementa = True
                ementa_lines.append(line)
            elif i == start_idx and len(line) > 50:
                # Primeira linha não-vazia é longa — provavelmente a própria ementa
                in_ementa = True
                ementa_lines.append(line)
        else:
            # Termina quando encontra a linha do tipo de recurso (APELAÇÃO CÍVEL Nº, AGRAVO DE INSTRUMENTO Nº, etc.)
            if re.match(
                r"^(APELAÇÃO|AGRAVO|AÇÃO|EMBARGOS|RECURSO|MANDADO|HABEAS|CONFLITO|INCIDENTE|REMESSA)",
                line.upper(),
            ):
                break
            if not line:
                # Linha em branco pode indicar fim da ementa
                if ementa_lines:
                    break
            else:
                ementa_lines.append(line)

    ementa = " ".join(ementa_lines).strip()

    # Fallback: se não achou ementa, usa as primeiras 500 chars
    if not ementa or len(ementa) < 20:
        ementa = text[:500].strip()

    return ementa[:2000]  # Limita a 2000 chars


def extract_tipo_recurso(text: str) -> str:
    """Extrai o tipo do recurso (Apelação Cível, Agravo de Instrumento, etc.)."""
    patterns = [
        r"(APELAÇÃO CÍVEL)",
        r"(AGRAVO DE INSTRUMENTO)",
        r"(AGRAVO INTERNO)",
        r"(EMBARGOS DE DECLARAÇÃO)",
        r"(EMBARGOS INFRINGENTES)",
        r"(AÇÃO RESCISÓRIA)",
        r"(RECURSO (EM SENTIDO )?ESTRITO)",
        r"(MANDADO DE SEGURANÇA)",
        r"(HABEAS CORPUS)",
        r"(CONFLITO DE COMPETÊNCIA)",
        r"(INCIDENTE DE [A-ZÇÃ ]+)",
        r"(REMESSA NECESSÁRIA)",
        r"(RECURSO INOMINADO)",
    ]
    text_upper = text[:3000].upper()
    for pat in patterns:
        m = re.search(pat, text_upper)
        if m:
            return m.group(1).title()
    return ""


def extract_relator(text: str) -> str:
    """Extrai o nome do relator."""
    # Padrões comuns: "Relator(a): Des.(a) NOME"
    m = re.search(
        r"Relator\(?a?\)?:?\s*Des\.?\(?a?\)?\s*(.+?)(?:\s*,|\s*\n|\s*-)",
        text[:5000],
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()[:200]
    return ""


def extract_comarca(text: str) -> str:
    """Extrai a comarca."""
    m = re.search(r"COMARCA DE\s+([A-ZÀ-Ü\s]+?)(?:\s*-|\s*\n)", text[:5000])
    if m:
        return m.group(1).strip().title()[:200]
    return ""


def create_database(db_path: str):
    """Cria o banco SQLite com tabela principal + FTS5."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS acordaos_fts")
    c.execute("DROP TABLE IF EXISTS acordaos")

    c.execute("""
        CREATE TABLE acordaos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            numero_processo TEXT,
            data_publicacao TEXT,
            ano INTEGER,
            mes INTEGER,
            tipo_recurso TEXT,
            relator TEXT,
            comarca TEXT,
            ementa TEXT,
            texto_completo TEXT,
            arquivo_origem TEXT
        )
    """)

    # FTS5 virtual table para busca full-text
    c.execute("""
        CREATE VIRTUAL TABLE acordaos_fts USING fts5(
            ementa,
            texto_completo,
            numero_processo,
            content='acordaos',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        )
    """)

    # Triggers para manter FTS sincronizado
    c.execute("""
        CREATE TRIGGER acordaos_ai AFTER INSERT ON acordaos BEGIN
            INSERT INTO acordaos_fts(rowid, ementa, texto_completo, numero_processo)
            VALUES (new.id, new.ementa, new.texto_completo, new.numero_processo);
        END
    """)

    c.execute("CREATE INDEX idx_acordaos_ano ON acordaos(ano)")
    c.execute("CREATE INDEX idx_acordaos_data ON acordaos(data_publicacao)")
    c.execute("CREATE INDEX idx_acordaos_tipo ON acordaos(tipo_recurso)")

    conn.commit()
    return conn


def index_files(source_dir: str, db_path: str):
    """Indexa todos os TXTs do diretório fonte no SQLite."""
    print(f"🔍 Buscando arquivos .txt em {source_dir}...")

    txt_files = glob.glob(os.path.join(source_dir, "**", "*.txt"), recursive=True)
    total = len(txt_files)
    print(f"📄 Encontrados {total:,} arquivos")

    if total == 0:
        print("❌ Nenhum arquivo encontrado!")
        return

    # Garante que o diretório de saída existe
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    print(f"🗃️  Criando banco em {db_path}...")
    conn = create_database(db_path)
    cursor = conn.cursor()

    start = time.time()
    batch = []
    batch_size = 500
    errors = 0

    for i, filepath in enumerate(txt_files, 1):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            if not text.strip():
                continue

            meta = extract_metadata(filepath)
            ementa = extract_ementa(text)
            tipo = extract_tipo_recurso(text)
            relator = extract_relator(text)
            comarca = extract_comarca(text)

            batch.append((
                meta["numero_processo"],
                meta["data_publicacao"],
                meta["ano"],
                meta["mes"],
                tipo,
                relator,
                comarca,
                ementa,
                text,
                os.path.relpath(filepath, source_dir),
            ))

            if len(batch) >= batch_size:
                cursor.executemany(
                    """INSERT INTO acordaos
                       (numero_processo, data_publicacao, ano, mes, tipo_recurso, relator, comarca, ementa, texto_completo, arquivo_origem)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    batch,
                )
                conn.commit()
                batch = []

                elapsed = time.time() - start
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  ⏳ {i:,}/{total:,} ({i*100//total}%) — {rate:.0f} docs/s — ETA {eta:.0f}s", end="\r")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  ⚠️ Erro em {filepath}: {e}")

    # Flush remaining batch
    if batch:
        cursor.executemany(
            """INSERT INTO acordaos
               (numero_processo, data_publicacao, ano, mes, tipo_recurso, relator, comarca, ementa, texto_completo, arquivo_origem)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch,
        )
        conn.commit()

    elapsed = time.time() - start
    count = cursor.execute("SELECT COUNT(*) FROM acordaos").fetchone()[0]

    print(f"\n\n✅ Indexação concluída!")
    print(f"   📄 {count:,} acórdãos indexados")
    print(f"   ⏱️  {elapsed:.1f}s ({count/elapsed:.0f} docs/s)")
    print(f"   💾 {os.path.getsize(db_path) / 1024 / 1024:.1f} MB")
    if errors:
        print(f"   ⚠️  {errors} erros")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Indexa jurisprudência TJMG em SQLite + FTS5")
    parser.add_argument(
        "--source",
        default=os.path.expanduser("~/Downloads/jurisprudencia"),
        help="Diretório raiz dos TXTs (default: ~/Downloads/jurisprudencia)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "data", "jurisprudencia.db"),
        help="Caminho do banco SQLite de saída (default: data/jurisprudencia.db)",
    )
    args = parser.parse_args()

    index_files(args.source, args.output)


if __name__ == "__main__":
    main()
