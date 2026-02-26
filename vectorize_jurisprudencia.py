"""
Vetorização de jurisprudência TJMG usando Azure OpenAI text-embedding-3-large (1024 dim).

Uso:
    python vectorize_jurisprudencia.py

Lê o SQLite (data/jurisprudencia.db), gera embeddings das ementas,
e salva no mesmo banco numa tabela 'embeddings' + arquivo numpy de backup.
"""

import os
import sys
import time
import json
import sqlite3
import struct
import numpy as np
from openai import AzureOpenAI

# ── Config ──────────────────────────────────────────────────────────────────
AZURE_ENDPOINT = os.environ.get("AZURE_EMBEDDING_ENDPOINT", "https://assistente-web-resource.cognitiveservices.azure.com")
AZURE_API_KEY = os.environ.get("AZURE_EMBEDDING_KEY", os.environ.get("AZURE_AI_KEY", ""))
AZURE_API_VERSION = "2024-12-01-preview"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 1024

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "jurisprudencia.db")
BACKUP_NPY = os.path.join(os.path.dirname(__file__), "data", "jurisprudencia_embeddings.npz")

# Azure OpenAI rate limits: ~350K tokens/min for text-embedding-3-large
# Ementa média ~200 tokens → ~1750 ementas/min → batch de 100 a cada ~3.5s
BATCH_SIZE = 100
MAX_TOKENS_PER_BATCH = 300_000  # Safety margin
RETRY_WAIT = 10  # seconds on rate limit


def create_embeddings_table(conn: sqlite3.Connection):
    """Cria tabela de embeddings se não existir."""
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            acordao_id INTEGER PRIMARY KEY,
            embedding BLOB,
            FOREIGN KEY (acordao_id) REFERENCES acordaos(id)
        )
    """)
    conn.commit()


def float_list_to_blob(floats: list) -> bytes:
    """Converte lista de floats para blob binário (compact storage)."""
    return struct.pack(f'{len(floats)}f', *floats)


def blob_to_float_list(blob: bytes) -> list:
    """Converte blob binário de volta para lista de floats."""
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def get_pending_documents(conn: sqlite3.Connection) -> list:
    """Retorna documentos ainda não vetorizados."""
    c = conn.cursor()
    c.execute("""
        SELECT a.id, a.ementa
        FROM acordaos a
        LEFT JOIN embeddings e ON a.id = e.acordao_id
        WHERE e.acordao_id IS NULL
        ORDER BY a.id
    """)
    return c.fetchall()


def truncate_text(text: str, max_chars: int = 6000) -> str:
    """Trunca texto para evitar excesso de tokens (1 token ≈ 4 chars PT)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def embed_batch(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """Gera embeddings para um batch de textos com retry em rate limits."""
    for attempt in range(5):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIM,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait = RETRY_WAIT * (attempt + 1)
                print(f"\n  ⏳ Rate limit — aguardando {wait}s (tentativa {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit excedido após 5 tentativas")


def main():
    print(f"🚀 Vetorização de Jurisprudência TJMG")
    print(f"   Modelo: {EMBEDDING_MODEL}")
    print(f"   Dimensões: {EMBEDDING_DIM}")
    print(f"   Endpoint: {AZURE_ENDPOINT}")
    print()

    if not os.path.exists(DB_PATH):
        print(f"❌ Banco não encontrado: {DB_PATH}")
        print("   Execute primeiro: python jurisprudence_indexer.py")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    create_embeddings_table(conn)

    # Busca documentos pendentes
    pending = get_pending_documents(conn)
    total = len(pending)
    print(f"📄 Documentos pendentes: {total:,}")

    if total == 0:
        print("✅ Todos os documentos já foram vetorizados!")
        conn.close()
        return

    # Init Azure client
    client = AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=AZURE_API_VERSION,
    )

    cursor = conn.cursor()
    start_time = time.time()
    processed = 0
    errors = 0

    # Process in batches
    for batch_start in range(0, total, BATCH_SIZE):
        batch = pending[batch_start : batch_start + BATCH_SIZE]
        ids = [row[0] for row in batch]
        texts = [truncate_text(row[1]) for row in batch]

        # Filter out empty texts
        valid = [(i, t) for i, t in zip(ids, texts) if t.strip()]
        if not valid:
            continue

        valid_ids, valid_texts = zip(*valid)

        try:
            embeddings = embed_batch(client, list(valid_texts))

            # Save to SQLite
            for doc_id, emb in zip(valid_ids, embeddings):
                blob = float_list_to_blob(emb)
                cursor.execute(
                    "INSERT OR REPLACE INTO embeddings (acordao_id, embedding) VALUES (?, ?)",
                    (doc_id, blob),
                )

            conn.commit()
            processed += len(valid_ids)

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(
                f"  ⚡ {processed:,}/{total:,} ({processed*100//total}%) — "
                f"{rate:.0f} docs/s — ETA {eta:.0f}s",
                end="\r",
            )

        except Exception as e:
            errors += 1
            print(f"\n  ⚠️ Erro no batch {batch_start}: {e}")
            if errors > 20:
                print("❌ Muitos erros consecutivos, abortando.")
                break
            time.sleep(5)

    # Final stats
    elapsed = time.time() - start_time
    total_done = cursor.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    print(f"\n\n✅ Vetorização concluída!")
    print(f"   📄 {total_done:,} documentos vetorizados")
    print(f"   📐 {EMBEDDING_DIM} dimensões (text-embedding-3-large)")
    print(f"   ⏱️  {elapsed:.1f}s")
    if errors:
        print(f"   ⚠️  {errors} erros")

    # Save numpy backup
    print(f"\n💾 Salvando backup numpy...")
    all_rows = cursor.execute(
        "SELECT acordao_id, embedding FROM embeddings ORDER BY acordao_id"
    ).fetchall()

    ids_arr = np.array([r[0] for r in all_rows], dtype=np.int32)
    emb_arr = np.array([blob_to_float_list(r[1]) for r in all_rows], dtype=np.float32)

    np.savez_compressed(
        BACKUP_NPY,
        ids=ids_arr,
        embeddings=emb_arr,
        metadata={
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIM,
            "count": len(ids_arr),
        },
    )

    backup_size = os.path.getsize(BACKUP_NPY) / 1024 / 1024
    db_size = os.path.getsize(DB_PATH) / 1024 / 1024
    print(f"   💾 Backup: {backup_size:.1f} MB ({BACKUP_NPY})")
    print(f"   💾 DB total: {db_size:.1f} MB")

    conn.close()
    print("\n🎉 Pronto!")


if __name__ == "__main__":
    main()
