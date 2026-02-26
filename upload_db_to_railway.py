"""
Upload do banco de jurisprudência para o Railway via endpoint admin.

Uso:
    python upload_db_to_railway.py --url https://SEU-APP.railway.app --key SUA_ADMIN_KEY

Comprime o DB com gzip (~687MB → ~200MB) e envia via HTTP multipart.
O endpoint descompacta e salva no Railway Volume.
"""

import os
import sys
import gzip
import time
import argparse
import requests

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "jurisprudencia.db")
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB chunks for progress display


def compress_db(db_path: str, out_path: str) -> int:
    """Comprime o DB com gzip. Retorna tamanho comprimido."""
    original_size = os.path.getsize(db_path)
    print(f"📦 Comprimindo {db_path} ({original_size / 1024 / 1024:.0f} MB)...")

    start = time.time()
    with open(db_path, "rb") as f_in:
        with gzip.open(out_path, "wb", compresslevel=6) as f_out:
            while True:
                chunk = f_in.read(CHUNK_SIZE)
                if not chunk:
                    break
                f_out.write(chunk)
                pos = f_in.tell()
                pct = pos * 100 // original_size
                print(f"  {pct}% comprimido...", end="\r")

    compressed_size = os.path.getsize(out_path)
    elapsed = time.time() - start
    ratio = compressed_size / original_size * 100
    print(f"\n  ✅ {original_size/1024/1024:.0f} MB → {compressed_size/1024/1024:.0f} MB ({ratio:.0f}%) em {elapsed:.1f}s")
    return compressed_size


def upload_to_railway(file_path: str, base_url: str, admin_key: str):
    """Upload do DB comprimido para o Railway."""
    file_size = os.path.getsize(file_path)
    print(f"\n🚀 Enviando {file_size / 1024 / 1024:.0f} MB para {base_url}...")

    url = f"{base_url.rstrip('/')}/api/admin/upload-jurisprudencia"

    start = time.time()
    with open(file_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": ("jurisprudencia.db.gz", f, "application/gzip")},
            headers={"X-Admin-Key": admin_key},
            timeout=600,  # 10 min timeout for large file
        )

    elapsed = time.time() - start

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Upload concluído em {elapsed:.0f}s!")
        print(f"   📄 {result.get('total_acordaos', '?')} acórdãos disponíveis")
        print(f"   💾 {result.get('db_size_mb', '?')} MB no servidor")
    else:
        print(f"\n❌ Erro {response.status_code}: {response.text[:500]}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload jurisprudência DB para Railway")
    parser.add_argument("--url", required=True, help="URL base do app Railway (ex: https://app.railway.app)")
    parser.add_argument("--key", required=True, help="Admin key (ADMIN_KEY env var do Railway)")
    parser.add_argument("--db", default=DB_PATH, help=f"Caminho do DB local (default: {DB_PATH})")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"❌ DB não encontrado: {args.db}")
        sys.exit(1)

    # Compress
    gz_path = args.db + ".gz"
    compress_db(args.db, gz_path)

    # Upload
    try:
        upload_to_railway(gz_path, args.url, args.key)
    finally:
        # Cleanup compressed file
        if os.path.exists(gz_path):
            os.remove(gz_path)
            print(f"🧹 Arquivo temporário removido: {gz_path}")


if __name__ == "__main__":
    main()
