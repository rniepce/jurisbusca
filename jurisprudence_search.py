"""
Jurisprudence Search Engine — SQLite FTS5 based.

Provides search, document retrieval, and statistics for TJMG case law.
"""

import os
import re
import sqlite3
from typing import Optional


_DB_PATH = os.environ.get(
    "JURISPRUDENCIA_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "jurisprudencia.db"),
)

# Module-level connection (lazy init)
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    """Lazy-init SQLite connection with WAL mode for concurrent reads."""
    global _conn
    if _conn is None:
        if not os.path.exists(_DB_PATH):
            raise FileNotFoundError(
                f"Banco de jurisprudência não encontrado em {_DB_PATH}. "
                f"Execute: python jurisprudence_indexer.py"
            )
        _conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA query_only=ON")
    return _conn


def _sanitize_fts_query(query: str) -> str:
    """
    Sanitiza a query do usuário para FTS5.
    Remove caracteres especiais que causam erro no parser FTS5.
    Mantém aspas duplas para busca de frase exata.
    """
    if not query or not query.strip():
        return ""

    # Se o usuário usou aspas duplas, tenta manter como phrase query
    if '"' in query:
        # Garante que as aspas estão balanceadas
        count = query.count('"')
        if count % 2 != 0:
            query = query.replace('"', '')
        else:
            # Remove especiais fora das aspas mas mantém as frases
            return query.strip()

    # Remove caracteres especiais do FTS5 que causam parse error
    query = re.sub(r'[*(){}[\]^~!@#$%&+=|\\/<>:;]', ' ', query)

    # Converte múltiplos espaços
    query = re.sub(r'\s+', ' ', query).strip()

    if not query:
        return ""

    # Adiciona * ao final de cada token para prefix matching
    tokens = query.split()
    # Filtra tokens muito curtos (ruído)
    tokens = [t for t in tokens if len(t) >= 2]

    if not tokens:
        return ""

    # Usa OR implícito (FTS5 default), cada token com prefix match
    return " OR ".join(f'"{t}"*' for t in tokens)


def search(
    query: str,
    ano_inicio: int = 0,
    ano_fim: int = 9999,
    tipo_recurso: str = "",
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """
    Busca full-text nos acórdãos.

    Returns:
        {
            "results": [{"id", "numero_processo", "data_publicacao", "ano", "tipo_recurso",
                          "relator", "comarca", "ementa", "snippet", "rank"}],
            "total": int,
            "page": int,
            "page_size": int,
            "pages": int,
            "query": str
        }
    """
    conn = _get_conn()

    fts_query = _sanitize_fts_query(query)
    if not fts_query:
        return {
            "results": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "pages": 0,
            "query": query,
        }

    offset = (max(1, page) - 1) * page_size

    # Build WHERE clause for filters
    where_parts = ["a.ano >= ? AND a.ano <= ?"]
    params_count = [ano_inicio, ano_fim]
    params_search = [ano_inicio, ano_fim]

    if tipo_recurso:
        where_parts.append("a.tipo_recurso LIKE ?")
        tipo_param = f"%{tipo_recurso}%"
        params_count.append(tipo_param)
        params_search.append(tipo_param)

    where_clause = " AND ".join(where_parts)

    try:
        # Count total results
        count_sql = f"""
            SELECT COUNT(*) FROM acordaos_fts
            JOIN acordaos a ON acordaos_fts.rowid = a.id
            WHERE acordaos_fts MATCH ?
            AND {where_clause}
        """
        total = conn.execute(count_sql, [fts_query] + params_count).fetchone()[0]

        # Fetch page of results with snippets
        search_sql = f"""
            SELECT
                a.id,
                a.numero_processo,
                a.data_publicacao,
                a.ano,
                a.mes,
                a.tipo_recurso,
                a.relator,
                a.comarca,
                a.ementa,
                snippet(acordaos_fts, 0, '<mark>', '</mark>', '...', 40) as snippet,
                rank
            FROM acordaos_fts
            JOIN acordaos a ON acordaos_fts.rowid = a.id
            WHERE acordaos_fts MATCH ?
            AND {where_clause}
            ORDER BY rank
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(
            search_sql, [fts_query] + params_search + [page_size, offset]
        ).fetchall()

        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "numero_processo": row["numero_processo"],
                "data_publicacao": row["data_publicacao"],
                "ano": row["ano"],
                "mes": row["mes"],
                "tipo_recurso": row["tipo_recurso"] or "",
                "relator": row["relator"] or "",
                "comarca": row["comarca"] or "",
                "ementa": row["ementa"] or "",
                "snippet": row["snippet"] or "",
            })

        pages = (total + page_size - 1) // page_size if total > 0 else 0

        return {
            "results": results,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "query": query,
        }

    except Exception as e:
        print(f"⚠️ Erro na busca de jurisprudência: {e}")
        return {
            "results": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "pages": 0,
            "query": query,
            "error": str(e),
        }


def get_document(doc_id: int) -> Optional[dict]:
    """Retorna o texto completo de um acórdão pelo ID."""
    conn = _get_conn()

    row = conn.execute(
        """SELECT id, numero_processo, data_publicacao, ano, mes,
                  tipo_recurso, relator, comarca, ementa, texto_completo, arquivo_origem
           FROM acordaos WHERE id = ?""",
        [doc_id],
    ).fetchone()

    if not row:
        return None

    return {
        "id": row["id"],
        "numero_processo": row["numero_processo"],
        "data_publicacao": row["data_publicacao"],
        "ano": row["ano"],
        "mes": row["mes"],
        "tipo_recurso": row["tipo_recurso"] or "",
        "relator": row["relator"] or "",
        "comarca": row["comarca"] or "",
        "ementa": row["ementa"] or "",
        "texto_completo": row["texto_completo"],
        "arquivo_origem": row["arquivo_origem"] or "",
    }


def get_stats() -> dict:
    """Retorna estatísticas do banco de jurisprudência."""
    conn = _get_conn()

    total = conn.execute("SELECT COUNT(*) FROM acordaos").fetchone()[0]

    anos = conn.execute(
        "SELECT ano, COUNT(*) as cnt FROM acordaos GROUP BY ano ORDER BY ano"
    ).fetchall()

    tipos = conn.execute(
        "SELECT tipo_recurso, COUNT(*) as cnt FROM acordaos "
        "WHERE tipo_recurso != '' GROUP BY tipo_recurso ORDER BY cnt DESC LIMIT 15"
    ).fetchall()

    return {
        "total": total,
        "por_ano": {row["ano"]: row["cnt"] for row in anos},
        "por_tipo": {row["tipo_recurso"]: row["cnt"] for row in tipos},
        "ano_min": min(row["ano"] for row in anos) if anos else 0,
        "ano_max": max(row["ano"] for row in anos) if anos else 0,
    }


# ── Semantic Search (Embedding-based) ─────────────────────────────────────────

import struct
import numpy as np

# In-memory cache for embeddings (lazy loaded)
_emb_cache = {
    "ids": None,        # np.ndarray of acordao IDs
    "matrix": None,     # np.ndarray (N, 1024) normalized
    "loaded": False,
}

# Azure OpenAI config for query embedding
_AZURE_EMBEDDING_ENDPOINT = os.environ.get(
    "AZURE_EMBEDDING_ENDPOINT",
    "https://assistente-web-resource.cognitiveservices.azure.com",
)
_AZURE_EMBEDDING_KEY = os.environ.get(
    "AZURE_EMBEDDING_KEY",
    os.environ.get("AZURE_AI_KEY", ""),
)
_EMBEDDING_MODEL = "text-embedding-3-large"
_EMBEDDING_DIM = 1024


def _load_embeddings():
    """Carrega todos os embeddings do SQLite para memória (uma vez)."""
    if _emb_cache["loaded"]:
        return _emb_cache["matrix"] is not None

    _emb_cache["loaded"] = True

    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT acordao_id, embedding FROM embeddings ORDER BY acordao_id"
        ).fetchall()

        if not rows:
            print("⚠️ Nenhum embedding encontrado no banco.")
            return False

        ids = np.array([r[0] for r in rows], dtype=np.int32)
        matrix = np.array(
            [list(struct.unpack(f"{_EMBEDDING_DIM}f", r[1])) for r in rows],
            dtype=np.float32,
        )

        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-8)

        _emb_cache["ids"] = ids
        _emb_cache["matrix"] = matrix
        print(f"📐 {len(ids):,} embeddings carregados ({matrix.nbytes / 1024 / 1024:.0f} MB)")
        return True

    except Exception as e:
        print(f"⚠️ Erro ao carregar embeddings: {e}")
        return False


def _embed_query(query: str) -> Optional[np.ndarray]:
    """Gera embedding da query do usuário via Azure OpenAI."""
    if not _AZURE_EMBEDDING_KEY:
        return None

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=_AZURE_EMBEDDING_KEY,
            azure_endpoint=_AZURE_EMBEDDING_ENDPOINT,
            api_version="2024-12-01-preview",
        )
        resp = client.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=query,
            dimensions=_EMBEDDING_DIM,
        )
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        emb /= np.linalg.norm(emb)
        return emb
    except Exception as e:
        print(f"⚠️ Erro ao gerar embedding da query: {e}")
        return None


def semantic_search(
    query: str,
    ano_inicio: int = 0,
    ano_fim: int = 9999,
    tipo_recurso: str = "",
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """
    Busca semântica nos acórdãos usando embeddings (text-embedding-3-large).

    Returns:
        {
            "results": [{"id", "numero_processo", "data_publicacao", "ano", "tipo_recurso",
                          "relator", "comarca", "ementa", "similarity"}],
            "total": int,
            "page": int,
            "page_size": int,
            "pages": int,
            "query": str,
            "mode": "semantic"
        }
    """
    # Load embeddings if not cached
    if not _load_embeddings():
        return {
            "results": [], "total": 0, "page": page, "page_size": page_size,
            "pages": 0, "query": query, "mode": "semantic",
            "error": "Embeddings não disponíveis. Execute vectorize_jurisprudencia.py",
        }

    # Embed the query
    q_emb = _embed_query(query)
    if q_emb is None:
        return {
            "results": [], "total": 0, "page": page, "page_size": page_size,
            "pages": 0, "query": query, "mode": "semantic",
            "error": "AZURE_EMBEDDING_KEY não configurada para busca semântica.",
        }

    # Compute cosine similarities
    scores = _emb_cache["matrix"] @ q_emb  # (N,)
    ids = _emb_cache["ids"]

    # Get full metadata to apply filters
    conn = _get_conn()

    # Sort by similarity (descending)
    sorted_indices = np.argsort(scores)[::-1]

    # We need to apply filters — fetch metadata for top candidates
    # To avoid fetching all 26K, we over-fetch the top 500 and filter
    top_n = min(500, len(sorted_indices))
    candidate_indices = sorted_indices[:top_n]
    candidate_ids = ids[candidate_indices].tolist()
    candidate_scores = scores[candidate_indices].tolist()

    # Build ID→score map
    id_score = {int(cid): cscore for cid, cscore in zip(candidate_ids, candidate_scores)}

    # Fetch metadata for candidates
    placeholders = ",".join("?" * len(candidate_ids))
    where_parts = [f"id IN ({placeholders})"]
    params = list(candidate_ids)

    if ano_inicio > 0:
        where_parts.append("ano >= ?")
        params.append(ano_inicio)
    if ano_fim < 9999:
        where_parts.append("ano <= ?")
        params.append(ano_fim)
    if tipo_recurso:
        where_parts.append("tipo_recurso LIKE ?")
        params.append(f"%{tipo_recurso}%")

    where_clause = " AND ".join(where_parts)

    rows = conn.execute(
        f"""SELECT id, numero_processo, data_publicacao, ano, mes,
                   tipo_recurso, relator, comarca, ementa
            FROM acordaos WHERE {where_clause}""",
        params,
    ).fetchall()

    # Sort by similarity score
    results_raw = []
    for row in rows:
        doc_id = row["id"]
        results_raw.append({
            "id": doc_id,
            "numero_processo": row["numero_processo"],
            "data_publicacao": row["data_publicacao"],
            "ano": row["ano"],
            "mes": row["mes"],
            "tipo_recurso": row["tipo_recurso"] or "",
            "relator": row["relator"] or "",
            "comarca": row["comarca"] or "",
            "ementa": row["ementa"] or "",
            "similarity": round(id_score.get(doc_id, 0), 4),
        })

    results_raw.sort(key=lambda x: x["similarity"], reverse=True)

    total = len(results_raw)
    pages = (total + page_size - 1) // page_size if total > 0 else 0
    offset = (max(1, page) - 1) * page_size
    page_results = results_raw[offset : offset + page_size]

    return {
        "results": page_results,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
        "query": query,
        "mode": "semantic",
    }


def reload_db():
    """Fecha a conexão atual e força reinicialização na próxima query."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None
    # Also reset embedding cache
    _emb_cache["ids"] = None
    _emb_cache["matrix"] = None
    _emb_cache["loaded"] = False


def is_available() -> bool:
    """Verifica se o banco de jurisprudência está disponível."""
    return os.path.exists(_DB_PATH)

