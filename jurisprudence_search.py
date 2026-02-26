"""
Jurisprudence Search Engine — SQLite FTS5 based.

Provides search, document retrieval, and statistics for TJMG case law.
"""

import os
import re
import sqlite3
from typing import Optional


_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "jurisprudencia.db")

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


def is_available() -> bool:
    """Verifica se o banco de jurisprudência está disponível."""
    return os.path.exists(_DB_PATH)
