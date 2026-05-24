import sqlite3
import json
import os
from contextlib import contextmanager

DB_PATH = os.getenv("HISTORY_DB_PATH", "history.db")

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_email TEXT NOT NULL,
                title TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key TEXT NOT NULL DEFAULT 'default',
                content TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auto_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_key TEXT NOT NULL,
                content TEXT NOT NULL,
                source_conversation_id TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_flows (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                description TEXT DEFAULT '',
                color TEXT DEFAULT '#3b82f6',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flow_shares (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_id TEXT NOT NULL,
                shared_with_email TEXT NOT NULL,
                shared_by_user_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(flow_id, shared_with_email)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS flow_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_id TEXT NOT NULL,
                version_num INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                name TEXT,
                description TEXT,
                color TEXT,
                label TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(flow_id, version_num)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flow_versions_flow ON flow_versions(flow_id)")
        try:
            conn.execute("ALTER TABLE agent_flows ADD COLUMN description TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE agent_flows ADD COLUMN color TEXT DEFAULT '#3b82f6'")
        except Exception:
            pass
        # Migrate: add response_style columns if they don't exist yet
        try:
            conn.execute("ALTER TABLE user_memories ADD COLUMN response_style TEXT DEFAULT 'default'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE user_memories ADD COLUMN response_style_text TEXT DEFAULT ''")
        except Exception:
            pass
        conn.commit()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_conversations(user_email: str):
    """Retrieve all conversations for a specific user."""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, title, updated_at FROM conversations WHERE user_email = ? ORDER BY updated_at DESC", 
            (user_email,)
        )
        return [{"id": row["id"], "title": row["title"], "updated_at": row["updated_at"]} for row in cur.fetchall()]

def get_messages(conversation_id: str, user_email: str):
    """Retrieve messages for a specific conversation of a user."""
    with get_db() as conn:
        # First verify ownership
        cur = conn.execute("SELECT id FROM conversations WHERE id = ? AND user_email = ?", (conversation_id, user_email))
        if not cur.fetchone():
            return None # Not found or unauthorized
            
        cur = conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC", 
            (conversation_id,)
        )
        return [{"role": row["role"], "content": row["content"]} for row in cur.fetchall()]

def save_message(conversation_id: str, user_email: str, role: str, content: str, title: str = "Nova Conversa"):
    """Save a message payload to a conversation, creating it if it doesn't exist."""
    with get_db() as conn:
        cur = conn.execute("SELECT user_email FROM conversations WHERE id = ?", (conversation_id,))
        row = cur.fetchone()
        if row is not None and row["user_email"] != user_email:
            # Conversation existe mas pertence a outro tenant — recusa para não anexar
            # mensagens cross-tenant via INSERT OR IGNORE silencioso.
            raise PermissionError(f"Conversation {conversation_id} não pertence a {user_email}")
        if row is None:
            conn.execute(
                "INSERT INTO conversations (id, user_email, title) VALUES (?, ?, ?)",
                (conversation_id, user_email, title)
            )
        else:
            conn.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_email = ?",
                (conversation_id, user_email)
            )

        conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content)
        )
        conn.commit()


# ── User Memory (Preferences) ────────────────────────────────────────────────

MAX_MEMORY_CHARS = 2000

def get_memory(user_key: str = "default") -> dict:
    """Retrieve user memory/preferences."""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT content, enabled, response_style, response_style_text FROM user_memories WHERE user_key = ?",
            (user_key,)
        )
        row = cur.fetchone()
        if row:
            return {
                "content": row["content"],
                "enabled": bool(row["enabled"]),
                "response_style": row["response_style"] or "default",
                "response_style_text": row["response_style_text"] or "",
            }
        return {"content": "", "enabled": True, "response_style": "default", "response_style_text": ""}

def save_memory(user_key: str = "default", content: str = "", enabled: bool = True,
                response_style: str = "default", response_style_text: str = ""):
    """Save or update user memory/preferences. Content is truncated to MAX_MEMORY_CHARS."""
    content = content[:MAX_MEMORY_CHARS]
    with get_db() as conn:
        conn.execute(
            """INSERT INTO user_memories (user_key, content, enabled, response_style, response_style_text, updated_at)
               VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_key) DO UPDATE SET
                 content = excluded.content,
                 enabled = excluded.enabled,
                 response_style = excluded.response_style,
                 response_style_text = excluded.response_style_text,
                 updated_at = CURRENT_TIMESTAMP""",
            (user_key, content, int(enabled), response_style, response_style_text)
        )
        conn.commit()
    return {"content": content, "enabled": enabled, "response_style": response_style,
            "response_style_text": response_style_text}


# ── Auto-Memories (extracted from conversations) ──────────────────────────────

def list_auto_memories(user_key: str) -> list:
    """List all auto-extracted memories for a user."""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, content, source_conversation_id, enabled, created_at FROM auto_memories WHERE user_key = ? ORDER BY created_at DESC",
            (user_key,)
        )
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "source_conversation_id": row["source_conversation_id"],
                "enabled": bool(row["enabled"]),
                "created_at": row["created_at"],
            }
            for row in cur.fetchall()
        ]

def add_auto_memory(user_key: str, content: str, source_conversation_id: str = None) -> dict:
    """Add a new auto-extracted memory."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO auto_memories (user_key, content, source_conversation_id) VALUES (?, ?, ?)",
            (user_key, content[:500], source_conversation_id)
        )
        conn.commit()
        return {"id": cur.lastrowid, "content": content, "enabled": True}

def update_auto_memory(memory_id: int, user_key: str, content: str = None, enabled: bool = None) -> bool:
    """Update content and/or enabled state of an auto-memory."""
    with get_db() as conn:
        if content is not None and enabled is not None:
            conn.execute(
                "UPDATE auto_memories SET content = ?, enabled = ? WHERE id = ? AND user_key = ?",
                (content[:500], int(enabled), memory_id, user_key)
            )
        elif content is not None:
            conn.execute(
                "UPDATE auto_memories SET content = ? WHERE id = ? AND user_key = ?",
                (content[:500], memory_id, user_key)
            )
        elif enabled is not None:
            conn.execute(
                "UPDATE auto_memories SET enabled = ? WHERE id = ? AND user_key = ?",
                (int(enabled), memory_id, user_key)
            )
        conn.commit()
    return True

def delete_auto_memory(memory_id: int, user_key: str) -> bool:
    """Delete an auto-extracted memory."""
    with get_db() as conn:
        conn.execute(
            "DELETE FROM auto_memories WHERE id = ? AND user_key = ?",
            (memory_id, user_key)
        )
        conn.commit()
    return True


# ── Agent Flows ───────────────────────────────────────────────────────────────

def create_flow(user_id: str, flow_id: str, name: str, config: dict, description: str = "", color: str = "#3b82f6") -> dict:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO agent_flows (id, user_id, name, config_json, description, color) VALUES (?, ?, ?, ?, ?, ?)",
            (flow_id, user_id, name, json.dumps(config, ensure_ascii=False), description, color)
        )
        conn.commit()
    return {"id": flow_id, "user_id": user_id, "name": name, "config": config, "description": description, "color": color}


def get_flow(flow_id: str, user_id: str) -> dict | None:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, user_id, name, config_json, description, color, created_at, updated_at FROM agent_flows WHERE id = ? AND user_id = ?",
            (flow_id, user_id)
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "name": row["name"],
            "config": json.loads(row["config_json"]),
            "description": row["description"] or "",
            "color": row["color"] or "#3b82f6",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def list_flows(user_id: str) -> list:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, name, description, color, created_at, updated_at FROM agent_flows WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        return [{
            "id": r["id"], "name": r["name"],
            "description": r["description"] or "",
            "color": r["color"] or "#3b82f6",
            "created_at": r["created_at"], "updated_at": r["updated_at"]
        } for r in cur.fetchall()]


def update_flow(flow_id: str, user_id: str, name: str, config: dict, description: str = "", color: str = "#3b82f6") -> bool:
    """Salva a versão ATUAL em flow_versions antes de sobrescrever com a nova."""
    with get_db() as conn:
        # snapshot da versão atual
        cur = conn.execute(
            "SELECT name, description, color, config_json FROM agent_flows WHERE id = ? AND user_id = ?",
            (flow_id, user_id),
        )
        row = cur.fetchone()
        if row:
            # próximo version_num
            cur2 = conn.execute(
                "SELECT COALESCE(MAX(version_num), 0) + 1 AS next FROM flow_versions WHERE flow_id = ?",
                (flow_id,),
            )
            next_num = cur2.fetchone()["next"]
            conn.execute(
                """INSERT INTO flow_versions (flow_id, version_num, config_json, name, description, color)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (flow_id, next_num, row["config_json"], row["name"], row["description"], row["color"]),
            )

        # atualiza a versão "live"
        conn.execute(
            "UPDATE agent_flows SET name = ?, config_json = ?, description = ?, color = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?",
            (name, json.dumps(config, ensure_ascii=False), description, color, flow_id, user_id)
        )
        conn.commit()
    return True


def list_flow_versions(flow_id: str, user_id: str) -> list:
    """Lista todas as versões salvas de um fluxo (somente se for o owner)."""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT user_id FROM agent_flows WHERE id = ?",
            (flow_id,),
        )
        row = cur.fetchone()
        if not row or row["user_id"] != user_id:
            return []
        cur = conn.execute(
            """SELECT version_num, name, description, color, label, created_at
               FROM flow_versions WHERE flow_id = ?
               ORDER BY version_num DESC""",
            (flow_id,),
        )
        return [{
            "version_num": r["version_num"],
            "name": r["name"],
            "description": r["description"] or "",
            "color": r["color"] or "#3b82f6",
            "label": r["label"] or "",
            "created_at": r["created_at"],
        } for r in cur.fetchall()]


def restore_flow_version(flow_id: str, user_id: str, version_num: int) -> bool:
    """Restaura uma versão antiga como atual (salva a atual como nova versão antes)."""
    with get_db() as conn:
        cur = conn.execute(
            """SELECT v.config_json, v.name, v.description, v.color
               FROM flow_versions v
               JOIN agent_flows f ON f.id = v.flow_id
               WHERE v.flow_id = ? AND v.version_num = ? AND f.user_id = ?""",
            (flow_id, version_num, user_id),
        )
        row = cur.fetchone()
        if not row:
            return False
        config = json.loads(row["config_json"])
        # update_flow já cuida do snapshot da atual
    update_flow(flow_id, user_id, row["name"], config, row["description"] or "", row["color"] or "#3b82f6")
    return True


def delete_flow(flow_id: str, user_id: str) -> bool:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM agent_flows WHERE id = ? AND user_id = ?",
            (flow_id, user_id)
        )
        conn.execute(
            "DELETE FROM flow_shares WHERE flow_id = ? AND shared_by_user_id = ?",
            (flow_id, user_id)
        )
        conn.execute(
            "DELETE FROM flow_versions WHERE flow_id = ?",
            (flow_id,)
        )
        conn.commit()
    return True


def share_flow(flow_id: str, owner_user_id: str, email: str) -> bool:
    """Compartilha um fluxo com outro usuário via email."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO flow_shares (flow_id, shared_with_email, shared_by_user_id) VALUES (?, ?, ?)",
            (flow_id, email.strip().lower(), owner_user_id),
        )
        conn.commit()
    return True


def list_shared_flows_for_email(email: str) -> list:
    """Retorna fluxos compartilhados com o email recebido (e ainda existentes)."""
    if not email:
        return []
    e = email.strip().lower()
    with get_db() as conn:
        cur = conn.execute(
            """
            SELECT f.id, f.name, f.description, f.color, f.created_at, f.updated_at, s.shared_by_user_id
            FROM flow_shares s
            JOIN agent_flows f ON f.id = s.flow_id
            WHERE s.shared_with_email = ?
            ORDER BY f.updated_at DESC
            """,
            (e,),
        )
        return [{
            "id": r["id"], "name": r["name"],
            "description": r["description"] or "",
            "color": r["color"] or "#3b82f6",
            "created_at": r["created_at"], "updated_at": r["updated_at"],
            "shared_by_user_id": r["shared_by_user_id"],
            "shared": True,
        } for r in cur.fetchall()]


def get_flow_shared(flow_id: str, email: str) -> dict | None:
    """Busca um fluxo se foi compartilhado com o email do solicitante."""
    if not email:
        return None
    e = email.strip().lower()
    with get_db() as conn:
        cur = conn.execute(
            """
            SELECT f.id, f.user_id, f.name, f.description, f.color, f.config_json, f.created_at, f.updated_at
            FROM flow_shares s
            JOIN agent_flows f ON f.id = s.flow_id
            WHERE s.flow_id = ? AND s.shared_with_email = ?
            """,
            (flow_id, e),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row["id"], "user_id": row["user_id"], "name": row["name"],
            "description": row["description"] or "",
            "color": row["color"] or "#3b82f6",
            "config": json.loads(row["config_json"]),
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }


_db_initialized = False

def ensure_db():
    """Lazy initializer — call once at server startup instead of on every import."""
    global _db_initialized
    if not _db_initialized:
        init_db()
        _db_initialized = True

# Initialize on import (safe to call multiple times due to IF NOT EXISTS)
ensure_db()
