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
            "SELECT content, enabled FROM user_memories WHERE user_key = ?",
            (user_key,)
        )
        row = cur.fetchone()
        if row:
            return {"content": row["content"], "enabled": bool(row["enabled"])}
        return {"content": "", "enabled": True}

def save_memory(user_key: str = "default", content: str = "", enabled: bool = True):
    """Save or update user memory/preferences. Content is truncated to MAX_MEMORY_CHARS."""
    content = content[:MAX_MEMORY_CHARS]
    with get_db() as conn:
        conn.execute(
            """INSERT INTO user_memories (user_key, content, enabled, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(user_key) DO UPDATE SET
                 content = excluded.content,
                 enabled = excluded.enabled,
                 updated_at = CURRENT_TIMESTAMP""",
            (user_key, content, int(enabled))
        )
        conn.commit()
    return {"content": content, "enabled": enabled}


_db_initialized = False

def ensure_db():
    """Lazy initializer — call once at server startup instead of on every import."""
    global _db_initialized
    if not _db_initialized:
        init_db()
        _db_initialized = True

# Initialize on import (safe to call multiple times due to IF NOT EXISTS)
ensure_db()
