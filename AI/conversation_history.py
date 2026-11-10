"""SQLite persistence for per-browser chat history and latency measurements."""

from contextlib import contextmanager
import os
import secrets
import sqlite3
import threading


class ConversationHistory:
    """Store conversations in SQLite, keyed by a signed Flask session ID."""

    def __init__(self, database_path: str):
        self.database_path = os.path.abspath(database_path)
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS app_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    duration_s REAL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversation_messages_session_id
                    ON conversation_messages(session_id, id);
                CREATE TABLE IF NOT EXISTS conversation_latency (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    duration_s REAL NOT NULL,
                    context_messages INTEGER NOT NULL,
                    context_chars INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS conversation_latency_session_id
                    ON conversation_latency(session_id, id);
                """
            )

    def get_secret_key(self):
        """Return a persistent Flask cookie-signing key stored in this database."""
        with self._lock, self._connect() as connection:
            secret_key = secrets.token_urlsafe(48)
            connection.execute(
                "INSERT OR IGNORE INTO app_config (key, value) VALUES ('flask_secret_key', ?)",
                (secret_key,),
            )
            connection.commit()
            row = connection.execute(
                "SELECT value FROM app_config WHERE key = 'flask_secret_key'"
            ).fetchone()
            return row["value"]

    def get_messages(self, session_id):
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, duration_s, created_at
                FROM conversation_messages
                WHERE session_id = ?
                ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                **(
                    {"duration_s": row["duration_s"]}
                    if row["duration_s"] is not None
                    else {}
                ),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def append_turn(
        self,
        session_id,
        user_content,
        assistant_content,
        duration_s,
        context_messages,
        context_chars,
        created_at,
    ):
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO conversation_messages
                    (session_id, role, content, duration_s, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (session_id, "user", user_content, None, created_at),
                    (
                        session_id,
                        "assistant",
                        assistant_content,
                        duration_s,
                        created_at,
                    ),
                ],
            )
            connection.execute(
                """
                INSERT INTO conversation_latency
                    (session_id, duration_s, context_messages, context_chars, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    duration_s,
                    context_messages,
                    context_chars,
                    created_at,
                ),
            )
            connection.commit()

    def clear(self, session_id):
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM conversation_messages WHERE session_id = ?",
                (session_id,),
            )
            connection.execute(
                "DELETE FROM conversation_latency WHERE session_id = ?",
                (session_id,),
            )
            connection.commit()
