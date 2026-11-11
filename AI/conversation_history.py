"""SQLite persistence for per-browser chat history and latency measurements."""

from contextlib import contextmanager
import os
import re
import secrets
import sqlite3
import threading


MEMORY_CATEGORIES = {
    "ai_models": {"ai", "llm", "model", "models", "ollama", "qwen", "gpt", "claude"},
    "software": {"software", "code", "coding", "python", "package", "library", "api"},
    "hardware": {"hardware", "cpu", "gpu", "memory", "ram", "processor"},
    "web_research": {"web", "internet", "search", "source", "research", "browser"},
}
MEMORY_STOP_WORDS = {
    "a", "an", "and", "are", "about", "as", "at", "be", "by", "can", "create",
    "describe", "does", "for", "from", "give", "how", "i", "in", "include", "is",
    "it", "list", "me", "of", "on", "or", "please", "review", "show", "table",
    "tell", "that", "the", "this", "to", "update", "what", "with", "you",
}


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
                CREATE TABLE IF NOT EXISTS research_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    lesson TEXT NOT NULL,
                    memory_type TEXT NOT NULL CHECK (
                        memory_type IN ('correction', 'validation', 'review')
                    ),
                    rank INTEGER NOT NULL DEFAULT 0 CHECK (rank BETWEEN -20 AND 20),
                    correct_count INTEGER NOT NULL DEFAULT 0,
                    incorrect_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(category, memory_key)
                );
                CREATE INDEX IF NOT EXISTS research_memory_category_rank
                    ON research_memory(category, rank DESC, updated_at DESC);
                CREATE INDEX IF NOT EXISTS research_memory_updated
                    ON research_memory(updated_at DESC);
                CREATE TABLE IF NOT EXISTS research_memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL REFERENCES research_memory(id)
                        ON DELETE CASCADE,
                    outcome TEXT NOT NULL CHECK (
                        outcome IN ('incorrect', 'review', 'correct')
                    ),
                    feedback TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS research_memory_events_memory
                    ON research_memory_events(memory_id, id DESC);
                CREATE TABLE IF NOT EXISTS recent_memory_cache (
                    session_id TEXT NOT NULL,
                    memory_id INTEGER NOT NULL REFERENCES research_memory(id)
                        ON DELETE CASCADE,
                    category TEXT NOT NULL,
                    rank_snapshot INTEGER NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 1,
                    accessed_at TEXT NOT NULL,
                    PRIMARY KEY(session_id, memory_id)
                );
                CREATE INDEX IF NOT EXISTS recent_memory_cache_session_access
                    ON recent_memory_cache(session_id, accessed_at DESC);
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
            connection.execute(
                "DELETE FROM recent_memory_cache WHERE session_id = ?",
                (session_id,),
            )
            connection.commit()

    @staticmethod
    def _memory_terms(text):
        return {
            token
            for token in re.findall(
                r"[a-z]+|\d+(?:\.\d+)*[a-z]*",
                text.lower(),
            )
            if len(token) > 1 and token not in MEMORY_STOP_WORDS
        }

    @classmethod
    def _memory_category(cls, text):
        terms = cls._memory_terms(text)
        if terms.intersection(MEMORY_CATEGORIES["ai_models"]):
            return "ai_models"
        if terms.intersection(MEMORY_CATEGORIES["software"]):
            return "software"
        if terms.intersection(MEMORY_CATEGORIES["hardware"]):
            return "hardware"
        if terms.intersection(MEMORY_CATEGORIES["web_research"]):
            return "web_research"
        return "general"

    @classmethod
    def _memory_key(cls, topic):
        return " ".join(sorted(cls._memory_terms(topic))) or "general"

    def record_research_memory(
        self,
        session_id,
        topic,
        lesson,
        outcome,
        created_at,
        memory_type=None,
    ):
        """Store a deduplicated, ranked research lesson and its feedback event."""
        if outcome not in {"incorrect", "review", "correct"}:
            raise ValueError("Invalid memory outcome.")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("A memory topic is required.")
        if not isinstance(lesson, str) or not lesson.strip():
            raise ValueError("A memory lesson is required.")

        topic = topic.strip()[:500]
        lesson = lesson.strip()[:1600]
        category = self._memory_category(topic)
        memory_key = self._memory_key(topic)
        memory_type = memory_type or {
            "incorrect": "correction",
            "review": "review",
            "correct": "validation",
        }[outcome]
        if memory_type not in {"correction", "validation", "review"}:
            raise ValueError("Invalid memory type.")
        rank_delta = {"incorrect": -2, "review": 0, "correct": 2}[outcome]
        correct_delta = int(outcome == "correct")
        incorrect_delta = int(outcome == "incorrect")

        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_memory (
                    category, memory_key, topic, lesson, memory_type, rank,
                    correct_count, incorrect_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(category, memory_key) DO UPDATE SET
                    topic = excluded.topic,
                    lesson = CASE
                        WHEN excluded.memory_type = 'review'
                            AND research_memory.memory_type != 'review'
                        THEN research_memory.lesson
                        ELSE excluded.lesson
                    END,
                    memory_type = CASE
                        WHEN excluded.memory_type = 'review'
                            AND research_memory.memory_type != 'review'
                        THEN research_memory.memory_type
                        ELSE excluded.memory_type
                    END,
                    rank = MAX(-20, MIN(20, research_memory.rank + excluded.rank)),
                    correct_count = research_memory.correct_count + excluded.correct_count,
                    incorrect_count = research_memory.incorrect_count + excluded.incorrect_count,
                    updated_at = excluded.updated_at
                """,
                (
                    category,
                    memory_key,
                    topic,
                    lesson,
                    memory_type,
                    rank_delta,
                    correct_delta,
                    incorrect_delta,
                    created_at,
                    created_at,
                ),
            )
            row = connection.execute(
                """
                SELECT id, rank FROM research_memory
                WHERE category = ? AND memory_key = ?
                """,
                (category, memory_key),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO research_memory_events
                    (memory_id, outcome, feedback, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (row["id"], outcome, lesson, created_at),
            )
            if session_id:
                self._cache_memory(
                    connection,
                    session_id,
                    row["id"],
                    category,
                    row["rank"],
                    created_at,
                )
            connection.commit()
        return row["id"]

    @staticmethod
    def _cache_memory(connection, session_id, memory_id, category, rank, accessed_at):
        connection.execute(
            """
            INSERT INTO recent_memory_cache
                (session_id, memory_id, category, rank_snapshot, accessed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, memory_id) DO UPDATE SET
                rank_snapshot = excluded.rank_snapshot,
                access_count = recent_memory_cache.access_count + 1,
                accessed_at = excluded.accessed_at
            """,
            (session_id, memory_id, category, rank, accessed_at),
        )
        connection.execute(
            """
            DELETE FROM recent_memory_cache
            WHERE session_id = ? AND memory_id NOT IN (
                SELECT memory_id FROM recent_memory_cache
                WHERE session_id = ?
                ORDER BY accessed_at DESC LIMIT 32
            )
            """,
            (session_id, session_id),
        )

    def recall_research_memory(self, session_id, query, created_at, limit=5):
        """Recall a few category- and topic-matched lessons, ranking stronger feedback higher."""
        terms = self._memory_terms(query)
        if not terms:
            return []

        category = self._memory_category(query)
        categories = (category, "general") if category != "general" else ("general",)
        placeholders = ", ".join("?" for _ in categories)
        with self._lock, self._connect() as connection:
            cache_rows = connection.execute(
                f"""
                SELECT m.*
                FROM recent_memory_cache c
                JOIN research_memory m ON m.id = c.memory_id
                WHERE c.session_id = ? AND c.category IN ({placeholders})
                ORDER BY c.accessed_at DESC
                LIMIT 32
                """,
                (session_id, *categories),
            ).fetchall()
            candidate_rows = connection.execute(
                f"""
                SELECT * FROM research_memory
                WHERE category IN ({placeholders})
                ORDER BY category, rank DESC, updated_at DESC
                LIMIT 200
                """,
                categories,
            ).fetchall()

            candidates = {row["id"]: row for row in [*cache_rows, *candidate_rows]}
            ranked = []
            for row in candidates.values():
                overlap = len(terms.intersection(self._memory_terms(row["topic"])))
                if overlap:
                    score = overlap * 5 + row["rank"]
                    ranked.append((score, row))
            ranked.sort(
                key=lambda item: (
                    item[0],
                    item[1]["rank"],
                    item[1]["updated_at"],
                ),
                reverse=True,
            )
            selected = [row for _, row in ranked[:limit]]
            for row in selected:
                self._cache_memory(
                    connection,
                    session_id,
                    row["id"],
                    row["category"],
                    row["rank"],
                    created_at,
                )
            connection.commit()
        return [dict(row) for row in selected]

    def get_research_memory(self, limit=100):
        """Return ranked lessons and counts for the memory review screen."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, category, topic, lesson, memory_type, rank,
                       correct_count, incorrect_count,
                       (SELECT COUNT(*) FROM research_memory_events e
                        WHERE e.memory_id = research_memory.id AND e.outcome = 'review')
                           AS review_count,
                       updated_at
                FROM research_memory
                ORDER BY rank DESC, updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            memory_ids = [row["id"] for row in rows]
            events_by_memory = {memory_id: [] for memory_id in memory_ids}
            if memory_ids:
                placeholders = ", ".join("?" for _ in memory_ids)
                events = connection.execute(
                    f"""
                    SELECT memory_id, outcome, feedback, created_at FROM (
                        SELECT memory_id, outcome, feedback, created_at,
                               ROW_NUMBER() OVER (
                                   PARTITION BY memory_id ORDER BY id DESC
                               ) AS event_rank
                        FROM research_memory_events
                        WHERE memory_id IN ({placeholders})
                    )
                    WHERE event_rank <= 5
                    ORDER BY memory_id, event_rank
                    """,
                    memory_ids,
                ).fetchall()
                for event in events:
                    event_list = events_by_memory[event["memory_id"]]
                    event_list.append(dict(event))
        return [
            {**dict(row), "events": events_by_memory[row["id"]]}
            for row in rows
        ]
