"""common/history.py — 대화 기록을 로컬 SQLite에 저장하고 검색한다."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    domain       TEXT NOT NULL,
    title        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    domain       TEXT NOT NULL,
    role         TEXT NOT NULL,        -- 'user' | 'assistant'
    content      TEXT NOT NULL,
    detected_scope TEXT,
    total_sec    REAL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_domain ON messages(domain);
"""


@contextmanager
def _conn():
    Path(config.HISTORY_DB).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(config.HISTORY_DB))
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(_SCHEMA)
        yield connection
        connection.commit()
    finally:
        connection.close()


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def ensure_session(session_id: str, domain: str, title_hint: str) -> None:
    """세션이 없으면 만든다. 있으면 updated_at만 갱신."""
    now = datetime.now().isoformat(timespec="seconds")
    title = (title_hint or "새 대화")[:40]
    with _conn() as c:
        row = c.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            c.execute(
                "INSERT INTO sessions (session_id, domain, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, domain, title, now, now),
            )
        else:
            c.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )


def add_message(
    session_id: str,
    domain: str,
    role: str,
    content: str,
    detected_scope: Optional[str] = None,
    total_sec: Optional[float] = None,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    with _conn() as c:
        c.execute(
            "INSERT INTO messages "
            "(session_id, domain, role, content, detected_scope, total_sec, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, domain, role, content, detected_scope, total_sec, now),
        )
        c.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )


def list_sessions(domain: str, limit: int = 30) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT session_id, title, updated_at FROM sessions "
            "WHERE domain = ? ORDER BY updated_at DESC LIMIT ?",
            (domain, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_messages(session_id: str) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content, detected_scope, total_sec, created_at "
            "FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def search_messages(domain: str, query: str, limit: int = 20) -> list[dict]:
    """대화 기록 검색. 세션 제목 + 메시지 본문을 LIKE로 검색한다."""
    query = (query or "").strip()
    if not query:
        return []
    like = f"%{query}%"
    with _conn() as c:
        rows = c.execute(
            "SELECT m.session_id, s.title, m.role, m.content, m.created_at "
            "FROM messages m JOIN sessions s ON s.session_id = m.session_id "
            "WHERE m.domain = ? AND (m.content LIKE ? OR s.title LIKE ?) "
            "ORDER BY m.created_at DESC LIMIT ?",
            (domain, like, like, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def rename_session_if_untitled(session_id: str, first_question: str) -> None:
    """세션 제목이 기본값이면 첫 질문으로 자동 갱신 (목록에서 알아보기 쉽게)."""
    with _conn() as c:
        row = c.execute(
            "SELECT title FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row and row["title"] == "새 대화":
            c.execute(
                "UPDATE sessions SET title = ? WHERE session_id = ?",
                (first_question[:40], session_id),
            )
