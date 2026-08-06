from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "kelsie.db")


# ============================================================
# CONNECTION AND DATETIME HELPERS
# ============================================================


def utc_now_datetime() -> datetime:
    return datetime.now(timezone.utc)


def to_utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def utc_now() -> str:
    return to_utc_iso(utc_now_datetime())


def parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()

    if not text:
        raise ValueError("A date and time are required.")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError("The date and time are not valid.") from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


@contextmanager
def get_connection() -> Iterable[sqlite3.Connection]:
    os.makedirs(DATA_DIR, exist_ok=True)

    connection = sqlite3.connect(
        DB_PATH,
        timeout=20,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> List[str]:
    if not _table_exists(connection, table_name):
        return []

    rows = connection.execute(
        "PRAGMA table_info({})".format(table_name)
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _first(
    available: List[str],
    candidates: List[str],
) -> Optional[str]:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    if column_name in _columns(connection, table_name):
        return

    connection.execute(
        "ALTER TABLE {} ADD COLUMN {} {}".format(
            table_name,
            column_name,
            definition,
        )
    )


def _json_dict(value: Any) -> Dict[str, Any]:
    if not value:
        return {}

    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "active",
    }


def _safe_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# ============================================================
# DATABASE INITIALIZATION AND LEGACY MIGRATION
# ============================================================


def init_db() -> None:
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles_v2 (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                legacy_id TEXT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New conversation',
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                legacy_id TEXT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (
                    role IN ('system', 'user', 'assistant')
                ),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations_v2(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                scheduled_for TEXT NOT NULL,
                alert_at TEXT NOT NULL,
                alert_offset_minutes INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'upcoming'
                    CHECK (status IN ('upcoming', 'completed')),
                completed_at TEXT,
                hidden_until TEXT,
                notified_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS open_loops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                normalized_action TEXT NOT NULL,
                person TEXT,
                project TEXT,
                timing_text TEXT,
                scheduled_for TEXT,
                status TEXT NOT NULL DEFAULT 'open'
                    CHECK (
                        status IN ('open', 'completed', 'dismissed')
                    ),
                confidence REAL NOT NULL DEFAULT 0.8,
                source_conversation_id INTEGER,
                source_message_id INTEGER,
                completed_at TEXT,
                dismissed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );


            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT PRIMARY KEY,
                memory_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id INTEGER PRIMARY KEY,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations_v2(id)
                    ON DELETE CASCADE
            );
            """
        )

        # Older copies of the v2 tables may not contain these columns.
        _ensure_column(connection, "conversations_v2", "legacy_id", "TEXT")
        _ensure_column(connection, "messages_v2", "legacy_id", "TEXT")

        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS
                idx_conversations_v2_user_updated
            ON conversations_v2 (user_id, updated_at DESC);

            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_conversations_v2_one_active
            ON conversations_v2 (user_id)
            WHERE is_active = 1;

            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_conversations_v2_legacy_id
            ON conversations_v2 (legacy_id)
            WHERE legacy_id IS NOT NULL;

            CREATE INDEX IF NOT EXISTS
                idx_messages_v2_conversation_created
            ON messages_v2 (conversation_id, created_at ASC, id ASC);

            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_messages_v2_legacy_id
            ON messages_v2 (legacy_id)
            WHERE legacy_id IS NOT NULL;

            CREATE INDEX IF NOT EXISTS
                idx_reminders_user_status_schedule
            ON reminders (user_id, status, scheduled_for ASC);

            CREATE INDEX IF NOT EXISTS
                idx_reminders_user_alert
            ON reminders (user_id, status, alert_at ASC);


            CREATE INDEX IF NOT EXISTS
                idx_open_loops_user_status_updated
            ON open_loops (user_id, status, updated_at DESC);

            CREATE INDEX IF NOT EXISTS
                idx_open_loops_user_action
            ON open_loops (user_id, normalized_action, status);


            CREATE INDEX IF NOT EXISTS
                idx_conversation_summaries_user_updated
            ON conversation_summaries (user_id, updated_at DESC);
            """
        )

        _migrate_profiles(connection)
        _migrate_conversations_and_messages(connection)


def _normalize_profile(
    profile_data: Dict[str, Any],
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    normalized = dict(profile_data)
    resolved_user_id = (
        user_id
        or normalized.get("user_id")
        or normalized.get("id")
    )

    if not resolved_user_id:
        raise ValueError("A profile requires a user_id.")

    normalized["user_id"] = str(resolved_user_id)
    normalized["id"] = str(resolved_user_id)
    normalized.setdefault("name", "")
    normalized.setdefault("mode", "both")
    normalized.setdefault("timezone", "America/Toronto")
    normalized.setdefault("daily_overview_enabled", True)
    normalized.setdefault("quiet_hours_start", None)
    normalized.setdefault("quiet_hours_end", None)
    normalized.setdefault("proactivity", "balanced")
    normalized.setdefault("proactivity_level", normalized.get("proactivity", "balanced"))
    normalized.setdefault("memory_enabled", True)
    normalized.setdefault("adaptive_tone", True)
    return normalized


def _migrate_profiles(connection: sqlite3.Connection) -> None:
    now = utc_now()

    for table_name in ("profiles", "users"):
        if not _table_exists(connection, table_name):
            continue

        columns = _columns(connection, table_name)
        user_id_column = _first(columns, ["user_id", "id", "profile_id"])

        if not user_id_column:
            continue

        json_column = _first(
            columns,
            ["profile_json", "profile_data", "data"],
        )
        created_column = _first(columns, ["created_at", "created"])
        updated_column = _first(columns, ["updated_at", "updated"])

        rows = connection.execute(
            "SELECT * FROM {}".format(table_name)
        ).fetchall()

        for row in rows:
            raw_user_id = row[user_id_column]
            if raw_user_id is None or not str(raw_user_id).strip():
                continue

            user_id = str(raw_user_id)
            existing = connection.execute(
                "SELECT 1 FROM profiles_v2 WHERE user_id = ? LIMIT 1",
                (user_id,),
            ).fetchone()

            if existing:
                continue

            profile_data = _json_dict(row[json_column]) if json_column else {}

            for column in columns:
                if column in {json_column, created_column, updated_column}:
                    continue
                if row[column] is not None:
                    profile_data[column] = row[column]

            normalized = _normalize_profile(profile_data, user_id)
            created_at = (
                str(row[created_column])
                if created_column and row[created_column]
                else now
            )
            updated_at = (
                str(row[updated_column])
                if updated_column and row[updated_column]
                else created_at
            )

            connection.execute(
                """
                INSERT OR IGNORE INTO profiles_v2 (
                    user_id,
                    profile_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    json.dumps(normalized),
                    created_at,
                    updated_at,
                ),
            )


def _migrate_conversations_and_messages(
    connection: sqlite3.Connection,
) -> None:
    if not _table_exists(connection, "conversations"):
        return

    columns = _columns(connection, "conversations")
    id_column = _first(columns, ["id", "conversation_id"])
    user_id_column = _first(columns, ["user_id", "profile_id"])

    if not id_column or not user_id_column:
        return

    title_column = _first(columns, ["title", "name"])
    active_column = _first(columns, ["is_active", "active"])
    created_column = _first(columns, ["created_at", "created"])
    updated_column = _first(
        columns,
        ["updated_at", "updated", "last_message_at"],
    )

    active_rows = connection.execute(
        "SELECT user_id FROM conversations_v2 WHERE is_active = 1"
    ).fetchall()
    active_users = {str(row["user_id"]) for row in active_rows}
    conversation_map: Dict[str, int] = {}
    now = utc_now()

    legacy_rows = connection.execute(
        "SELECT * FROM conversations"
    ).fetchall()

    for row in legacy_rows:
        raw_legacy_id = row[id_column]
        raw_user_id = row[user_id_column]

        if raw_legacy_id is None or raw_user_id is None:
            continue

        legacy_id = str(raw_legacy_id)
        user_id = str(raw_user_id)

        existing = connection.execute(
            """
            SELECT id
            FROM conversations_v2
            WHERE legacy_id = ?
            LIMIT 1
            """,
            (legacy_id,),
        ).fetchone()

        if existing:
            conversation_map[legacy_id] = int(existing["id"])
            continue

        requested_active = (
            _truthy(row[active_column]) if active_column else False
        )
        is_active = 0

        if requested_active and user_id not in active_users:
            is_active = 1
            active_users.add(user_id)

        created_at = (
            str(row[created_column])
            if created_column and row[created_column]
            else now
        )
        updated_at = (
            str(row[updated_column])
            if updated_column and row[updated_column]
            else created_at
        )
        title = (
            str(row[title_column]).strip()
            if title_column and row[title_column]
            else "New conversation"
        )

        cursor = connection.execute(
            """
            INSERT INTO conversations_v2 (
                legacy_id,
                user_id,
                title,
                is_active,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                legacy_id,
                user_id,
                title or "New conversation",
                is_active,
                created_at,
                updated_at,
            ),
        )
        conversation_map[legacy_id] = int(cursor.lastrowid)

    if _table_exists(connection, "messages"):
        _migrate_messages(connection, conversation_map)

    untitled_rows = connection.execute(
        """
        SELECT id
        FROM conversations_v2
        WHERE title IS NULL
           OR TRIM(title) = ''
           OR title = 'New conversation'
        """
    ).fetchall()

    for row in untitled_rows:
        conversation_id = int(row["id"])
        first_user_message = connection.execute(
            """
            SELECT content
            FROM messages_v2
            WHERE conversation_id = ?
              AND role = 'user'
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()

        if first_user_message:
            connection.execute(
                "UPDATE conversations_v2 SET title = ? WHERE id = ?",
                (
                    _derive_title(str(first_user_message["content"])),
                    conversation_id,
                ),
            )

    user_rows = connection.execute(
        "SELECT DISTINCT user_id FROM conversations_v2"
    ).fetchall()

    for row in user_rows:
        user_id = str(row["user_id"])
        active = connection.execute(
            """
            SELECT 1
            FROM conversations_v2
            WHERE user_id = ? AND is_active = 1
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

        if active:
            continue

        latest = connection.execute(
            """
            SELECT id
            FROM conversations_v2
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

        if latest:
            connection.execute(
                "UPDATE conversations_v2 SET is_active = 1 WHERE id = ?",
                (int(latest["id"]),),
            )


def _migrate_messages(
    connection: sqlite3.Connection,
    conversation_map: Dict[str, int],
) -> None:
    columns = _columns(connection, "messages")
    message_id_column = _first(columns, ["id", "message_id"])
    conversation_id_column = _first(
        columns,
        ["conversation_id", "chat_id"],
    )
    role_column = _first(columns, ["role", "sender"])
    content_column = _first(columns, ["content", "message", "text"])
    created_column = _first(columns, ["created_at", "created"])

    if not conversation_id_column or not content_column:
        return

    now = utc_now()
    rows = connection.execute("SELECT * FROM messages").fetchall()

    for row in rows:
        raw_conversation_id = row[conversation_id_column]
        content = row[content_column]

        if raw_conversation_id is None or content is None:
            continue

        old_key = str(raw_conversation_id)
        conversation_id = conversation_map.get(old_key)

        if conversation_id is None:
            mapped = connection.execute(
                """
                SELECT id
                FROM conversations_v2
                WHERE legacy_id = ?
                LIMIT 1
                """,
                (old_key,),
            ).fetchone()
            if mapped:
                conversation_id = int(mapped["id"])

        if conversation_id is None:
            numeric_id = _safe_int(raw_conversation_id)
            if numeric_id is not None:
                numeric_match = connection.execute(
                    "SELECT 1 FROM conversations_v2 WHERE id = ? LIMIT 1",
                    (numeric_id,),
                ).fetchone()
                if numeric_match:
                    conversation_id = numeric_id

        if conversation_id is None:
            continue

        raw_role = (
            str(row[role_column]).strip().lower()
            if role_column and row[role_column] is not None
            else "assistant"
        )

        if raw_role in {"user", "human", "me"}:
            role = "user"
        elif raw_role == "system":
            role = "system"
        else:
            role = "assistant"

        created_at = (
            str(row[created_column])
            if created_column and row[created_column]
            else now
        )
        legacy_message_id = (
            str(row[message_id_column])
            if message_id_column and row[message_id_column] is not None
            else None
        )

        if legacy_message_id:
            existing = connection.execute(
                "SELECT 1 FROM messages_v2 WHERE legacy_id = ? LIMIT 1",
                (legacy_message_id,),
            ).fetchone()
            if existing:
                continue

        connection.execute(
            """
            INSERT INTO messages_v2 (
                legacy_id,
                conversation_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                legacy_message_id,
                int(conversation_id),
                role,
                str(content),
                created_at,
            ),
        )

        connection.execute(
            """
            UPDATE conversations_v2
            SET updated_at = CASE
                WHEN updated_at < ? THEN ?
                ELSE updated_at
            END
            WHERE id = ?
            """,
            (created_at, created_at, int(conversation_id)),
        )


# ============================================================
# PROFILE FUNCTIONS
# ============================================================


def create_profile(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _normalize_profile(profile_data)
    user_id = str(normalized["user_id"])
    now = utc_now()

    with get_connection() as connection:
        existing = connection.execute(
            "SELECT created_at FROM profiles_v2 WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing else now

        connection.execute(
            """
            INSERT INTO profiles_v2 (
                user_id,
                profile_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                json.dumps(normalized),
                created_at,
                now,
            ),
        )

    return get_profile(user_id) or normalized


def update_profile(
    user_id: str,
    profile_data: Dict[str, Any],
) -> Dict[str, Any]:
    current = get_profile(user_id) or {
        "user_id": user_id,
        "id": user_id,
    }
    current.update(profile_data)
    return create_profile(_normalize_profile(current, user_id))


def get_profile(user_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT profile_json
            FROM profiles_v2
            WHERE user_id = ?
            LIMIT 1
            """,
            (str(user_id),),
        ).fetchone()

    if not row:
        return None

    return _normalize_profile(
        _json_dict(row["profile_json"]),
        str(user_id),
    )


# ============================================================
# CONVERSATION FUNCTIONS
# ============================================================


def _derive_title(content: str, maximum_length: int = 48) -> str:
    cleaned = " ".join(str(content).strip().split())

    if not cleaned:
        return "New conversation"
    if len(cleaned) <= maximum_length:
        return cleaned

    return cleaned[: maximum_length - 1].rstrip() + "…"


def _create_conversation(
    connection: sqlite3.Connection,
    user_id: str,
    is_active: bool = True,
) -> int:
    now = utc_now()
    cursor = connection.execute(
        """
        INSERT INTO conversations_v2 (
            user_id,
            title,
            is_active,
            created_at,
            updated_at
        )
        VALUES (?, 'New conversation', ?, ?, ?)
        """,
        (str(user_id), 1 if is_active else 0, now, now),
    )
    return int(cursor.lastrowid)


def get_or_create_active_conversation(user_id: str) -> int:
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT id
            FROM conversations_v2
            WHERE user_id = ? AND is_active = 1
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (str(user_id),),
        ).fetchone()

        if row:
            return int(row["id"])

        connection.execute(
            "UPDATE conversations_v2 SET is_active = 0 WHERE user_id = ?",
            (str(user_id),),
        )
        return _create_conversation(connection, str(user_id), True)


def start_new_conversation(user_id: str) -> int:
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE conversations_v2 SET is_active = 0 WHERE user_id = ?",
            (str(user_id),),
        )
        return _create_conversation(connection, str(user_id), True)


def get_conversation(
    user_id: str,
    conversation_id: int,
) -> Optional[Dict[str, Any]]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                c.id,
                c.user_id,
                c.title,
                c.is_active,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations_v2 AS c
            LEFT JOIN messages_v2 AS m
                ON m.conversation_id = c.id
            WHERE c.id = ? AND c.user_id = ?
            GROUP BY c.id
            LIMIT 1
            """,
            (int(conversation_id), str(user_id)),
        ).fetchone()

    if not row:
        return None

    return {
        "id": int(row["id"]),
        "user_id": str(row["user_id"]),
        "title": str(row["title"] or "New conversation"),
        "is_active": bool(row["is_active"]),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "message_count": int(row["message_count"] or 0),
    }


def list_conversations(
    user_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 100))
    get_or_create_active_conversation(str(user_id))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id,
                c.title,
                c.is_active,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count,
                (
                    SELECT content
                    FROM messages_v2 AS latest_message
                    WHERE latest_message.conversation_id = c.id
                    ORDER BY
                        latest_message.created_at DESC,
                        latest_message.id DESC
                    LIMIT 1
                ) AS preview
            FROM conversations_v2 AS c
            LEFT JOIN messages_v2 AS m
                ON m.conversation_id = c.id
            WHERE c.user_id = ?
            GROUP BY c.id
            ORDER BY c.is_active DESC, c.updated_at DESC, c.id DESC
            LIMIT ?
            """,
            (str(user_id), safe_limit),
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "title": str(row["title"] or "New conversation"),
            "is_active": bool(row["is_active"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "message_count": int(row["message_count"] or 0),
            "preview": str(row["preview"] or ""),
        }
        for row in rows
    ]


def activate_conversation(
    user_id: str,
    conversation_id: int,
) -> Dict[str, Any]:
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        exists = connection.execute(
            """
            SELECT 1
            FROM conversations_v2
            WHERE id = ? AND user_id = ?
            LIMIT 1
            """,
            (int(conversation_id), str(user_id)),
        ).fetchone()

        if not exists:
            raise ValueError("Conversation not found.")

        connection.execute(
            "UPDATE conversations_v2 SET is_active = 0 WHERE user_id = ?",
            (str(user_id),),
        )
        connection.execute(
            """
            UPDATE conversations_v2
            SET is_active = 1, updated_at = ?
            WHERE id = ?
            """,
            (utc_now(), int(conversation_id)),
        )

    conversation = get_conversation(user_id, conversation_id)
    if not conversation:
        raise ValueError("Conversation not found.")
    return conversation


def delete_conversation(
    user_id: str,
    conversation_id: int,
) -> Dict[str, Any]:
    with get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT is_active
            FROM conversations_v2
            WHERE id = ? AND user_id = ?
            LIMIT 1
            """,
            (int(conversation_id), str(user_id)),
        ).fetchone()

        if not row:
            raise ValueError("Conversation not found.")

        was_active = bool(row["is_active"])
        connection.execute(
            """
            DELETE FROM conversations_v2
            WHERE id = ? AND user_id = ?
            """,
            (int(conversation_id), str(user_id)),
        )

        active = connection.execute(
            """
            SELECT id
            FROM conversations_v2
            WHERE user_id = ? AND is_active = 1
            LIMIT 1
            """,
            (str(user_id),),
        ).fetchone()

        if active:
            active_conversation_id = int(active["id"])
        else:
            latest = connection.execute(
                """
                SELECT id
                FROM conversations_v2
                WHERE user_id = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (str(user_id),),
            ).fetchone()

            if latest:
                active_conversation_id = int(latest["id"])
                connection.execute(
                    """
                    UPDATE conversations_v2
                    SET is_active = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (utc_now(), active_conversation_id),
                )
            else:
                active_conversation_id = _create_conversation(
                    connection,
                    str(user_id),
                    True,
                )

    return {
        "deleted_conversation_id": int(conversation_id),
        "was_active": was_active,
        "active_conversation_id": active_conversation_id,
    }


# ============================================================
# MESSAGE FUNCTIONS
# ============================================================


def add_message(
    conversation_id: int,
    role: str,
    content: str,
) -> Dict[str, Any]:
    normalized_role = str(role).strip().lower()
    cleaned_content = str(content).strip()

    if normalized_role not in {"system", "user", "assistant"}:
        raise ValueError("Invalid message role.")
    if not cleaned_content:
        raise ValueError("Message content cannot be empty.")

    now = utc_now()

    with get_connection() as connection:
        conversation = connection.execute(
            """
            SELECT id, title
            FROM conversations_v2
            WHERE id = ?
            LIMIT 1
            """,
            (int(conversation_id),),
        ).fetchone()

        if not conversation:
            raise ValueError("Conversation not found.")

        cursor = connection.execute(
            """
            INSERT INTO messages_v2 (
                conversation_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                int(conversation_id),
                normalized_role,
                cleaned_content,
                now,
            ),
        )

        current_title = str(conversation["title"] or "")
        if normalized_role == "user" and current_title in {
            "",
            "New conversation",
        }:
            first_user_message = connection.execute(
                """
                SELECT content
                FROM messages_v2
                WHERE conversation_id = ? AND role = 'user'
                ORDER BY created_at ASC, id ASC
                LIMIT 1
                """,
                (int(conversation_id),),
            ).fetchone()

            if first_user_message:
                connection.execute(
                    "UPDATE conversations_v2 SET title = ? WHERE id = ?",
                    (
                        _derive_title(str(first_user_message["content"])),
                        int(conversation_id),
                    ),
                )

        connection.execute(
            "UPDATE conversations_v2 SET updated_at = ? WHERE id = ?",
            (now, int(conversation_id)),
        )
        message_id = int(cursor.lastrowid)

    return {
        "id": message_id,
        "conversation_id": int(conversation_id),
        "role": normalized_role,
        "content": cleaned_content,
        "created_at": now,
    }


def get_conversation_messages(
    user_id: str,
    conversation_id: int,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))

    with get_connection() as connection:
        exists = connection.execute(
            """
            SELECT 1
            FROM conversations_v2
            WHERE id = ? AND user_id = ?
            LIMIT 1
            """,
            (int(conversation_id), str(user_id)),
        ).fetchone()

        if not exists:
            raise ValueError("Conversation not found.")

        rows = connection.execute(
            """
            SELECT *
            FROM (
                SELECT id, conversation_id, role, content, created_at
                FROM messages_v2
                WHERE conversation_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            )
            ORDER BY created_at ASC, id ASC
            """,
            (int(conversation_id), safe_limit),
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "conversation_id": int(row["conversation_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "created_at": str(row["created_at"]),
        }
        for row in rows
    ]


def get_recent_messages(
    conversation_id: int,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 100))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM (
                SELECT id, conversation_id, role, content, created_at
                FROM messages_v2
                WHERE conversation_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            )
            ORDER BY created_at ASC, id ASC
            """,
            (int(conversation_id), safe_limit),
        ).fetchall()

    return [
        {
            "id": int(row["id"]),
            "conversation_id": int(row["conversation_id"]),
            "role": str(row["role"]),
            "content": str(row["content"]),
            "created_at": str(row["created_at"]),
        }
        for row in rows
    ]


def get_active_conversation_messages(
    user_id: str,
    limit: int = 100,
) -> Dict[str, Any]:
    conversation_id = get_or_create_active_conversation(str(user_id))
    return {
        "conversation_id": conversation_id,
        "messages": get_conversation_messages(
            str(user_id),
            conversation_id,
            limit=limit,
        ),
    }


def get_user_conversation_history(
    user_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    return get_active_conversation_messages(
        str(user_id),
        limit=limit,
    )["messages"]


# ============================================================
# REMINDER FUNCTIONS
# ============================================================


def _clean_reminder_title(title: Any) -> str:
    cleaned = " ".join(str(title or "").strip().split())

    if not cleaned:
        raise ValueError("A reminder title is required.")

    return cleaned[:200]


def _serialize_reminder(
    row: sqlite3.Row,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    current_time = now or utc_now_datetime()
    scheduled_for = parse_datetime(row["scheduled_for"])
    alert_at = parse_datetime(row["alert_at"])
    hidden_until = (
        parse_datetime(row["hidden_until"])
        if row["hidden_until"]
        else None
    )
    status = str(row["status"])
    is_due = status == "upcoming" and alert_at <= current_time
    is_overdue = status == "upcoming" and scheduled_for < current_time
    should_popup = is_due and (
        hidden_until is None or hidden_until <= current_time
    )

    return {
        "id": int(row["id"]),
        "user_id": str(row["user_id"]),
        "title": str(row["title"]),
        "scheduled_for": str(row["scheduled_for"]),
        "alert_at": str(row["alert_at"]),
        "alert_offset_minutes": int(row["alert_offset_minutes"] or 0),
        "status": status,
        "completed_at": (
            str(row["completed_at"]) if row["completed_at"] else None
        ),
        "hidden_until": (
            str(row["hidden_until"]) if row["hidden_until"] else None
        ),
        "notified_at": (
            str(row["notified_at"]) if row["notified_at"] else None
        ),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "is_due": is_due,
        "is_overdue": is_overdue,
        "should_popup": should_popup,
    }


def _get_reminder_row(
    connection: sqlite3.Connection,
    user_id: str,
    reminder_id: int,
) -> sqlite3.Row:
    row = connection.execute(
        """
        SELECT *
        FROM reminders
        WHERE id = ? AND user_id = ?
        LIMIT 1
        """,
        (int(reminder_id), str(user_id)),
    ).fetchone()

    if not row:
        raise ValueError("Reminder not found.")

    return row


def create_reminder(
    user_id: str,
    title: str,
    scheduled_for: str,
    alert_offset_minutes: int = 0,
) -> Dict[str, Any]:
    cleaned_title = _clean_reminder_title(title)
    scheduled_datetime = parse_datetime(scheduled_for)
    offset = max(0, min(int(alert_offset_minutes), 10080))
    alert_datetime = scheduled_datetime - timedelta(minutes=offset)
    now = utc_now()

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO reminders (
                user_id,
                title,
                scheduled_for,
                alert_at,
                alert_offset_minutes,
                status,
                completed_at,
                hidden_until,
                notified_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'upcoming', NULL, NULL, NULL, ?, ?)
            """,
            (
                str(user_id),
                cleaned_title,
                to_utc_iso(scheduled_datetime),
                to_utc_iso(alert_datetime),
                offset,
                now,
                now,
            ),
        )
        reminder_id = int(cursor.lastrowid)
        row = _get_reminder_row(connection, str(user_id), reminder_id)

    return _serialize_reminder(row)


def list_reminders(
    user_id: str,
    status: str = "all",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    normalized_status = str(status or "all").strip().lower()

    if normalized_status not in {"all", "upcoming", "completed"}:
        raise ValueError("Invalid reminder status.")

    safe_limit = max(1, min(int(limit), 250))
    where_status = ""
    parameters: List[Any] = [str(user_id)]

    if normalized_status != "all":
        where_status = " AND status = ?"
        parameters.append(normalized_status)

    parameters.append(safe_limit)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM reminders
            WHERE user_id = ?
            {where_status}
            ORDER BY
                CASE WHEN status = 'upcoming' THEN 0 ELSE 1 END,
                CASE
                    WHEN status = 'upcoming' THEN scheduled_for
                    ELSE completed_at
                END ASC,
                id ASC
            LIMIT ?
            """.format(where_status=where_status),
            tuple(parameters),
        ).fetchall()

    now = utc_now_datetime()
    reminders = [_serialize_reminder(row, now) for row in rows]

    # Completed reminders should show newest first.
    if normalized_status == "completed":
        reminders.sort(
            key=lambda item: item.get("completed_at") or "",
            reverse=True,
        )

    return reminders


def get_reminder_state(user_id: str) -> Dict[str, Any]:
    reminders = list_reminders(user_id, status="all", limit=250)
    upcoming = [item for item in reminders if item["status"] == "upcoming"]
    completed = [item for item in reminders if item["status"] == "completed"]
    completed.sort(
        key=lambda item: item.get("completed_at") or "",
        reverse=True,
    )

    due = [item for item in upcoming if item["is_due"]]
    popup_candidates = [item for item in due if item["should_popup"]]
    popup_candidates.sort(key=lambda item: (item["alert_at"], item["id"]))

    return {
        "upcoming": upcoming,
        "completed": completed,
        "due_count": len(due),
        "next_popup": popup_candidates[0] if popup_candidates else None,
        "notification_permission_supported": True,
        "server_time": utc_now(),
    }


def update_reminder(
    user_id: str,
    reminder_id: int,
    title: Optional[str] = None,
    scheduled_for: Optional[str] = None,
    alert_offset_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    with get_connection() as connection:
        current = _get_reminder_row(connection, user_id, reminder_id)
        new_title = (
            _clean_reminder_title(title)
            if title is not None
            else str(current["title"])
        )
        new_scheduled = (
            parse_datetime(scheduled_for)
            if scheduled_for is not None
            else parse_datetime(current["scheduled_for"])
        )
        new_offset = (
            max(0, min(int(alert_offset_minutes), 10080))
            if alert_offset_minutes is not None
            else int(current["alert_offset_minutes"] or 0)
        )
        new_alert = new_scheduled - timedelta(minutes=new_offset)
        schedule_changed = (
            to_utc_iso(new_scheduled) != str(current["scheduled_for"])
            or new_offset != int(current["alert_offset_minutes"] or 0)
        )

        connection.execute(
            """
            UPDATE reminders
            SET
                title = ?,
                scheduled_for = ?,
                alert_at = ?,
                alert_offset_minutes = ?,
                hidden_until = CASE WHEN ? THEN NULL ELSE hidden_until END,
                notified_at = CASE WHEN ? THEN NULL ELSE notified_at END,
                updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                new_title,
                to_utc_iso(new_scheduled),
                to_utc_iso(new_alert),
                new_offset,
                1 if schedule_changed else 0,
                1 if schedule_changed else 0,
                utc_now(),
                int(reminder_id),
                str(user_id),
            ),
        )
        row = _get_reminder_row(connection, user_id, reminder_id)

    return _serialize_reminder(row)


def complete_reminder(
    user_id: str,
    reminder_id: int,
) -> Dict[str, Any]:
    now = utc_now()

    with get_connection() as connection:
        _get_reminder_row(connection, user_id, reminder_id)
        connection.execute(
            """
            UPDATE reminders
            SET
                status = 'completed',
                completed_at = ?,
                hidden_until = NULL,
                updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (now, now, int(reminder_id), str(user_id)),
        )
        row = _get_reminder_row(connection, user_id, reminder_id)

    return _serialize_reminder(row)


def hide_reminder(
    user_id: str,
    reminder_id: int,
    minutes: int = 15,
) -> Dict[str, Any]:
    delay_minutes = max(1, min(int(minutes), 1440))
    hidden_until = to_utc_iso(
        utc_now_datetime() + timedelta(minutes=delay_minutes)
    )

    with get_connection() as connection:
        _get_reminder_row(connection, user_id, reminder_id)
        connection.execute(
            """
            UPDATE reminders
            SET hidden_until = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                hidden_until,
                utc_now(),
                int(reminder_id),
                str(user_id),
            ),
        )
        row = _get_reminder_row(connection, user_id, reminder_id)

    return _serialize_reminder(row)


def mark_reminder_notified(
    user_id: str,
    reminder_id: int,
) -> Dict[str, Any]:
    with get_connection() as connection:
        _get_reminder_row(connection, user_id, reminder_id)
        connection.execute(
            """
            UPDATE reminders
            SET notified_at = COALESCE(notified_at, ?), updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                utc_now(),
                utc_now(),
                int(reminder_id),
                str(user_id),
            ),
        )
        row = _get_reminder_row(connection, user_id, reminder_id)

    return _serialize_reminder(row)


def delete_reminder(user_id: str, reminder_id: int) -> Dict[str, Any]:
    with get_connection() as connection:
        row = _get_reminder_row(connection, user_id, reminder_id)
        reminder = _serialize_reminder(row)
        connection.execute(
            "DELETE FROM reminders WHERE id = ? AND user_id = ?",
            (int(reminder_id), str(user_id)),
        )

    return {
        "deleted_reminder_id": int(reminder_id),
        "reminder": reminder,
    }

# ============================================================
# OPEN LOOP FUNCTIONS
# ============================================================


def _clean_open_loop_text(
    value: Any,
    field_name: str,
    max_length: int,
    required: bool = False,
) -> Optional[str]:
    cleaned = " ".join(str(value or "").strip().split())

    if not cleaned:
        if required:
            raise ValueError(f"{field_name} is required.")
        return None

    return cleaned[:max_length]


def _normalize_open_loop_action(action: str) -> str:
    normalized = " ".join(
        "".join(
            character.lower()
            if character.isalnum() or character.isspace()
            else " "
            for character in str(action)
        ).split()
    )

    if not normalized:
        raise ValueError("Open-loop action is required.")

    return normalized[:220]


def _serialize_open_loop(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "user_id": str(row["user_id"]),
        "action": str(row["action"]),
        "normalized_action": str(row["normalized_action"]),
        "person": str(row["person"]) if row["person"] else None,
        "project": str(row["project"]) if row["project"] else None,
        "timing_text": (
            str(row["timing_text"]) if row["timing_text"] else None
        ),
        "scheduled_for": (
            str(row["scheduled_for"])
            if row["scheduled_for"]
            else None
        ),
        "status": str(row["status"]),
        "confidence": float(row["confidence"] or 0.0),
        "source_conversation_id": (
            int(row["source_conversation_id"])
            if row["source_conversation_id"] is not None
            else None
        ),
        "source_message_id": (
            int(row["source_message_id"])
            if row["source_message_id"] is not None
            else None
        ),
        "completed_at": (
            str(row["completed_at"]) if row["completed_at"] else None
        ),
        "dismissed_at": (
            str(row["dismissed_at"]) if row["dismissed_at"] else None
        ),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def _get_open_loop_row(
    connection: sqlite3.Connection,
    user_id: str,
    open_loop_id: int,
) -> sqlite3.Row:
    row = connection.execute(
        """
        SELECT *
        FROM open_loops
        WHERE id = ? AND user_id = ?
        LIMIT 1
        """,
        (int(open_loop_id), str(user_id)),
    ).fetchone()

    if not row:
        raise ValueError("Open loop not found.")

    return row


def create_open_loop(
    user_id: str,
    action: str,
    person: Optional[str] = None,
    project: Optional[str] = None,
    timing_text: Optional[str] = None,
    scheduled_for: Optional[str] = None,
    confidence: float = 0.8,
    source_conversation_id: Optional[int] = None,
    source_message_id: Optional[int] = None,
) -> Dict[str, Any]:
    cleaned_action = _clean_open_loop_text(
        action,
        "Open-loop action",
        220,
        required=True,
    )
    assert cleaned_action is not None

    normalized_action = _normalize_open_loop_action(cleaned_action)
    cleaned_person = _clean_open_loop_text(person, "Person", 120)
    cleaned_project = _clean_open_loop_text(project, "Project", 160)
    cleaned_timing = _clean_open_loop_text(
        timing_text,
        "Timing",
        160,
    )

    normalized_scheduled_for: Optional[str] = None
    if scheduled_for:
        normalized_scheduled_for = to_utc_iso(
            parse_datetime(scheduled_for)
        )

    try:
        normalized_confidence = float(confidence)
    except (TypeError, ValueError):
        normalized_confidence = 0.8

    normalized_confidence = max(0.0, min(normalized_confidence, 1.0))
    now = utc_now()

    with get_connection() as connection:
        duplicate = connection.execute(
            """
            SELECT *
            FROM open_loops
            WHERE user_id = ?
              AND normalized_action = ?
              AND status = 'open'
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            (str(user_id), normalized_action),
        ).fetchone()

        if duplicate:
            return {
                "open_loop": _serialize_open_loop(duplicate),
                "created": False,
            }

        cursor = connection.execute(
            """
            INSERT INTO open_loops (
                user_id,
                action,
                normalized_action,
                person,
                project,
                timing_text,
                scheduled_for,
                status,
                confidence,
                source_conversation_id,
                source_message_id,
                completed_at,
                dismissed_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (
                str(user_id),
                cleaned_action,
                normalized_action,
                cleaned_person,
                cleaned_project,
                cleaned_timing,
                normalized_scheduled_for,
                normalized_confidence,
                (
                    int(source_conversation_id)
                    if source_conversation_id is not None
                    else None
                ),
                (
                    int(source_message_id)
                    if source_message_id is not None
                    else None
                ),
                now,
                now,
            ),
        )
        open_loop_id = int(cursor.lastrowid)
        row = _get_open_loop_row(
            connection,
            str(user_id),
            open_loop_id,
        )

    return {
        "open_loop": _serialize_open_loop(row),
        "created": True,
    }


def list_open_loops(
    user_id: str,
    status: str = "open",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    normalized_status = str(status or "open").strip().lower()

    if normalized_status not in {
        "all",
        "open",
        "completed",
        "dismissed",
    }:
        raise ValueError("Invalid open-loop status.")

    safe_limit = max(1, min(int(limit), 250))
    where_status = ""
    parameters: List[Any] = [str(user_id)]

    if normalized_status != "all":
        where_status = " AND status = ?"
        parameters.append(normalized_status)

    parameters.append(safe_limit)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM open_loops
            WHERE user_id = ?
            {where_status}
            ORDER BY
                CASE WHEN scheduled_for IS NULL THEN 1 ELSE 0 END,
                scheduled_for ASC,
                updated_at DESC,
                id DESC
            LIMIT ?
            """.format(where_status=where_status),
            tuple(parameters),
        ).fetchall()

    return [_serialize_open_loop(row) for row in rows]


def get_open_loop_state(user_id: str) -> Dict[str, Any]:
    open_items = list_open_loops(user_id, status="open", limit=250)
    completed = list_open_loops(user_id, status="completed", limit=100)

    return {
        "open": open_items,
        "completed": completed,
        "open_count": len(open_items),
        "server_time": utc_now(),
    }


def complete_open_loop(
    user_id: str,
    open_loop_id: int,
) -> Dict[str, Any]:
    now = utc_now()

    with get_connection() as connection:
        current = _get_open_loop_row(
            connection,
            user_id,
            open_loop_id,
        )

        if str(current["status"]) == "dismissed":
            raise ValueError("A dismissed open loop cannot be completed.")

        connection.execute(
            """
            UPDATE open_loops
            SET
                status = 'completed',
                completed_at = COALESCE(completed_at, ?),
                dismissed_at = NULL,
                updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (now, now, int(open_loop_id), str(user_id)),
        )
        row = _get_open_loop_row(
            connection,
            user_id,
            open_loop_id,
        )

    return _serialize_open_loop(row)


def dismiss_open_loop(
    user_id: str,
    open_loop_id: int,
) -> Dict[str, Any]:
    now = utc_now()

    with get_connection() as connection:
        _get_open_loop_row(connection, user_id, open_loop_id)
        connection.execute(
            """
            UPDATE open_loops
            SET
                status = 'dismissed',
                dismissed_at = ?,
                updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (now, now, int(open_loop_id), str(user_id)),
        )
        row = _get_open_loop_row(
            connection,
            user_id,
            open_loop_id,
        )

    return _serialize_open_loop(row)


def undo_open_loop_capture(
    user_id: str,
    open_loop_id: int,
) -> Dict[str, Any]:
    with get_connection() as connection:
        row = _get_open_loop_row(connection, user_id, open_loop_id)
        open_loop = _serialize_open_loop(row)

        if open_loop["status"] != "open":
            raise ValueError("Only an open capture can be undone.")

        connection.execute(
            "DELETE FROM open_loops WHERE id = ? AND user_id = ?",
            (int(open_loop_id), str(user_id)),
        )

    return {
        "deleted_open_loop_id": int(open_loop_id),
        "open_loop": open_loop,
    }


def get_most_recent_open_loop(
    user_id: str,
    source_conversation_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    parameters: List[Any] = [str(user_id)]
    conversation_filter = ""

    if source_conversation_id is not None:
        conversation_filter = " AND source_conversation_id = ?"
        parameters.append(int(source_conversation_id))

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM open_loops
            WHERE user_id = ?
              AND status = 'open'
              {conversation_filter}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """.format(conversation_filter=conversation_filter),
            tuple(parameters),
        ).fetchone()

    return _serialize_open_loop(row) if row else None

# ============================================================
# PERSISTENT USER MEMORY
# ============================================================


MEMORY_CATEGORIES = (
    "facts",
    "relationships",
    "situations",
    "preferences",
    "patterns",
)


def _empty_memory() -> Dict[str, List[Dict[str, Any]]]:
    return {category: [] for category in MEMORY_CATEGORIES}


def _clean_memory_text(value: Any, maximum: int = 500) -> str:
    return " ".join(str(value or "").strip().split())[:maximum]


def _memory_identity(item: Dict[str, Any]) -> str:
    key = _clean_memory_text(item.get("key"), 160).lower()
    value = _clean_memory_text(item.get("value"), 500).lower()
    raw = key or value
    return re.sub(r"[^a-z0-9]+", " ", raw).strip()


def _normalize_memory_object(value: Any) -> Dict[str, List[Dict[str, Any]]]:
    source = value if isinstance(value, dict) else {}
    normalized = _empty_memory()

    for category in MEMORY_CATEGORIES:
        raw_items = source.get(category)
        if not isinstance(raw_items, list):
            continue

        seen = set()
        for raw_item in raw_items[-80:]:
            if isinstance(raw_item, str):
                raw_item = {"value": raw_item}
            if not isinstance(raw_item, dict):
                continue

            key = _clean_memory_text(raw_item.get("key"), 160)
            item_value = _clean_memory_text(raw_item.get("value"), 500)
            if not item_value:
                continue

            try:
                confidence = float(raw_item.get("confidence", 0.8))
            except (TypeError, ValueError):
                confidence = 0.8

            item = {
                "key": key or item_value[:80],
                "value": item_value,
                "confidence": max(0.0, min(confidence, 1.0)),
                "source_conversation_id": _safe_int(
                    raw_item.get("source_conversation_id")
                ),
                "source_message_id": _safe_int(raw_item.get("source_message_id")),
                "updated_at": _clean_memory_text(
                    raw_item.get("updated_at") or utc_now(),
                    80,
                ),
            }
            identity = _memory_identity(item)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            normalized[category].append(item)

    return normalized


def get_user_memory(user_id: str) -> Dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT memory_json, created_at, updated_at
            FROM user_memory
            WHERE user_id = ?
            LIMIT 1
            """,
            (str(user_id),),
        ).fetchone()

    if not row:
        return {
            "user_id": str(user_id),
            "memory": _empty_memory(),
            "created_at": None,
            "updated_at": None,
        }

    return {
        "user_id": str(user_id),
        "memory": _normalize_memory_object(_json_dict(row["memory_json"])),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def upsert_user_memory(
    user_id: str,
    updates: List[Dict[str, Any]],
    source_conversation_id: Optional[int] = None,
    source_message_id: Optional[int] = None,
) -> Dict[str, Any]:
    current_record = get_user_memory(user_id)
    memory = _normalize_memory_object(current_record.get("memory"))
    now = utc_now()
    saved: List[Dict[str, Any]] = []

    for raw_update in updates[:20]:
        if not isinstance(raw_update, dict):
            continue

        category = _clean_memory_text(raw_update.get("category"), 40).lower()
        if category not in MEMORY_CATEGORIES:
            continue

        key = _clean_memory_text(raw_update.get("key"), 160)
        value = _clean_memory_text(raw_update.get("value"), 500)
        if not value:
            continue

        try:
            confidence = float(raw_update.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        if confidence < 0.78:
            continue

        new_item = {
            "key": key or value[:80],
            "value": value,
            "confidence": confidence,
            "source_conversation_id": source_conversation_id,
            "source_message_id": source_message_id,
            "updated_at": now,
        }
        identity = _memory_identity(new_item)
        if not identity:
            continue

        category_items = memory[category]
        replacement_index: Optional[int] = None
        for index, existing in enumerate(category_items):
            existing_key = _clean_memory_text(existing.get("key"), 160).lower()
            new_key = _clean_memory_text(new_item.get("key"), 160).lower()
            if existing_key and new_key and existing_key == new_key:
                replacement_index = index
                break
            if _memory_identity(existing) == identity:
                replacement_index = index
                break

        if replacement_index is None:
            category_items.append(new_item)
        else:
            category_items[replacement_index] = new_item

        memory[category] = category_items[-80:]
        saved.append({"category": category, **new_item})

    created_at = current_record.get("created_at") or now
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO user_memory (
                user_id,
                memory_json,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                memory_json = excluded.memory_json,
                updated_at = excluded.updated_at
            """,
            (
                str(user_id),
                json.dumps(memory, ensure_ascii=False),
                str(created_at),
                now,
            ),
        )

    return {
        **get_user_memory(user_id),
        "saved": saved,
    }


def clear_user_memory(user_id: str) -> Dict[str, Any]:
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM user_memory WHERE user_id = ?",
            (str(user_id),),
        )
    return {
        "user_id": str(user_id),
        "memory": _empty_memory(),
        "cleared": True,
    }


# ============================================================
# ROLLING CONVERSATION SUMMARIES
# ============================================================


def get_conversation_summary(
    user_id: str,
    conversation_id: int,
) -> str:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT summary
            FROM conversation_summaries
            WHERE conversation_id = ? AND user_id = ?
            LIMIT 1
            """,
            (int(conversation_id), str(user_id)),
        ).fetchone()
    return _clean_memory_text(row["summary"], 1800) if row else ""


def save_conversation_summary(
    user_id: str,
    conversation_id: int,
    summary: Any,
) -> str:
    cleaned = _clean_memory_text(summary, 1800)
    if not cleaned:
        return get_conversation_summary(user_id, conversation_id)

    now = utc_now()
    with get_connection() as connection:
        existing = connection.execute(
            """
            SELECT created_at
            FROM conversation_summaries
            WHERE conversation_id = ? AND user_id = ?
            LIMIT 1
            """,
            (int(conversation_id), str(user_id)),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing else now
        connection.execute(
            """
            INSERT INTO conversation_summaries (
                conversation_id,
                user_id,
                summary,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                user_id = excluded.user_id,
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (
                int(conversation_id),
                str(user_id),
                cleaned,
                created_at,
                now,
            ),
        )
    return cleaned


def get_recent_conversation_summaries(
    user_id: str,
    exclude_conversation_id: Optional[int] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    parameters: List[Any] = [str(user_id)]
    exclusion = ""
    if exclude_conversation_id is not None:
        exclusion = " AND cs.conversation_id != ?"
        parameters.append(int(exclude_conversation_id))
    parameters.append(max(1, min(int(limit), 20)))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                cs.conversation_id,
                cs.summary,
                cs.updated_at,
                c.title
            FROM conversation_summaries AS cs
            LEFT JOIN conversations_v2 AS c
                ON c.id = cs.conversation_id
            WHERE cs.user_id = ?
              {exclusion}
            ORDER BY cs.updated_at DESC
            LIMIT ?
            """.format(exclusion=exclusion),
            tuple(parameters),
        ).fetchall()

    return [
        {
            "conversation_id": int(row["conversation_id"]),
            "title": str(row["title"] or "Conversation"),
            "summary": _clean_memory_text(row["summary"], 1800),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]
