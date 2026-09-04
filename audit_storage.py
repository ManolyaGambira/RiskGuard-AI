import sqlite3
from pathlib import Path
from datetime import datetime

DB_FILE = Path(__file__).parent / "riskguard_audit.db"


def initialize_audit_db():
    connection = sqlite3.connect(DB_FILE)

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            event TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )

    connection.commit()
    connection.close()


def save_audit_event(transaction_id: str, event: str):
    connection = sqlite3.connect(DB_FILE)

    existing = connection.execute(
        """
        SELECT id
        FROM audit_events
        WHERE transaction_id = ?
        AND event = ?
        """,
        (transaction_id, event),
    ).fetchone()

    if existing:
        connection.close()
        return

    connection.execute(
        """
        INSERT INTO audit_events
        (transaction_id, event, timestamp)
        VALUES (?, ?, ?)
        """,
        (
            transaction_id,
            event,
            datetime.now().isoformat(),
        ),
    )

    connection.commit()
    connection.close()


def get_audit_events(transaction_id: str):
    connection = sqlite3.connect(DB_FILE)

    cursor = connection.execute(
        """
        SELECT event, timestamp
        FROM audit_events
        WHERE transaction_id = ?
        ORDER BY id ASC
        """,
        (transaction_id,),
    )

    events = cursor.fetchall()
    connection.close()

    return events