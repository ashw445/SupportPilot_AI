import sqlite3
from contextlib import closing

DB_PATH = "tickets.db"


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: str = DB_PATH) -> None:
    with closing(connect(db_path)) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    employee_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    title TEXT NOT NULL,
                    department TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT DEFAULT 'Open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )


def insert_ticket(
    employee_name: str,
    email: str,
    title: str,
    department: str,
    description: str,
    category: str,
    severity: str,
    priority: str,
    confidence: float,
    status: str = "Open",
    db_path: str = DB_PATH,
) -> int:
    with closing(connect(db_path)) as conn:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO tickets (
                    employee_name, email, title, department, description,
                    category, severity, priority, confidence, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    employee_name,
                    email,
                    title,
                    department,
                    description,
                    category,
                    severity,
                    priority,
                    confidence,
                    status,
                ),
            )
            return int(cursor.lastrowid)


def get_all_tickets(db_path: str = DB_PATH) -> list[dict]:
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM tickets ORDER BY created_at DESC, ticket_id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

def update_status()