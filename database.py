import sqlite3
from contextlib import closing

DB_PATH = "tickets.db"
VALID_STATUSES = {"Open", "In Progress", "Resolved"}


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


def get_all_tickets(
    db_path: str = DB_PATH,
    status_filter: str | None = None,
    priority_filter: str | None = None,
    search_query: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM tickets WHERE 1=1"
    params = []

    if status_filter and status_filter != "All":
        query += " AND status = ?"
        params.append(status_filter)

    if priority_filter and priority_filter != "All":
        query += " AND priority = ?"
        params.append(priority_filter)

    if search_query:
        query += " AND (title LIKE ? OR description LIKE ? OR employee_name LIKE ? OR category LIKE ?)"
        wildcard = f"%{search_query.strip()}%"
        params.extend([wildcard, wildcard, wildcard, wildcard])

    query += " ORDER BY created_at DESC, ticket_id DESC"

    with closing(connect(db_path)) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def update_ticket_status(ticket_id: int, new_status: str, db_path: str = DB_PATH) -> bool:
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {new_status}")

    with closing(connect(db_path)) as conn:
        with conn:
            cursor = conn.execute(
                "UPDATE tickets SET status = ? WHERE ticket_id = ?",
                (new_status, ticket_id),
            )
            return cursor.rowcount > 0


def delete_ticket(ticket_id: int, db_path: str = DB_PATH) -> bool:
    with closing(connect(db_path)) as conn:
        with conn:
            cursor = conn.execute(
                "DELETE FROM tickets WHERE ticket_id = ?",
                (ticket_id,),
            )
            return cursor.rowcount > 0


def get_ticket_stats(db_path: str = DB_PATH) -> dict:
    with closing(connect(db_path)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        open_count = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'Open'").fetchone()[0]
        in_progress = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'In Progress'").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM tickets WHERE status = 'Resolved'").fetchone()[0]
        critical = conn.execute("SELECT COUNT(*) FROM tickets WHERE priority = 'P1' AND status != 'Resolved'").fetchone()[0]

    return {
        "total": total,
        "open": open_count,
        "in_progress": in_progress,
        "resolved": resolved,
        "critical_active": critical,
    }