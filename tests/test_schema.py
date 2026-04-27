import sqlite3

from zeroalpha.db.schema import initialize_sqlite


def test_initialize_sqlite_schema(tmp_path) -> None:
    db_path = tmp_path / "zeroalpha.sqlite"
    initialize_sqlite(db_path)
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "orders" in tables
    assert "fills" in tables
    assert "candidate_events" in tables
