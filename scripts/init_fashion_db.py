# scripts/init_fashion_db.py
from pathlib import Path
import sqlite3
import argparse

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS items(
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_asin         TEXT,
    main_category       TEXT,
    title               TEXT,
    average_rating      REAL,
    rating_number       INTEGER,
    price               REAL,
    store               TEXT,
    date_first_available TEXT,
    image_main_url      TEXT,
    created_at          TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS item_features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id     INTEGER NOT NULL,
    position    INTEGER NOT NULL,
    feature     TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES items(id)
);

CREATE TABLE IF NOT EXISTS item_images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id     INTEGER NOT NULL,
    position    INTEGER NOT NULL,
    variant     TEXT,
    thumb_url   TEXT,
    large_url   TEXT,
    hi_res_url  TEXT,
    FOREIGN KEY (item_id) REFERENCES items(id)
);

CREATE TABLE IF NOT EXISTS item_details (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id     INTEGER NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT,
    FOREIGN KEY (item_id) REFERENCES items(id)
);

CREATE TABLE IF NOT EXISTS item_docs (
    item_id   INTEGER PRIMARY KEY,
    doc_text  TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES items(id)
);
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="생성할 SQLite DB 경로 (예: data/fashion_items.db)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    db_path: Path = args.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        print(f"[INFO] Initialized DB schema at {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
