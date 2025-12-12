# tests/conftest.py
import pickle
import sqlite3
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """sqlite 존재하는지"""
    db_path = tmp_path / "test_fashion_items.db"
    conn = sqlite3.connect(db_path)
    conn.close()
    return db_path


@pytest.fixture
def bm25_pickle_path(tmp_path: Path) -> Path:
    """BM250api + item_ids를 기대하는 포멧으로 저장"""
    corpus = [
        ["red", "dress"],
        ["blue", "jeans"],
        ["red", "shoes"],
        ["black", "shoes"],
    ]
    bm25 = BM25Okapi(corpus)
    item_ids = [101, 102, 103, 104]

    pkl = tmp_path / "bm25.pkl"
    with pkl.open("wb") as f:
        pickle.dump({"bm25": bm25, "item_ids": item_ids}, f)
    return pkl
