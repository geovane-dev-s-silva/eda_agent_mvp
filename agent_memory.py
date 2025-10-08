"""
agent_memory.py

Utilities to load/save conversational memory for a given dataset.
Uses DB_CONN from eda_agent (SQLite).
"""
import time
import uuid
from typing import List, Tuple
from eda_agent import DB_CONN

def save_memory(dataset_id: str, question: str, answer: str):
    cur = DB_CONN.cursor()
    qid = str(uuid.uuid4())
    cur.execute("INSERT INTO queries VALUES (?,?,?,?,?,?,?)", (
        qid, dataset_id, question, (answer[:2000] if answer else ""), (answer[:2000] if answer else ""), time.strftime("%Y-%m-%d %H:%M:%S"), "memory"
    ))
    DB_CONN.commit()
    return qid

def load_memory(dataset_id: str, limit: int = 5):
    cur = DB_CONN.cursor()
    rows = cur.execute("SELECT question, response_summary FROM queries WHERE dataset_id=? ORDER BY created_at DESC LIMIT ?", (dataset_id, limit)).fetchall()
    rows = rows[::-1]
    return [(r[0], r[1]) for r in rows]
