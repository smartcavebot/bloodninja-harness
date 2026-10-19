"""Single-producer, single-consumer, SQLite-backed work queue.

The broker commits before delivery and acks *after* processing. Failed jobs are
retained with an error marker. There is deliberately no blind retry of toolful
worker jobs: replaying a partially completed task can repeat non-idempotent effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import threading
import time
import uuid


@dataclass(frozen=True)
class Job:
    id: str
    role: str
    round_num: int
    message: str


class Broker:
    def __init__(self, path: str | Path = ':memory:'):
        self.path = str(path)
        if self.path != ':memory:':
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute('PRAGMA busy_timeout=10000')
        self.db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE NOT NULL, role TEXT NOT NULL, round_num INTEGER NOT NULL,
            message TEXT NOT NULL, state TEXT NOT NULL,
            published_ns INTEGER NOT NULL,
            started_ns INTEGER, completed_ns INTEGER, failure TEXT
        )''')
        self.db.commit()

    def publish(self, role: str, round_num: int, message: str, job_id: str | None = None) -> str:
        if not role or not isinstance(message, str):
            raise ValueError('role and string message required')
        job_id = job_id or str(uuid.uuid4())
        with self.lock:
            self.db.execute('''INSERT OR IGNORE INTO jobs
                (job_id, role, round_num, message, state, published_ns)
                VALUES (?, ?, ?, ?, 'pending', ?)''',
                (job_id, role, round_num, message, time.time_ns()))
            self.db.commit()
        return job_id

    def take(self) -> Job | None:
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('''SELECT job_id, role, round_num, message
                FROM jobs WHERE state='pending' ORDER BY sequence LIMIT 1''').fetchone()
            if row:
                self.db.execute('''UPDATE jobs SET state='running', started_ns=? WHERE job_id=?''',
                                (time.time_ns(), row[0]))
            self.db.commit()
        return Job(*row) if row else None

    def finish(self, job_id: str, error: str | None = None) -> None:
        with self.lock:
            cur = self.db.execute('''UPDATE jobs SET state=?, completed_ns=?, failure=?
                WHERE job_id=? AND state='running' ''',
                ('failed' if error else 'complete', time.time_ns(), error, job_id))
            if cur.rowcount != 1:
                raise ValueError(f'job was not running: {job_id}')
            self.db.commit()

    def states(self) -> list[tuple[str, str, str]]:
        return self.db.execute('SELECT job_id, role, state FROM jobs ORDER BY sequence').fetchall()

    def close(self) -> None:
        with self.lock:
            self.db.close()
