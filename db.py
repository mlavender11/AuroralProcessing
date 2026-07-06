"""
db.py -- the SQLite processing log and the event catalog.

Two tables, one file:
  nights : one row per night = the processing log AND the cheap stats (triage).
  events : one row per interesting thing you find = the catalog.

SQLite is just a single file (config.DB_PATH). Multiple worker processes can
share it safely because we turn on WAL mode and a long busy timeout, and
because writes are infrequent (once per finished night, not per frame).
"""
import sqlite3
from datetime import datetime, timezone
from config import DB_PATH


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)   # wait up to 30s if another worker holds the lock
    conn.execute("PRAGMA journal_mode=WAL;")       # readers + one writer can coexist
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.row_factory = sqlite3.Row                 # rows behave like dicts: row["status"]
    return conn


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS nights (
        night_id        TEXT PRIMARY KEY,
        drive_id        TEXT,
        raw_path        TEXT,
        status          TEXT DEFAULT 'pending',   -- pending | in_progress | done | failed
        error           TEXT,
        frame_count     INTEGER,
        max_brightness  REAL,
        mean_brightness REAL,
        brightness_std  REAL,    -- spread of per-frame brightness across the night
        motion          REAL,    -- mean frame-to-frame change (good activity proxy)
        keogram_path    TEXT,
        video_path      TEXT,
        hdf_path        TEXT,    -- archived HDF if kept, else NULL
        started_at      TEXT,
        finished_at     TEXT,
        updated_at      TEXT
    );
    CREATE TABLE IF NOT EXISTS events (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        night_id     TEXT,
        drive_id     TEXT,
        start_time   TEXT,
        end_time     TEXT,
        event_type   TEXT,
        morphology   TEXT,
        notes        TEXT,
        keogram_path TEXT,
        raw_path     TEXT,       -- pointer back to where the raw data lives
        hdf_path     TEXT,       -- archived full-res HDF for this night, if kept
        frame_start  INTEGER,
        frame_end    INTEGER,
        created_at   TEXT
    );
    """)
    conn.commit()
    conn.close()


# --- processing-log helpers --------------------------------------------------
def register_night(conn, night_id, drive_id, raw_path):
    """Ensure a row exists for this night. No-op if it's already there."""
    conn.execute(
        "INSERT OR IGNORE INTO nights (night_id, drive_id, raw_path, status, updated_at) "
        "VALUES (?, ?, ?, 'pending', ?)",
        (night_id, drive_id, str(raw_path), now()),
    )
    conn.commit()


def get_status(conn, night_id):
    row = conn.execute("SELECT status FROM nights WHERE night_id=?", (night_id,)).fetchone()
    return row["status"] if row else None


def mark_in_progress(conn, night_id):
    conn.execute(
        "UPDATE nights SET status='in_progress', started_at=?, error=NULL, updated_at=? "
        "WHERE night_id=?",
        (now(), now(), night_id),
    )
    conn.commit()


def mark_done(conn, night_id, stats, keogram_path, video_path, hdf_path=None):
    conn.execute(
        "UPDATE nights SET status='done', error=NULL, "
        "frame_count=?, max_brightness=?, mean_brightness=?, brightness_std=?, motion=?, "
        "keogram_path=?, video_path=?, hdf_path=?, finished_at=?, updated_at=? WHERE night_id=?",
        (stats["frame_count"], stats["max_brightness"], stats["mean_brightness"],
         stats["brightness_std"], stats["motion"],
         str(keogram_path), str(video_path),
         (str(hdf_path) if hdf_path else None), now(), now(), night_id),
    )
    conn.commit()


def mark_failed(conn, night_id, error):
    conn.execute(
        "UPDATE nights SET status='failed', error=?, updated_at=? WHERE night_id=?",
        (str(error)[:2000], now(), night_id),
    )
    conn.commit()


def reset_stale_in_progress(conn):
    """A crash leaves nights stuck at 'in_progress'. Call once at startup to
    requeue them so they get retried."""
    n = conn.execute("UPDATE nights SET status='pending' WHERE status='in_progress'").rowcount
    conn.commit()
    return n


# --- catalog helper ----------------------------------------------------------
def add_event(conn, **f):
    conn.execute(
        "INSERT INTO events (night_id, drive_id, start_time, end_time, event_type, "
        "morphology, notes, keogram_path, raw_path, hdf_path, frame_start, frame_end, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (f.get("night_id"), f.get("drive_id"), f.get("start_time"), f.get("end_time"),
         f.get("event_type"), f.get("morphology"), f.get("notes"), f.get("keogram_path"),
         f.get("raw_path"), f.get("hdf_path"), f.get("frame_start"), f.get("frame_end"), now()),
    )
    conn.commit()
