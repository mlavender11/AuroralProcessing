"""
run.py -- the orchestrator.

What it does:
  - discovers every night on every drive
  - runs exactly ONE worker per physical drive, so two readers never fight
    over the same disk (which on spinning disks is slower than going serial)
  - within a drive, processes nights one at a time, sequentially
  - skips nights already 'done', so it is fully resumable
  - recovers nights left 'in_progress' by a previous crash

Usage:
  python run.py                 # process everything not yet done
  python run.py --retry-failed  # also retry nights that previously failed
  python run.py --serial        # one drive at a time (easiest for debugging)
  python run.py --limit 1       # only the first pending night per drive (a test run)
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

from config import DRIVES, discover_nights
import db
from process_night import process_night


def nights_for_drive(drive, retry_failed, limit):
    """Runs in its own process. Opens its OWN db connection (connections can't
    be shared across processes), decides this drive's to-do list, works it."""
    conn = db.connect()
    todo = []
    for i, (night_id, raw_path) in enumerate(discover_nights(drive)):
        db.register_night(conn, night_id, drive["id"], raw_path)
        status = db.get_status(conn, night_id)
        if status == "done":
            continue
        if status == "failed" and not retry_failed:
            continue
        todo.append((night_id, raw_path))
        if limit and len(todo) >= limit:  # stop once we have `limit` nights to actually do
            break

    print(f"[{drive['id']}] {len(todo)} nights to process")
    for night_id, raw_path in todo:
        process_night(conn, night_id, drive["id"], raw_path)
    conn.close()
    return drive["id"], len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    db.init_db()

    # Recover from any previous crash before starting fresh workers.
    conn = db.connect()
    requeued = db.reset_stale_in_progress(conn)
    if requeued:
        print(f"Requeued {requeued} night(s) left 'in_progress' by a previous run.")
    conn.close()

    if args.serial:
        for drive in DRIVES:
            nights_for_drive(drive, args.retry_failed, args.limit)
    else:
        # One worker process per drive. max_workers == number of drives.
        with ProcessPoolExecutor(max_workers=len(DRIVES)) as ex:
            futures = [ex.submit(nights_for_drive, d, args.retry_failed, args.limit) for d in DRIVES]
            for fut in futures:
                drive_id, n = fut.result()
                print(f"[{drive_id}] finished ({n} nights attempted)")


if __name__ == "__main__":
    main()
