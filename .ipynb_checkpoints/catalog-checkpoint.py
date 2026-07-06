"""
catalog.py -- build and query your catalog of interesting events.

Triage (find candidate nights to look at):
  python catalog.py triage --top 20
  python catalog.py triage --percentile 95

Add an event you decided is interesting:
  python catalog.py add --night 20240115 --start 08:32 --end 08:47 \
      --type substorm --morphology "breakup, fast westward motion" \
      --notes "bright surge, good for the catalog" --frame-start 14200 --frame-end 18050

List / export the catalog:
  python catalog.py list
  python catalog.py export catalog.csv
"""
import argparse
import csv
import db


def cmd_triage(args):
    conn = db.connect()
    rows = conn.execute(
        "SELECT night_id, drive_id, frame_count, mean_brightness, brightness_std, "
        "motion, keogram_path, hdf_path FROM nights WHERE status='done' ORDER BY motion DESC"
    ).fetchall()
    conn.close()

    if not rows:
        print("No processed nights yet. Run the pipeline first.")
        return

    if args.percentile is not None:
        vals = sorted((r["motion"] or 0) for r in rows)
        k = int(len(vals) * args.percentile / 100)
        thresh = vals[min(k, len(vals) - 1)]
        rows = [r for r in rows if (r["motion"] or 0) >= thresh]
        print(f"Nights at/above the {args.percentile:g}th percentile of motion "
              f"(>= {thresh:.3f}):")
    elif args.top:
        rows = rows[:args.top]
        print(f"Top {args.top} nights by motion:")
    else:
        print("All processed nights by motion (use --top N or --percentile P to filter):")

    # 'hdf' column shows whether the full-resolution HDF was archived for this night.
    print(f"{'night':>10}  {'drive':>7}  {'motion':>8}  {'b_std':>8}  {'frames':>8}  {'hdf':>4}  keogram")
    for r in rows:
        has_hdf = "yes" if r["hdf_path"] else "-"
        print(f"{r['night_id']:>10}  {r['drive_id'] or '':>7}  "
              f"{(r['motion'] or 0):8.3f}  {(r['brightness_std'] or 0):8.2f}  "
              f"{(r['frame_count'] or 0):8d}  {has_hdf:>4}  {r['keogram_path'] or ''}")


def cmd_add(args):
    conn = db.connect()
    # Pull drive/raw_path/hdf_path from the nights table so the event points
    # back to both the raw data AND the archived HDF (if one was kept).
    row = conn.execute("SELECT drive_id, raw_path, hdf_path FROM nights WHERE night_id=?",
                       (args.night,)).fetchone()
    db.add_event(
        conn,
        night_id=args.night,
        drive_id=row["drive_id"] if row else None,
        raw_path=row["raw_path"] if row else None,
        hdf_path=row["hdf_path"] if row else None,
        start_time=args.start, end_time=args.end,
        event_type=args.type, morphology=args.morphology, notes=args.notes,
        frame_start=args.frame_start, frame_end=args.frame_end,
        keogram_path=args.keogram,
    )
    conn.close()
    print(f"Added event on {args.night}.")


def cmd_list(args):
    conn = db.connect()
    rows = conn.execute("SELECT * FROM events ORDER BY night_id, start_time").fetchall()
    conn.close()
    for r in rows:
        print(f"#{r['id']} {r['night_id']} {r['start_time'] or '?'}-{r['end_time'] or '?'} "
              f"[{r['event_type'] or '-'}] {r['morphology'] or ''} :: {r['notes'] or ''}")
    print(f"\n{len(rows)} event(s).")


def cmd_export(args):
    conn = db.connect()
    rows = conn.execute("SELECT * FROM events ORDER BY night_id, start_time").fetchall()
    conn.close()
    if not rows:
        print("No events to export.")
        return
    with open(args.path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(rows[0].keys())
        for r in rows:
            w.writerow([r[k] for k in r.keys()])
    print(f"Wrote {len(rows)} event(s) to {args.path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="rank processed nights by activity")
    t.add_argument("--top", type=int)
    t.add_argument("--percentile", type=float)
    t.set_defaults(func=cmd_triage)

    a = sub.add_parser("add", help="add an interesting event to the catalog")
    a.add_argument("--night", required=True)
    a.add_argument("--start")
    a.add_argument("--end")
    a.add_argument("--type")
    a.add_argument("--morphology")
    a.add_argument("--notes")
    a.add_argument("--frame-start", type=int, dest="frame_start")
    a.add_argument("--frame-end", type=int, dest="frame_end")
    a.add_argument("--keogram")
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="print the catalog")
    l.set_defaults(func=cmd_list)

    e = sub.add_parser("export", help="export the catalog to CSV")
    e.add_argument("path")
    e.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
