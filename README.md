# Aurora processing pipeline

A resumable batch pipeline that turns raw binary nights into keograms + summary
videos, records a per-night processing log with cheap triage statistics, and
lets you build a catalog of interesting events.

## Files
- `config.py`        -- the ONLY file you must edit. Paths, drive list, and wrappers around your scripts.
- `db.py`            -- SQLite processing log + catalog (schema and helpers).
- `stats.py`         -- cheap per-night statistics, streamed from the HDF.
- `process_night.py` -- all the work for one night, with error handling.
- `run.py`           -- the orchestrator: one worker per drive, resumable.
- `catalog.py`       -- triage candidate nights, and add/list/export events.

## One-time setup
```
pip install h5py numpy        # the only dependencies
```
Then open `config.py` and fill in the four `>>> EDIT <<<` sections:
1. your physical drives (this also sets how many workers run -- one per drive),
2. scratch / output / database / HDF-archive paths,
3. how nights are laid out on disk (`discover_nights`),
4. the commands that run your three existing scripts.

## Keeping HDF files
`KEEP_HDF` in `config.py` controls what happens to each night's HDF after its
keogram and video are made:
- `"none"`        -- delete every HDF (smallest footprint).
- `"all"`         -- keep every HDF. Storage = (total nights) x (HDF size); can be huge.
- `"interesting"` -- keep only nights whose `motion` stat is at/above
  `KEEP_MOTION_THRESHOLD`. Recommended: captures the active nights, drops the quiet ones.

Kept HDFs are moved off scratch into `HDF_DIR` (make that large and persistent,
on a different physical disk from scratch and from the raw drives). The archived
path is recorded per night, so `catalog.py triage` shows which nights have an
HDF, and any event you catalog points straight at the full-resolution file.

## Test before the big run (do this first!)
```
python run.py --serial --limit 1     # process ONE night, one drive at a time
```
`--serial` gives you clean tracebacks if a wrapper command is wrong. Check that
a keogram and video appeared in OUTPUT_DIR and that the night shows up:
```
python catalog.py triage             # should list your one processed night
```

## The full run
```
python run.py                        # one worker per drive, skips finished nights
```
Safe to stop with Ctrl-C and restart any time -- it resumes where it left off.
To run it unattended for days, use `nohup` or `tmux`/`screen` so it survives logout:
```
nohup python run.py > run.log 2>&1 &
tail -f run.log                      # watch progress
```

## After a crash or to retry failures
```
python run.py                        # in_progress nights are auto-requeued at startup
python run.py --retry-failed         # also re-attempt nights that previously failed
```
Inspect what failed and why:
```
sqlite3 <DB_PATH> "SELECT night_id, error FROM nights WHERE status='failed';"
```

## Triage and cataloging
```
python catalog.py triage --top 20          # 20 most active nights
python catalog.py triage --percentile 95   # the busiest 5% of nights

python catalog.py add --night 20240115 --start 08:32 --end 08:47 \
    --type substorm --morphology "breakup, fast motion" \
    --notes "bright westward surge" --frame-start 14200 --frame-end 18050

python catalog.py list
python catalog.py export catalog.csv
```
Every catalog row automatically stores the drive and raw path for that night,
so you (or the next student) can always find the full-resolution data later.
