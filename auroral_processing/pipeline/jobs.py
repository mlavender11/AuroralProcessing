"""
Run jobs (sources x chains x grid) and keep an index of every output in an output folder.

``index.csv`` sits at the top of the output folder, one row per output, rebuilt from the
``frames.h5`` attributes, so it is never out of date with the files and can be regenerated any time
with ``build_index`` / ``aurora pipeline list``.
"""

from __future__ import annotations
from pathlib import Path
import traceback
import pandas as pd

from auroral_processing.pipeline.config import Job
from auroral_processing.pipeline.processing_chain import (
    OUTPUT_FILE, PREVIEW_FILE, find_output, is_complete, read_output_attrs, run_chain,
)

INDEX_FILE = "index.csv"


def run_job(job: Job, *, dry_run=False, force=False, progress=True) -> pd.DataFrame:
    """
    Run every task of ``job``. A task that fails is reported and the rest still run.

    Returns a table of the tasks: source, label, folder, status ("done", "exists", "failed: ...",
    or with ``dry_run`` "would run" / "exists"). Rebuilds the output folder's ``index.csv`` afterwards.
    """
    tasks = job.tasks()
    rows = []
    for n, task in enumerate(tasks, 1):
        out = find_output(task.source, task.chain, job.output, label=task.label, frames=job.frames, times=job.times)
        exists = is_complete(out, task.chain)
        row = {"source": task.source.name, "folder": str(out.relative_to(job.output)), **task.overrides}
        if dry_run:
            row["status"] = "exists" if exists and not force else "would run"
        else:
            if progress:
                print(f"[{n}/{len(tasks)}] {task.source.name} -> {out.name}")
            try:
                run_chain(task.source, task.chain, job.output, label=task.label, frames=job.frames,
                          times=job.times, preview=job.preview, force=force, progress=progress)
                row["status"] = "exists" if exists and not force else "done"
            except Exception as e:  # keep going: one bad file shouldn't stop a night's batch
                if progress:
                    traceback.print_exc()
                row["status"] = f"failed: {type(e).__name__}: {e}"
        rows.append(row)
    if not dry_run and job.output.exists():
        build_index(job.output)
    return pd.DataFrame(rows)


def stage_columns(chain) -> dict:
    """
    {"wiener.on": True, "binned_temporal_average.bin_size": 30, ...}; a stage that repeats gets
    "#2", "#3". ``<stage>.on`` marks the stages present, so a table can tell "wiener off" apart
    (fill the missing ones with ``fill_stage_columns``).
    """
    cols, seen = {}, {}
    for s in chain:
        name = s["stage"]
        seen[name] = seen.get(name, 0) + 1
        prefix = name if seen[name] == 1 else f"{name}#{seen[name]}"
        cols[f"{prefix}.on"] = True
        for k, v in s.items():
            if k != "stage":
                cols[f"{prefix}.{k}"] = v
    return cols


def fill_stage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """``<stage>.on`` is False where a row's chain doesn't have the stage."""
    for c in df.columns:
        if c.endswith(".on"):
            df[c] = df[c].astype("boolean").fillna(False).astype(bool)
    return df


def chain_summary(chain) -> str:
    """One-line description, e.g. "normalize > binned_temporal_average(bin_size=30) > to_uint16"."""
    parts = []
    for s in chain:
        settings = ", ".join(f"{k}={v}" for k, v in s.items() if k != "stage" and v is not None)
        parts.append(f"{s['stage']}({settings})" if settings else s["stage"])
    return " > ".join(parts)


def build_index(out_root) -> Path:
    """Scan ``<out_root>/<source>/<output>/frames.h5`` and write ``<out_root>/index.csv``."""
    out_root = Path(out_root)
    rows = []
    for path in sorted(out_root.glob(f"*/*/{OUTPUT_FILE}")):
        try:
            a = read_output_attrs(path)
        except OSError as e:
            print(f"skipping unreadable {path}: {e}")
            continue
        fitted = a.get("chain_fitted", a.get("chain", []))
        rows.append({
            "source": a.get("source_name", path.parent.parent.name),
            "folder": f"{path.parent.parent.name}/{path.parent.name}",
            "label": a.get("label", ""),
            "hash": a.get("hash", ""),
            "n_frames": a.get("n_frames"),
            "dt": a.get("dt"),
            "dtype": a.get("dtype", ""),
            "complete": bool(a.get("complete", False)),
            "preview": (path.parent / PREVIEW_FILE).exists(),
            "created": a.get("created", ""),
            "git_commit": a.get("git_commit", ""),
            "migrated_from": a.get("migrated_from", ""),
            "chain": chain_summary(a.get("chain", [])),
            **stage_columns(fitted),
        })
    index = out_root / INDEX_FILE
    fill_stage_columns(pd.DataFrame(rows)).to_csv(index, index=False)
    return index

