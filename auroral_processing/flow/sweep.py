from __future__ import annotations
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

from auroral_processing.flow.compute_flow import compute_flow
from auroral_processing.flow.params import TVL1Params, external_name
from auroral_processing.flow.io import open_flow


def run_name(settings: dict) -> str:
    """Folder name for one sweep run, e.g. {"gamma": 1, "lambda": 0.15} -> "gamma1_lambda0.15"."""
    return "_".join(f"{external_name(k)}{v:g}" for k, v in settings.items())


def summarise(flow_path) -> dict:
    """Mean and median of each per-pair metric in a flow file."""
    with open_flow(flow_path) as f:
        row = {"pairs_done": int(f["done"][:].sum()), "complete": bool(f.attrs["complete"])}
        for name, ds in f.get("metrics", {}).items():
            values = ds[:]
            row[f"{name}_mean"] = float(np.nanmean(values))
            row[f"{name}_median"] = float(np.nanmedian(values))
    return row


def run_sweep(
    src,
    grid: dict[str, list],
    outdir,
    *,
    base_params: TVL1Params | None = None,
    videos=("quiver",),
    video_kw=None,
    progress=True,
    **compute_kw,
) -> Path:
    """
    Run ``compute_flow`` for every combination of the TV-L1 settings in ``grid``.

    Each run writes ``outdir/<run name>/flow.h5`` (and ``flow_<kind>.mp4`` for each of ``videos``).
    Runs that are already complete are skipped, and partial runs are resumed. Writes
    ``outdir/summary.csv`` with each run's settings and mean/median quality metrics.

    Parameters
    ----------
    src : str or pathlib.Path
        Source HDF5 file.
    grid : dict[str, list]
        TV-L1 setting name -> values to try, e.g. ``{"gamma": [1, 3], "lambda": [0.05, 0.15, 0.4]}``.
    outdir : str or pathlib.Path
        Folder for the runs and the summary.
    base_params : TVL1Params, optional
        Settings not in ``grid``. OpenCV defaults if not given.
    videos : iterable of {"quiver", "speed"}
        Videos to render per run. Empty for none.
    video_kw : dict, optional
        Extra arguments for ``make_flow_video``.
    **compute_kw
        Passed to ``compute_flow`` (frames, times, gap, stride, start, stop, workers, metrics, ...).

    Returns
    -------
    pathlib.Path
        The summary CSV.
    """
    from auroral_processing.products.flow_video import make_flow_video

    outdir = Path(outdir)
    base_params = base_params or TVL1Params()
    names = list(grid)
    rows = []

    combos = list(product(*grid.values()))
    for n, values in enumerate(combos, 1):
        settings = dict(zip(names, values))
        params = base_params.replace(**settings)
        run_dir = outdir / run_name(settings)
        run_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            print(f"[{n}/{len(combos)}] {run_dir.name}")

        flow_path = compute_flow(src, run_dir / "flow.h5", params=params, progress=progress, **compute_kw)
        for kind in videos:
            video = run_dir / f"flow_{kind}.mp4"
            if not video.exists():  # render under a temporary name so an interrupted video isn't skipped later
                partial = video.with_suffix(".partial.mp4")
                make_flow_video(flow_path, partial, kind=kind, progress=progress, **(video_kw or {}))
                partial.rename(video)

        rows.append({"run": run_dir.name, **params.to_dict(), **summarise(flow_path)})

    summary = outdir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    return summary
