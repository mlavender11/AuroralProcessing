from __future__ import annotations
from itertools import product
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from auroral_processing.flow.compute_flow import compute_flow
from auroral_processing.flow.params import TVL1Params, external_name
from auroral_processing.flow.io import open_flow

PAIRING = ("gap", "stride")  # grid settings that change the frame pairs rather than TV-L1


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
        May also include ``"gap"`` and ``"stride"``. Swept gaps all use the same first frames:
        ``stride`` defaults to the largest gap and ``stop`` is shortened per gap, so runs with the
        same stride have the same pairs' i0. A
        ``video_kw`` ``vmax`` is then for the ``gap`` in ``compute_kw`` (default 1) and is scaled
        with the gap (and ``arrow_scale`` inversely), so colours and arrows mean the same px/s.
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
    gaps = [int(g) for g in grid.get("gap", [])]
    if gaps:
        ref_gap = compute_kw.pop("gap", 1) or 1
        if compute_kw.get("stride") is None:
            compute_kw["stride"] = max(gaps)
        with h5py.File(src, "r") as f:
            n_frames = f[compute_kw.get("frames", "rawimg")].shape[0]
        stop = min(compute_kw.get("stop") or n_frames, n_frames)

    combos = list(product(*grid.values()))
    for n, values in enumerate(combos, 1):
        settings = dict(zip(names, values))
        params = base_params.replace(**{k: v for k, v in settings.items() if k not in PAIRING})
        kw, vkw = dict(compute_kw), dict(video_kw or {})
        if "stride" in settings:
            kw["stride"] = int(settings["stride"])
        if gaps:
            gap = kw["gap"] = int(settings["gap"])
            kw["stop"] = stop - max(gaps) + gap
            if vkw.get("vmax"):
                vkw["vmax"] *= gap / ref_gap
            if vkw.get("arrow_scale"):
                vkw["arrow_scale"] *= ref_gap / gap
        run_dir = outdir / run_name(settings)
        run_dir.mkdir(parents=True, exist_ok=True)
        if progress:
            print(f"[{n}/{len(combos)}] {run_dir.name}")

        flow_path = compute_flow(src, run_dir / "flow.h5", params=params, progress=progress, **kw)
        for kind in videos:
            video = run_dir / f"flow_{kind}.mp4"
            if not video.exists():  # render under a temporary name so an interrupted video isn't skipped later
                partial = video.with_suffix(".partial.mp4")
                make_flow_video(flow_path, partial, kind=kind, progress=progress, **vkw)
                partial.rename(video)

        rows.append({"run": run_dir.name, "gap": kw.get("gap", 1), "stride": kw.get("stride") or kw.get("gap", 1), **params.to_dict(), **summarise(flow_path)})

    summary = outdir / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    return summary
