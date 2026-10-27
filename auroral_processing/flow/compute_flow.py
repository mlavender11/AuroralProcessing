from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import h5py
import cv2

from auroral_processing.flow.params import TVL1Params
from auroral_processing.flow.methods import ESTIMATORS, make_estimator
from auroral_processing.flow.io import FlowWriter
from auroral_processing.flow import metrics as M

INPUT_DTYPES = ("float32", "uint8")


@dataclass(frozen=True)
class FrameScaler:
    """
    Linear map of raw frames onto TV-L1's input range, with the same ``lo``/``hi`` for every frame
    (per-frame stretching would turn brightness changes into fake motion).

    OpenCV's TV-L1 only accepts uint8 (0-255) or float32 (0-1, scaled x255 internally).
    ``dtype="uint8"`` truncates like the old opencv-opt-flow.py prototype, for comparing to its sweeps.
    """

    lo: float
    hi: float
    dtype: str = "float32"

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        x = np.clip((frame.astype(np.float32) - self.lo) / (self.hi - self.lo), 0, 1).astype(np.float32)
        if self.dtype == "uint8":
            return (x * 255).astype(np.uint8)
        return x


def fit_scaler(ds, start, stop, percentiles=(0.5, 99.5), dtype="float32", n_sample=20) -> FrameScaler:
    """Choose ``lo``/``hi`` as percentiles of ``n_sample`` evenly spaced frames in [start, stop)."""
    idx = np.unique(np.linspace(start, stop - 1, n_sample).astype(int))
    lo, hi = np.percentile(ds[idx.tolist()], percentiles)
    if hi <= lo:
        raise ValueError(f"degenerate scaling: percentiles {percentiles} give lo={lo}, hi={hi}")
    return FrameScaler(float(lo), float(hi), dtype)


def make_pairs(n_frames, gap=1, stride=None, start=0, stop=None) -> np.ndarray:
    """(P, 2) array of (i0, i1) frame indices: i1 = i0 + gap, i0 steps by stride, both in [start, stop)."""
    stride = gap if stride is None else stride
    stop = n_frames if stop is None else min(stop, n_frames)
    if gap < 1 or stride < 1:
        raise ValueError(f"gap and stride must be >= 1, got gap={gap}, stride={stride}")
    i0 = np.arange(start, stop - gap, stride)
    return np.stack([i0, i0 + gap], axis=1)


def default_out_path(src) -> Path:
    src = Path(src)
    return src.with_name(f"{src.stem}_flow.h5")


def _to_unit(img):
    """Image as float32 0-1 for metrics, whichever dtype TV-L1 was given."""
    return img.astype(np.float32) / 255 if img.dtype == np.uint8 else img


def iter_flow(src, frames, pairs, ks, params, scaler, metrics=M.DEFAULT_METRICS, warm_start=False):
    """
    Generator: compute the flow for pairs ``ks`` (indices into ``pairs``), yielding
    ``(k, flow (H, W, 2) float32, {metric: value})``. The metrics include the method's own
    (e.g. PIV/FLCT ``invalid_frac``).

    With ``warm_start`` each pair starts from the previous pair's flow when they are consecutive.
    """
    est = make_estimator(params)
    back = make_estimator(params) if "fb_error" in metrics else None
    cache = {}  # the last two scaled frames; with stride == gap each pair's i1 is the next pair's i0
    prev_k, prev_flow = None, None

    with h5py.File(src, "r") as f:
        ds = f[frames]

        def load(i):
            if i not in cache:
                if len(cache) >= 2:
                    cache.pop(next(iter(cache)))
                cache[i] = scaler(ds[i])
            return cache[i]

        for k in ks:
            i0, i1 = (int(i) for i in pairs[k])
            a, b = load(i0), load(i1)
            init = prev_flow.copy() if warm_start and prev_k == k - 1 else None
            flow = est.calc(a, b, init)

            m = dict(est.last)
            if "warp_mae" in metrics:
                m["warp_mae"] = M.warp_mae(_to_unit(a), _to_unit(b), flow)
            if "zero_mae" in metrics:
                m["zero_mae"] = M.zero_mae(_to_unit(a), _to_unit(b))
            if back is not None:
                m["fb_error"] = M.fb_error(flow, back.calc(b, a, None))

            yield k, flow, m
            prev_k, prev_flow = k, flow


def _init_worker(threads):
    cv2.setNumThreads(threads)  # share the cores between workers instead of each using all of them


def _run_chunk(*args):
    return list(iter_flow(*args))


def compute_flow(
    src,
    out=None,
    *,
    frames="rawimg",
    times="ut1_unix",
    params: TVL1Params | None = None,
    gap=1,
    stride=None,
    start=0,
    stop=None,
    clip_percentiles=(0.5, 99.5),
    input_dtype="float32",
    metrics=M.DEFAULT_METRICS,
    warm_start=False,
    workers=1,
    chunk_size=None,
    overwrite=False,
    progress=True,
) -> Path:
    """
    Run TV-L1 optical flow over frame pairs of an HDF5 frame dataset and write a flow file.

    Parameters
    ----------
    src : str or pathlib.Path
        Source HDF5 file.
    out : str or pathlib.Path, optional
        Output flow file. Defaults to ``<src stem>_flow.h5`` next to the source.
    frames, times : str
        Frame dataset (N, H, W) and time dataset (N,) in ``src``.
    params : TVL1Params, PIVParams or FLCTParams, optional
        The method and its settings; TV-L1 with OpenCV defaults if not given.
    gap : int
        Frames between the two images of a pair (i1 = i0 + gap).
    stride : int, optional
        Frames between successive pairs. Defaults to ``gap``.
    start, stop : int, optional
        Frame index range [start, stop) to use.
    clip_percentiles : (float, float)
        Percentiles of a frame sample used as the global ``lo``/``hi`` input scaling.
    input_dtype : {"float32", "uint8"}
        Image type given to TV-L1. "uint8" reproduces the old prototype's quantised input.
        PIV and FLCT always get float 0-1.
    metrics : iterable of str
        Per-pair quality metrics to store, from ``metrics.METRICS``. "fb_error" doubles the run time.
    warm_start : bool
        Start each pair from the previous pair's flow (TV-L1 only). Only applies within a chunk when
        ``workers > 1``.
    workers : int
        Processes to run in parallel. 1 runs in this process. Scripts using workers > 1 must call
        this under ``if __name__ == "__main__":`` (macOS starts workers by re-importing the script).
    chunk_size : int, optional
        Pairs per parallel task. Defaults to about 4 tasks per worker, to keep workers evenly loaded.
    overwrite : bool
        Replace an existing output with different settings. An existing output with the same
        settings is resumed: pairs already written are skipped.

    Returns
    -------
    pathlib.Path
        The flow file.
    """
    src = Path(src)
    out = Path(out) if out else default_out_path(src)
    params = params or TVL1Params()
    metrics = tuple(metrics)
    if input_dtype not in INPUT_DTYPES:
        raise ValueError(f"input_dtype must be one of {INPUT_DTYPES}, got {input_dtype!r}")
    if unknown := set(metrics) - set(M.METRICS):
        raise ValueError(f"unknown metrics {unknown}; allowed: {M.METRICS}")
    est_cls = ESTIMATORS[params.METHOD]
    if warm_start and not est_cls.supports_init:
        raise ValueError(f"warm_start is only supported for TV-L1, not {params.LABEL}")
    stored_metrics = metrics + tuple(m for m in est_cls.extra_metrics if m not in metrics)

    with h5py.File(src, "r") as f:
        ds, t = f[frames], f[times][:]
        if ds.ndim != 3 or len(t) != ds.shape[0]:
            raise ValueError(f"expected frames (N, H, W) and times (N,), got {ds.shape} and {t.shape}")
        stop = ds.shape[0] if stop is None else min(stop, ds.shape[0])
        pairs = make_pairs(ds.shape[0], gap, stride, start, stop)
        if len(pairs) == 0:
            raise ValueError(f"no frame pairs with gap={gap} in [{start}, {stop})")
        scaler = fit_scaler(ds, start, stop, clip_percentiles, input_dtype)
        shape = ds.shape[1:]

    config = dict(
        source=str(src.resolve()),
        frames=frames,
        times=times,
        gap=gap,
        stride=gap if stride is None else stride,
        start=start,
        stop=stop,
        clip_percentiles=list(clip_percentiles),
        input_dtype=input_dtype,
        metrics=list(metrics),
        warm_start=warm_start,
        **{params.METHOD: params.to_dict()},
    )
    if params.METHOD != "tvl1":  # TV-L1 configs stay exactly as before, so old runs still resume
        config["method"] = params.METHOD
    writer = FlowWriter(
        out,
        config=config,
        pairs=pairs,
        t0=t[pairs[:, 0]],
        t1=t[pairs[:, 1]],
        shape=shape,
        metrics=stored_metrics,
        scaling=dict(lo=scaler.lo, hi=scaler.hi),
        overwrite=overwrite,
    )
    with writer:
        scaler = FrameScaler(**writer.scaling, dtype=input_dtype)  # a resumed run keeps its original scaling
        todo = np.flatnonzero(~writer.done)
        bar = tqdm(total=len(pairs), initial=len(pairs) - len(todo), desc=out.stem, unit="pair", disable=not progress)
        args = (src, frames, pairs)

        if workers <= 1:
            for k, flow, m in iter_flow(*args, todo, params, scaler, metrics, warm_start):
                writer.write(k, flow, m)
                bar.update()
        else:
            # OpenCV's own threading speeds one pair up ~4x, so each worker keeps a share of the cores
            threads = max(1, cv2.getNumberOfCPUs() // workers)
            size = chunk_size or max(1, -(-len(todo) // (4 * workers)))
            chunks = [todo[i : i + size] for i in range(0, len(todo), size)]
            with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(threads,)) as pool:
                futures = [pool.submit(_run_chunk, *args, c, params, scaler, metrics, warm_start) for c in chunks]
                for fut in as_completed(futures):
                    for k, flow, m in fut.result():
                        writer.write(k, flow, m)
                        bar.update()
        bar.close()
        writer.finish()
    return out
