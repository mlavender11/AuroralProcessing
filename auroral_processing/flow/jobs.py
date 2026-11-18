"""
Optical flow from YAML: flow settings files and flow job files (``aurora flow run``).

Flow settings file (reusable; its label defaults to the file name)::

    method: tvl1                 # tvl1 | piv | flct
    params: {lambda: 0.15}       # names as in the --tvl1-* / --piv-* / --flct-* flags
    gap: 1                       # optional pairing and input settings
    stride: null
    warm_start: false
    clip: [0.5, 99.5]
    input_dtype: float32

Flow job file (the job's name is the file name)::

    output: ../flow              # parent of the job folder; default optical_flow/flow
    inputs:                      # files, folders, globs: pipeline outputs or raw sources
      - ../filtered/*/*/frames.h5
    frames: auto                 # auto: frames/frames_ut1_unix in pipeline outputs, else rawimg/ut1_unix
    times: auto
    start: 0
    stop: null
    flows:                       # flow settings files, or inline settings
      - ../flows/tvl1_default.yaml
      - {label: piv16, method: piv, params: {window: 16}}
    grid:                        # a key applies to every flow whose method has it; gap/stride to all
      lambda: [0.05, 0.15]
    metrics: [warp_mae, zero_mae, fb_error]
    workers: 1
    video: []                    # quiver and/or speed
    video_options: {}            # make_flow_video settings, e.g. {grid: 32, fps: 10}
    score: false                 # also run score_sweep on the job's runs

Where results go: everything is in the job folder ``<output>/<job>/``, nothing next to the inputs.

* a pipeline output ``.../<source>/<output>/frames.h5``: ``<job folder>/<source>/<output>/<run>/flow.h5``
* any other file ``.../clip.h5``: ``<job folder>/clip/raw/<run>/flow.h5``

Videos go in ``<job folder>/videos/<input>-<run>_<kind>.mp4``.

The job folder also holds ``job.yaml`` (a copy), ``runs.csv`` (status of the last ``run_flow_job``)
and ``summary.csv`` (one row per run folder found in the job folder, with its pre-processing, flow
settings and mean/median metrics, rebuilt from the flow files). There are no links, so a job folder
can be copied anywhere (rsync, scp, a USB drive) as plain files.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
import os
import shutil
import traceback
import h5py
import numpy as np
import pandas as pd

from auroral_processing.flow import metrics as M
from auroral_processing.flow.compute_flow import INPUT_DTYPES, compute_flow
from auroral_processing.flow.io import open_flow, read_config, read_method
from auroral_processing.flow.params import METHODS, external_name, params_from_dict
from auroral_processing.flow.sweep import PAIRING, expand_flow_grid, run_name, summarise
from auroral_processing.pipeline.config import _as_list, _resolve, load_yaml, resolve_sources

DEFAULT_FLOW_OUTPUT = Path(__file__).resolve().parents[2] / "optical_flow" / "flow"
FLOW_KEYS = {"label", "method", "params", "gap", "stride", "warm_start", "clip", "input_dtype"}
JOB_KEYS = {"output", "inputs", "frames", "times", "start", "stop", "flows", "grid", "metrics", "workers",
            "video", "video_options", "score"}
VIDEO_KINDS = ("quiver", "speed")
RUNS_FILE, SUMMARY_FILE, JOB_COPY = "runs.csv", "summary.csv", "job.yaml"
VIDEOS_DIR = "videos"


# ---------- flow settings ----------
@dataclass
class FlowSpec:
    """One flow recipe: the method and its settings, plus pairing and input scaling."""

    label: str
    params: object  # TVL1Params, PIVParams or FLCTParams
    gap: int = 1
    stride: int | None = None
    warm_start: bool = False
    clip: tuple = (0.5, 99.5)
    input_dtype: str = "float32"

    def has(self, key) -> bool:
        """True if grid/--set key ``key`` applies to this flow."""
        return key in PAIRING or external_name(key) in self.params.to_dict()

    def with_settings(self, settings: dict) -> FlowSpec:
        """Copy with flow settings and/or gap/stride changed."""
        pairing = {k: (None if v is None else int(v)) for k, v in settings.items() if k in PAIRING}
        params = {external_name(k): v for k, v in settings.items() if k not in PAIRING}
        return replace(self, params=self.params.replace(**params) if params else self.params, **pairing)

    def compute_kw(self) -> dict:
        return dict(gap=self.gap, stride=self.stride, warm_start=self.warm_start,
                    clip_percentiles=tuple(self.clip), input_dtype=self.input_dtype)


def load_flow(spec, base_dir=None) -> FlowSpec:
    """A flow from a settings file path, or an inline mapping (``{"file": path, "label": ...}`` also works)."""
    base_dir = Path(base_dir or ".")
    label = None
    if isinstance(spec, (str, Path)):
        path = _resolve(spec, base_dir)
        data = load_yaml(path) or {}
        label = path.stem
        where = str(path)
    elif isinstance(spec, dict) and "file" in spec:
        if unknown := set(spec) - {"file", "label"}:
            raise ValueError(f"flow entry: with 'file', only 'label' may be given, not {sorted(unknown)}")
        inner = load_flow(spec["file"], base_dir)
        return replace(inner, label=str(spec.get("label", inner.label)))
    elif isinstance(spec, dict):
        data, where = spec, "inline flow"
    else:
        raise ValueError(f"a flow must be a settings file or a mapping, got {spec!r}")
    if not isinstance(data, dict):
        raise ValueError(f"{where}: a flow must be a mapping (method, params, ...)")
    if unknown := set(data) - FLOW_KEYS:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(FLOW_KEYS)}")
    method = data.get("method", "tvl1")
    if method not in METHODS:
        raise ValueError(f"{where}: unknown method {method!r}; allowed: {sorted(METHODS)}")
    params = params_from_dict(method, data.get("params") or {})
    label = str(data.get("label", label or method))
    gap = int(data.get("gap", 1))
    stride = data.get("stride")
    clip = tuple(float(c) for c in data.get("clip", (0.5, 99.5)))
    input_dtype = data.get("input_dtype", "float32")
    if gap < 1 or (stride is not None and int(stride) < 1):
        raise ValueError(f"{where}: gap and stride must be >= 1")
    if len(clip) != 2 or not 0 <= clip[0] < clip[1] <= 100:
        raise ValueError(f"{where}: clip must be two percentiles, low < high, got {clip}")
    if input_dtype not in INPUT_DTYPES:
        raise ValueError(f"{where}: input_dtype must be one of {INPUT_DTYPES}, got {input_dtype!r}")
    return FlowSpec(label, params, gap, None if stride is None else int(stride), bool(data.get("warm_start", False)),
                    clip, input_dtype)


# ---------- inputs ----------
def is_pipeline_output(path) -> bool:
    """True for a ``frames.h5`` written by ``run_chain``."""
    try:
        with h5py.File(path, "r") as f:
            return "chain" in f.attrs and "frames" in f
    except OSError:
        return False


def auto_datasets(path) -> tuple[str, str]:
    """(frames, times) dataset names: pipeline outputs use frames/frames_ut1_unix, raw files rawimg/ut1_unix."""
    with h5py.File(path, "r") as f:
        if "frames" in f and "frames_ut1_unix" in f:
            return "frames", "frames_ut1_unix"
    return "rawimg", "ut1_unix"


# ---------- jobs ----------
@dataclass
class FlowRun:
    """One input, one flow, one grid combination."""

    input: Path
    flow: FlowSpec
    settings: dict  # grid values
    name: str  # run folder name
    tag: str  # <input>-<run>, unique within the job: used to name videos
    folder: Path
    params: object
    compute_kw: dict
    video_kw: dict


@dataclass
class FlowJob:
    name: str
    inputs: list[Path]
    flows: list[FlowSpec]
    output: Path = DEFAULT_FLOW_OUTPUT
    frames: str = "auto"
    times: str = "auto"
    start: int = 0
    stop: int | None = None
    grid: dict = field(default_factory=dict)
    sets: dict = field(default_factory=dict)  # overrides for every flow that has the setting (--set)
    metrics: tuple = M.METRICS  # warp_mae, zero_mae, fb_error
    workers: int = 1
    video: list = field(default_factory=list)
    video_options: dict = field(default_factory=dict)
    score: bool = False
    source_text: str | None = None  # the job file's text, copied into the job folder
    note: str = ""  # command-line additions, recorded in the copy

    @property
    def folder(self) -> Path:
        return Path(self.output) / self.name

    def check(self):
        """Errors that should stop a job before anything runs."""
        for what, keys in (("--set", self.sets), ("grid", self.grid)):
            if unused := [k for k in keys if not any(f.has(k) for f in self.flows)]:
                raise ValueError(f"{what} {unused}: no flow has that setting (check the spelling)")
        if unknown := set(self.metrics) - set(M.METRICS):
            raise ValueError(f"unknown metrics {sorted(unknown)}; allowed: {list(M.METRICS)}")
        if unknown := set(self.video) - set(VIDEO_KINDS):
            raise ValueError(f"unknown video kinds {sorted(unknown)}; allowed: {list(VIDEO_KINDS)}")
        labels = [f.label for f in self.flows]
        if len(labels) != len(set(labels)):
            raise ValueError(f"flow labels must be unique, got {labels}; give inline flows a label")

    def datasets(self, path) -> tuple[str, str]:
        frames, times = auto_datasets(path) if "auto" in (self.frames, self.times) else (None, None)
        return (frames if self.frames == "auto" else self.frames), (times if self.times == "auto" else self.times)

    def runs(self) -> list[FlowRun]:
        """inputs x flows x grid combinations, each with its folder and compute_flow arguments."""
        self.check()
        out, tags, folders = [], {}, {}
        for path in self.inputs:
            frames, times = self.datasets(path)
            pipeline = is_pipeline_output(path)
            for flow in self.flows:
                flow = flow.with_settings({k: v for k, v in self.sets.items() if flow.has(k)})
                grid = {k: v for k, v in self.grid.items() if flow.has(k)}
                kw = dict(frames=frames, times=times, start=self.start, stop=self.stop, metrics=tuple(self.metrics),
                          workers=self.workers, **flow.compute_kw())
                for settings, params, ckw, vkw in expand_flow_grid(path, grid, flow.params, kw, self.video_options):
                    name = flow.label + (f"_{run_name(settings)}" if settings else "")
                    if pipeline:  # <job>/<source>/<output>/<run>
                        folder, base = self.folder / path.parent.parent.name / path.parent.name / name, path.parent.name
                    else:
                        folder, base = self.folder / path.stem / "raw" / name, path.stem
                    if folders.setdefault(folder, path) != path:
                        raise ValueError(f"{folders[folder]} and {path} would share the run folder {folder}; "
                                         "give them different names or put them in separate jobs")
                    tag = _unique(f"{base}-{name}", path, tags)
                    out.append(FlowRun(path, flow, settings, name, tag, folder, params, ckw, vkw))
        return out


def _unique(name, path, taken: dict) -> str:
    """``name``, prefixed with parent folder names if another input already uses it."""
    candidate, parents = name, list(path.resolve().parents)
    i = 1 if path.name == "frames.h5" else 0  # a pipeline output's own folder is already in the name
    while candidate in taken and taken[candidate] != path:
        if i >= len(parents) - 1:
            raise ValueError(f"can't give {path} a distinct run name ({name})")
        candidate = f"{parents[i].name}-{candidate}"
        i += 1
    taken[candidate] = path
    return candidate


def load_flow_job(path) -> FlowJob:
    """A flow job from a YAML job file. Relative paths in it are relative to the file's folder."""
    path = Path(path)
    data = load_yaml(path) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a flow job file must be a mapping")
    if unknown := set(data) - JOB_KEYS:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(JOB_KEYS)}")
    for key in ("inputs", "flows"):
        if not data.get(key):
            raise ValueError(f"{path}: '{key}' is required")
    base = path.parent
    grid = data.get("grid") or {}
    if not isinstance(grid, dict):
        raise ValueError(f"{path}: grid must be a mapping of NAME: [values]")
    return FlowJob(
        name=path.stem,
        inputs=resolve_sources(data["inputs"], base),
        flows=[load_flow(f, base) for f in _as_list(data["flows"])],
        output=_resolve(data["output"], base).resolve() if data.get("output") else DEFAULT_FLOW_OUTPUT,
        frames=data.get("frames", "auto"),
        times=data.get("times", "auto"),
        start=int(data.get("start") or 0),
        stop=None if data.get("stop") is None else int(data["stop"]),
        grid={k: _as_list(v) for k, v in grid.items()},
        metrics=tuple(_as_list(data["metrics"]) if data.get("metrics") is not None else M.METRICS),
        workers=int(data.get("workers") or 1),
        video=_as_list(data.get("video") or []),
        video_options=dict(data.get("video_options") or {}),
        score=bool(data.get("score", False)),
        source_text=path.read_text(),
    )


# ---------- running ----------
def _complete(flow_path) -> bool:
    try:
        with open_flow(flow_path) as f:
            return bool(f.attrs.get("complete", False))
    except OSError:
        return False


def _same_settings(flow_path, run: FlowRun) -> bool:
    """Whether an existing flow file was made with this run's settings (the ones a dry run can check)."""
    with open_flow(flow_path) as f:
        cfg = read_config(f)
    method, settings = read_method(cfg)
    kw = run.compute_kw
    return (method == run.params.METHOD and settings == run.params.to_dict() and cfg["frames"] == kw["frames"]
            and cfg["gap"] == kw["gap"] and cfg["stride"] == (kw["stride"] or kw["gap"])
            and cfg["metrics"] == list(kw["metrics"]) and cfg["start"] == kw["start"])


def run_flow_job(job: FlowJob, *, dry_run=False, force=False, progress=True) -> pd.DataFrame:
    """
    Run every flow run of ``job``. A run that fails is reported and the rest still run.

    Returns a table of the runs: input, flow, run, folder and status ("done", "exists", "failed: ...",
    or with ``dry_run`` "would run" / "exists" / "partial" / "exists, different settings").
    Afterwards writes the job folder: ``job.yaml``, ``runs.csv`` and ``summary.csv``.
    """
    from auroral_processing.products.flow_video import make_flow_video

    runs = job.runs()
    rows = []
    for n, run in enumerate(runs, 1):
        flow_path = run.folder / "flow.h5"
        row = {"input": _input_name(run.input), "flow": run.flow.label, "run": run.name, **run.settings,
               "folder": run.folder.relative_to(job.folder).as_posix()}
        complete = flow_path.exists() and _complete(flow_path)
        if dry_run:
            if not flow_path.exists() or force:
                row["status"] = "would run"
            elif not _same_settings(flow_path, run):
                row["status"] = "exists, different settings"
            else:
                row["status"] = "exists" if complete else "partial"
            rows.append(row)
            continue
        if progress:
            print(f"[{n}/{len(runs)}] {row['input']} -> {run.name}")
        try:
            compute_flow(run.input, flow_path, params=run.params, overwrite=force, progress=progress, **run.compute_kw)
            for kind in job.video:
                video = job.folder / VIDEOS_DIR / f"{run.tag}_{kind}.mp4"
                old = run.folder / f"flow_{kind}.mp4"  # where videos used to go
                video.parent.mkdir(parents=True, exist_ok=True)
                if force:
                    video.unlink(missing_ok=True)
                    old.unlink(missing_ok=True)
                elif old.exists() and not video.exists():
                    shutil.move(old, video)
                if not video.exists():  # render under a temporary name so an interrupted video isn't skipped later
                    partial = video.with_suffix(".partial.mp4")
                    make_flow_video(flow_path, partial, kind=kind, progress=progress, **run.video_kw)
                    partial.rename(video)
            row["status"] = "exists" if complete and not force else "done"
        except Exception as e:  # keep going: one bad input shouldn't stop the rest
            if progress:
                traceback.print_exc()
            row["status"] = f"failed: {type(e).__name__}: {e}"
        rows.append(row)

    table = pd.DataFrame(rows)
    if not dry_run:
        job.folder.mkdir(parents=True, exist_ok=True)
        if job.source_text is not None:
            (job.folder / JOB_COPY).write_text(job.source_text + (f"\n# {job.note}\n" if job.note else ""))
        table.to_csv(job.folder / RUNS_FILE, index=False)
        if run_folders(job.folder):
            build_summary(job.folder)
            if job.score:
                score_job(job.folder, progress=progress)
    return table


def _input_name(path) -> str:
    return path.parent.name if path.name == "frames.h5" else path.stem


def run_folders(job_folder) -> list[str]:
    """Every run in a job folder: the folders holding a flow.h5, relative to the job folder, sorted."""
    job_folder = Path(job_folder)
    found = []
    for root, dirs, files in os.walk(job_folder):  # doesn't follow links, e.g. an old runs/ folder
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != VIDEOS_DIR)
        if "flow.h5" in files:
            found.append(Path(root).relative_to(job_folder).as_posix())
    return sorted(found)


def build_summary(job_folder) -> Path:
    """
    ``summary.csv`` of a job folder: one row per run folder in it (``run`` is its path inside the job
    folder) with the input's pre-processing (chain and ``<stage>.<setting>`` columns), the flow
    settings, seconds per pair, and the mean and median of each metric, plus ``warp_over_zero``
    (mean warp_mae / mean zero_mae; lower is better). A deleted run folder simply drops out.
    """
    from auroral_processing.flow.score import _preprocessing
    from auroral_processing.pipeline.jobs import fill_stage_columns

    job_folder = Path(job_folder)
    rows = []
    for name in run_folders(job_folder):
        flow_path = job_folder / name / "flow.h5"
        with open_flow(flow_path) as f:
            cfg = read_config(f)
            dt = float(np.median(f["t1"][:] - f["t0"][:]))
        method, settings = read_method(cfg)
        row = {"run": name, "status": "complete" if _complete(flow_path) else "partial",
               **_preprocessing(cfg["source"]), "method": method, **settings,
               "gap": cfg["gap"], "stride": cfg["stride"], "dt": dt, **summarise(flow_path)}
        if row.get("warp_mae_mean") and row.get("zero_mae_mean"):
            row["warp_over_zero"] = row["warp_mae_mean"] / row["zero_mae_mean"]
        rows.append(row)
    df = fill_stage_columns(pd.DataFrame(rows))
    first = [c for c in ("run", "status", "output", "chain", "method", "warp_over_zero", "warp_mae_mean",
                         "zero_mae_mean", "fb_error_mean") if c in df]
    df = df[first + [c for c in df.columns if c not in first]]
    out = job_folder / SUMMARY_FILE
    df.to_csv(out, index=False)
    return out


def score_job(job_folder, progress=True) -> Path:
    """
    ``aurora flow score`` on a job's runs (the ``run`` column of its summary.csv); ``scores.csv``
    and the heatmaps go in the job folder.
    """
    from auroral_processing.flow.score import score_sweep

    return score_sweep(job_folder, plot=True, progress=progress)
