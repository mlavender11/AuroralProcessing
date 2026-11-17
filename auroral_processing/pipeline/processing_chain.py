"""
Run a pre-processing chain on a source HDF5 file.

``run_chain`` writes each result to its own output folder, never touching the source:

    <out_root>/<source stem>/<label>-<hash8>/
        frames.h5      frames (N, H, W) + frames_ut1_unix (N,)
        preview.mp4    optional

``hash8`` identifies the chain (settings with defaults filled in, before fitting) and the source
file name and datasets, so the same chain on the same source always maps to the same folder and is
skipped when it already exists, and a different chain can never overwrite it. ``frames.h5`` attrs
record everything needed to reproduce it: ``chain`` and ``chain_fitted`` (JSON), the source,
``git_commit`` and ``created``.

``apply_processing`` is the older interface that writes a dataset into the source file itself.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import os
import re
import subprocess
import warnings
import h5py
import numpy as np
from auroral_processing.pipeline.drivers import write_h5, read_h5
from auroral_processing.pipeline.stages import Stage

FRAMES, TIMES = "frames", "frames_ut1_unix"
OUTPUT_FILE, PREVIEW_FILE = "frames.h5", "preview.mp4"

IMAGE_ATTRS = {
    "CLASS": np.bytes_("IMAGE"),
    "IMAGE_VERSION": np.bytes_("1.2"),
    "IMAGE_SUBCLASS": np.bytes_("IMAGE_GRAYSCALE"),
    "DISPLAY_ORIGIN": np.bytes_("UL"),
    "IMAGE_WHITE_IS_ZERO": np.uint8(0),
}


# ---------- building a chain ----------
def build_stages(chain):
    """
    Stage objects from a chain of Stage objects and/or dicts. Dict entries with ``enabled: false``
    are left out. The inputs are not modified.
    """
    stages = []
    for s in chain:
        if isinstance(s, Stage):
            stages.append(copy.deepcopy(s))
        elif s.get("enabled", True) not in (False, "off", "false"):
            stages.append(Stage.from_dict(s))
    return stages


def canonical_chain(chain) -> list[dict]:
    """The chain as plain dicts with every setting filled in (defaults included), disabled stages removed."""
    return [s.to_dict() for s in build_stages(chain)]


def chain_json(chain) -> str:
    return json.dumps(chain, sort_keys=True)


def chain_hash(chain, source_name, frames="rawimg", times="ut1_unix") -> str:
    """8-character id of a chain on a source: the canonical chain plus the source file name and datasets."""
    key = {"chain": canonical_chain(chain), "source": Path(source_name).name, "frames": frames, "times": times}
    return hashlib.sha1(chain_json(key).encode()).hexdigest()[:8]


def auto_label(chain) -> str:
    """Readable name from the stages' short tags, e.g. "norm_bin30_med3_wiener"."""
    return "_".join(t for t in (s.label() for s in build_stages(chain)) if t) or "chain"


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "chain"


def output_dir(src, chain, out_root, *, label=None, frames="rawimg", times="ut1_unix") -> Path:
    """Folder ``run_chain`` writes to for this source and chain."""
    src = Path(src)
    name = f"{_safe(label or auto_label(chain))}-{chain_hash(chain, src.name, frames, times)}"
    return Path(out_root) / src.stem / name


def connect_stages(stages, src):
    """Needs a driver at the end - e.g. write_h5"""
    for stage in stages:
        src = stage.run(src)
    return src


def num_output_frames(stages, n):
    for stage in stages:
        n = stage.n_out(n)
    return n


def fit_stages(stages, frames, times):
    """
    Let each stage estimate its settings (e.g. normalize's vmin/vmax) from the source frames.

    Fitting sees the source frames, not the output of earlier stages, so a stage that needs fitting
    must come first in the chain (or be given its settings explicitly).
    """
    for i, stage in enumerate(stages):
        if stage.needs_fit():
            if i > 0:
                raise ValueError(
                    f"stage {i} ({stage.name}) estimates its settings from the source frames, so it must be "
                    f"the first stage; move it first or give its settings explicitly (e.g. vmin/vmax)"
                )
            stage.fit(frames, times)
    return stages


# ---------- output folders ----------
def git_commit() -> str:
    """Commit of this package's repository, with "-dirty" if it has uncommitted changes."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_output_attrs(path) -> dict:
    """Attributes of an output ``frames.h5`` (or its folder), with the JSON ones parsed."""
    path = Path(path)
    if path.is_dir():
        path = path / OUTPUT_FILE
    with h5py.File(path, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        attrs["n_frames"] = int(f[FRAMES].shape[0]) if FRAMES in f else 0
        attrs["dtype"] = str(f[FRAMES].dtype) if FRAMES in f else ""
        if TIMES in f and attrs["n_frames"] > 1:
            attrs["dt"] = float(np.median(np.diff(f[TIMES][:])))
    for k in ("chain", "chain_fitted"):
        if k in attrs:
            attrs[k] = json.loads(attrs[k])
    return attrs


def is_complete(out_dir, chain=None) -> bool:
    """True if ``out_dir`` holds a finished output (of ``chain``, if given)."""
    path = Path(out_dir) / OUTPUT_FILE
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            if not f.attrs.get("complete", False):
                return False
            return chain is None or f.attrs.get("chain") == chain_json(canonical_chain(chain))
    except OSError:
        return False


def write_output(out_dir, stream, *, total_frames, chain, chain_fitted, source, frames, times, label,
                 extra_attrs=None, progress=True) -> Path:
    """
    Drain ``stream`` into ``out_dir/frames.h5``. Written as ``frames.partial.h5`` and renamed when
    complete, so an interrupted run never looks finished.

    ``chain_fitted`` may be a callable, called after the stream is drained (stages such as
    ``wiener noise: auto`` only know their settings then).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final, partial = out_dir / OUTPUT_FILE, out_dir / "frames.partial.h5"
    source = Path(source)

    with h5py.File(source, "r") as s:
        src_time_attrs = dict(s[times].attrs) if times in s else {}

    with h5py.File(partial, "w") as f:
        n = write_h5(stream, f, FRAMES, total_frames=total_frames, time_name=TIMES, overwrite=True, progress=progress)
        if n == 0:
            raise ValueError(f"the chain produced no frames from {source}:{frames}")
        fitted = chain_fitted() if callable(chain_fitted) else chain_fitted
        write_attrs(f[FRAMES], f[TIMES], src_time_attrs, f"{source.name}:{frames}", fitted, source_times=times)
        a = f.attrs
        a["chain"] = chain_json(chain)
        a["chain_fitted"] = chain_json(fitted)
        a["label"] = label
        a["hash"] = out_dir.name.rsplit("-", 1)[-1]
        a["source"] = str(source.resolve())
        a["source_name"] = source.name
        a["source_frames"] = frames
        a["source_times"] = times
        a["git_commit"] = git_commit()
        a["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for k, v in (extra_attrs or {}).items():
            a[k] = v
        a["complete"] = True
    os.replace(partial, final)
    return final


def render_preview(out_dir, force=False) -> Path | None:
    """``preview.mp4`` of an output folder (uint16 outputs only, as the video's scaling needs them)."""
    from auroral_processing.products.video import make_video_from_source

    out_dir = Path(out_dir)
    video = out_dir / PREVIEW_FILE
    if video.exists() and not force:
        return video
    with h5py.File(out_dir / OUTPUT_FILE, "r") as f:
        dtype = f[FRAMES].dtype
    if dtype != np.uint16:
        warnings.warn(f"no preview for {out_dir.name}: frames are {dtype}, previews need uint16 (end the chain in to_uint16)")
        return None
    partial = out_dir / "preview.partial.mp4"
    make_video_from_source(out_dir / OUTPUT_FILE, out_fn=partial.name, img_dset=FRAMES, ut_dset=TIMES)
    os.replace(partial, video)
    return video


def run_chain(
    src,
    chain,
    out_root,
    *,
    label=None,
    frames="rawimg",
    times="ut1_unix",
    preview=False,
    force=False,
    progress=True,
) -> Path:
    """
    Run a pre-processing chain on one source file and write its own output folder.

    Parameters
    ----------
    src : str or pathlib.Path
        Source HDF5 file. It is only read.
    chain : list of dict or Stage
        Stages in order, e.g. ``[{"stage": "normalize"}, {"stage": "binned_temporal_average", "bin_size": 30}]``.
    out_root : str or pathlib.Path
        Output folder; the result goes to ``<out_root>/<source stem>/<label>-<hash8>/``.
    label : str, optional
        Readable start of the folder name. Defaults to one built from the stages, e.g. "norm_bin30_med3".
    frames, times : str
        Frame and time datasets in ``src``.
    preview : bool
        Also render ``preview.mp4``.
    force : bool
        Rebuild even if the output already exists.

    Returns
    -------
    pathlib.Path
        The output folder.
    """
    src = Path(src)
    canonical = canonical_chain(chain)
    label = _safe(label or auto_label(chain))
    out_dir = output_dir(src, chain, out_root, label=label, frames=frames, times=times)

    if is_complete(out_dir, chain) and not force:
        if progress:
            print(f"exists, skipping: {out_dir}")
    else:
        stages = build_stages(chain)
        with h5py.File(src, "r") as f:
            ds, ut = f[frames], f[times]
            fit_stages(stages, ds, ut)
            write_output(
                out_dir,
                connect_stages(stages, read_h5(ds, ut)),
                total_frames=num_output_frames(stages, len(ds)),
                chain=canonical,
                chain_fitted=lambda: [s.to_dict() for s in stages],
                source=src,
                frames=frames,
                times=times,
                label=label,
                progress=progress,
            )
        if progress:
            print(f"wrote {out_dir}")
    if preview:
        render_preview(out_dir, force=force)
    return out_dir


# ---------- older interface: write into the source file ----------
def apply_processing(hdf_fn, chain, iteration, frames_name="rawimg", times_name="ut1_unix", overwrite=False):
    """
    Run ``chain`` and write the result as dataset ``iteration`` (+ ``<iteration>_ut1_unix``) in the
    source file itself. Prefer ``run_chain``, which leaves the source untouched.
    """
    stages = build_stages(chain)

    with h5py.File(hdf_fn, "r+") as f:
        frames = f[frames_name]
        ut = f[times_name]
        fit_stages(stages, frames, ut)

        write_h5(
            connect_stages(stages, read_h5(frames, ut)),
            f,
            iteration,
            total_frames=num_output_frames(stages, len(frames)),
            overwrite=overwrite,
        )
        write_attrs(f[iteration], f[f"{iteration}_ut1_unix"], dict(ut.attrs), frames_name,
                    [s.to_dict() for s in stages], source_times=times_name)


def write_attrs(dest_imgs, dest_ut, src_time_attrs, source_dataset, chain_fitted, source_times="ut1_unix"):
    """Attributes linking an image dataset and its time dataset, and recording the chain (as JSON)."""
    time_attrs = dict(src_time_attrs)
    time_attrs.update(
        units="seconds since 1970-01-01",
        time_scale="UT1",
        source_dataset=source_times,
    )
    for key, val in time_attrs.items():
        dest_ut.attrs[key] = val

    dest_imgs.attrs["source_dataset"] = source_dataset
    dest_imgs.attrs["preprocessing_chain"] = chain_json(chain_fitted)
    dest_imgs.attrs["time_dataset"] = dest_ut.name.rsplit("/", 1)[-1]
    dest_ut.attrs["frame_dataset"] = dest_imgs.name.rsplit("/", 1)[-1]

    for key, val in IMAGE_ATTRS.items():
        dest_imgs.attrs[key] = val
