"""
Flow output file layout (one file per run, separate from the source):

    flow        (P, H, W, 2) float32   u = +x right, v = +y down, frame i0 -> i1, px per pair
    i0, i1      (P,) int64             source frame indices of each pair
    t0, t1      (P,) float64           source times of each pair
    done        (P,) bool              pair written (used to resume)
    metrics/<name> (P,) float32        per-pair quality metrics (NaN until written)

``flow`` is LZF-compressed (MATLAB needs the LZF filter plugin; ``repack_flow`` converts to gzip).

attrs: ``config`` (JSON of everything that defines the run), each TV-L1 setting as ``tvl1_<name>``,
``lo``/``hi`` input scaling, ``cv2_version``, ``created``, ``complete``.
"""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import h5py
import numpy as np
import cv2


class FlowWriter:
    """Create or reopen a flow file and write pairs into it."""

    def __init__(self, path, *, config, pairs, t0, t1, shape, metrics, scaling, overwrite=False, compression="lzf"):
        self.path = Path(path)
        config_json = json.dumps(config, sort_keys=True)

        if self.path.exists() and not overwrite:
            self.f = h5py.File(self.path, "r+")
            if self.f.attrs.get("config") != config_json:
                self.f.close()
                raise FileExistsError(
                    f"{self.path} exists with different settings; pass overwrite=True or choose another output"
                )
            return

        P, (H, W) = len(pairs), shape
        self.f = h5py.File(self.path, "w")
        self.f.create_dataset(
            "flow", (P, H, W, 2), dtype=np.float32, chunks=(1, H, W, 2), compression=compression, fillvalue=np.nan,
            compression_opts=4 if compression == "gzip" else None,
        )
        self.f["i0"], self.f["i1"] = pairs[:, 0], pairs[:, 1]
        self.f["t0"], self.f["t1"] = t0, t1
        self.f["done"] = np.zeros(P, dtype=bool)
        for name in metrics:
            self.f.create_dataset(f"metrics/{name}", (P,), dtype=np.float32, fillvalue=np.nan)

        a = self.f.attrs
        a["config"] = config_json
        for k, v in config["tvl1"].items():
            a[f"tvl1_{k}"] = v
        for k, v in scaling.items():
            a[k] = v
        a["cv2_version"] = cv2.__version__
        a["created"] = datetime.now(timezone.utc).isoformat()
        a["complete"] = False

    @property
    def done(self) -> np.ndarray:
        return self.f["done"][:]

    @property
    def scaling(self) -> dict:
        return {k: float(self.f.attrs[k]) for k in ("lo", "hi")}

    def write(self, k, flow, metrics):
        self.f["flow"][k] = flow
        for name, value in metrics.items():
            self.f[f"metrics/{name}"][k] = value
        self.f["done"][k] = True

    def finish(self):
        self.f.attrs["complete"] = bool(self.done.all())
        self.close()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_flow(path) -> h5py.File:
    """Open a flow file read-only."""
    return h5py.File(path, "r")


def read_config(f: h5py.File) -> dict:
    """The run configuration stored in a flow file (source, datasets, pairing, TV-L1 settings, ...)."""
    return json.loads(f.attrs["config"])


def repack_flow(path, out=None) -> Path:
    """
    Rewrite a flow file with gzip compression (readable and writable by MATLAB), in place unless
    ``out`` is given. For MATLAB installs without the LZF filter plugin.
    """
    path = Path(path)
    out = Path(out) if out else path
    tmp = out.with_name(out.name + ".repack")
    with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v

        def copy(name, obj):
            if isinstance(obj, h5py.Dataset):
                ds = dst.create_dataset(name, data=obj[()], chunks=obj.chunks,
                                        compression="gzip" if obj.chunks else None,
                                        compression_opts=4 if obj.chunks else None)
                for k, v in obj.attrs.items():
                    ds.attrs[k] = v

        src.visititems(copy)
    os.replace(tmp, out)
    return out
