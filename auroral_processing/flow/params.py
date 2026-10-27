from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from typing import Any, ClassVar
import cv2


def _param(default, setter, help):
    return field(default=default, metadata={"setter": setter, "help": help})


class _Params:
    """Saving, loading and copying shared by the settings classes of every flow method."""

    METHOD: ClassVar[str]
    LABEL: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict with external names (``lambda`` not ``lambda_``)."""
        return {external_name(k): v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        """Inverse of ``to_dict``. Accepts ``lambda`` or ``lambda_``; rejects unknown keys."""
        by_external = {external_name(f.name): f for f in fields(cls)}
        kwargs = {}
        for k, v in d.items():
            f = by_external.get(external_name(k))
            if f is None:
                raise ValueError(f"unknown {cls.LABEL} setting {k!r}; allowed: {sorted(by_external)}")
            kwargs[f.name] = _cast(f, v)
        return cls(**kwargs)

    def replace(self, **changes):
        """Copy with some settings changed. Keys may use external names."""
        return type(self).from_dict({**self.to_dict(), **changes})


@dataclass
class TVL1Params(_Params):
    """
    Settings for OpenCV's Dual TV-L1 optical flow. Defaults match OpenCV's.

    ``lambda_`` has a trailing underscore because ``lambda`` is a Python keyword; everywhere
    outside Python (CLI flags, HDF5 attrs, sweep names) it is called ``lambda``.
    """

    lambda_: float = _param(0.15, "setLambda", "data weight: higher = follow the images more, smooth less")
    gamma: float = _param(0.0, "setGamma", "brightness-change weight (>0 tolerates illumination change)")
    tau: float = _param(0.25, "setTau", "solver time step (keep <= 0.25 for stability)")
    theta: float = _param(0.3, "setTheta", "coupling between data and smoothness terms")
    scales: int = _param(5, "setScalesNumber", "pyramid levels: more = can follow larger motions")
    scale_step: float = _param(0.8, "setScaleStep", "size ratio between pyramid levels")
    warps: int = _param(5, "setWarpingsNumber", "warps per level: more = better for large/nonlinear motion")
    epsilon: float = _param(0.01, "setEpsilon", "convergence threshold (smaller = more accurate, slower)")
    inner_iterations: int = _param(30, "setInnerIterations", "inner iterations per warp")
    outer_iterations: int = _param(10, "setOuterIterations", "outer iterations per warp")
    median_filtering: int = _param(5, "setMedianFiltering", "median filter size on the flow (1 = off; odd)")

    METHOD: ClassVar[str] = "tvl1"
    LABEL: ClassVar[str] = "TV-L1"


@dataclass
class PIVParams(_Params):
    """
    Settings for multi-pass, window-deforming PIV cross-correlation (``openpiv.windef``).

    The first pass uses ``first_window`` px windows to find large motions (up to about a third of
    the window); each later pass halves the window, shifting and deforming it by the flow found so
    far, and the last pass uses ``window``. E.g. 64 -> 32 -> 16, or 64 -> 32 -> 16 -> 12.
    """

    window: int = _param(16, None, "final interrogation window, px")
    first_window: int = _param(64, None, "first-pass window, px: sets the largest trackable motion (~1/3 of it)")
    overlap: float = _param(0.5, None, "window overlap fraction (0.75 = a vector every window/4 px)")
    max_disp: float = _param(40.0, None, "largest accepted displacement, px/pair")
    s2n_threshold: float = _param(1.05, None, "peak-to-mean signal-to-noise below which a vector is replaced")
    median_threshold: float = _param(3.0, None, "normalised median test: larger = fewer vectors replaced")
    normalized: int = _param(1, None, "normalised correlation (1/0): less sensitive to brightness changes")

    METHOD: ClassVar[str] = "piv"
    LABEL: ClassVar[str] = "PIV"


@dataclass
class FLCTParams(_Params):
    """Settings for Fourier local correlation tracking (``pyflct``): a Gaussian window at every pixel."""

    sigma: float = _param(8.0, None, "Gaussian window width, px")
    thresh: float = _param(0.05, None, "skip pixels darker than this fraction of the brightest (0-1)")
    kr: float = _param(0.0, None, "low-pass filter, fraction of Nyquist (0 = off)")
    skip: int = _param(1, None, "compute every n-th pixel and interpolate (faster)")
    biascor: int = _param(1, None, "bias correction (1/0); without it FLCT underestimates speeds")

    METHOD: ClassVar[str] = "flct"
    LABEL: ClassVar[str] = "FLCT"


METHODS = {c.METHOD: c for c in (TVL1Params, PIVParams, FLCTParams)}


def params_from_dict(method: str, d: dict[str, Any]) -> _Params:
    """Settings object of ``method`` ("tvl1", "piv", "flct") from its ``to_dict``."""
    if method not in METHODS:
        raise ValueError(f"unknown flow method {method!r}; allowed: {sorted(METHODS)}")
    return METHODS[method].from_dict(d)


def external_name(name: str) -> str:
    return name.rstrip("_")


def _cast(f, v):
    return int(v) if f.type in ("int", int) else float(v)


def make_tvl1(params: TVL1Params, *, use_initial_flow=False):
    """Build a cv2 DualTVL1OpticalFlow configured with ``params``."""
    if not hasattr(cv2, "optflow"):
        raise ImportError("TV-L1 needs the OpenCV contrib modules (pip install opencv-contrib-python)")
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    for f in fields(params):
        getattr(tvl1, f.metadata["setter"])(getattr(params, f.name))
    tvl1.setUseInitialFlow(use_initial_flow)
    return tvl1
