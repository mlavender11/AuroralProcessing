from __future__ import annotations
from dataclasses import dataclass, field, fields, asdict
from typing import Any
import cv2


def _param(default, setter, help):
    return field(default=default, metadata={"setter": setter, "help": help})


@dataclass
class TVL1Params:
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

    # ---- Saving and Loading ----
    def to_dict(self) -> dict[str, Any]:
        """Plain dict with external names (``lambda`` not ``lambda_``)."""
        return {external_name(k): v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TVL1Params:
        """Inverse of ``to_dict``. Accepts ``lambda`` or ``lambda_``; rejects unknown keys."""
        by_external = {external_name(f.name): f for f in fields(cls)}
        kwargs = {}
        for k, v in d.items():
            f = by_external.get(external_name(k))
            if f is None:
                raise ValueError(f"unknown TV-L1 setting {k!r}; allowed: {sorted(by_external)}")
            kwargs[f.name] = _cast(f, v)
        return cls(**kwargs)

    def replace(self, **changes) -> TVL1Params:
        """Copy with some settings changed. Keys may use external names."""
        return TVL1Params.from_dict({**self.to_dict(), **changes})


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
