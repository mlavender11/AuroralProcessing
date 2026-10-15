from abc import abstractmethod, ABC
from typing import Any, ClassVar, Iterable, Iterator
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from auroral_processing.utils.norm import compute_norm
import numpy as np
from collections import deque
from scipy.signal import wiener

Stream = Iterable[tuple[float, np.ndarray]]
STAGES: dict[str, type[Stage]] = {}


@dataclass
class Stage(ABC):
    """One step of the processing chain. Each step type is a subclass of Stage"""

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        return super().__init_subclass__(**kwargs)
        STAGES[cls.name] = cls

    def fit(self, frames, times) -> Stage:
        """Estimate settings from data. Default: nothing to estimate.
        e.g. use in the case of normalize to find vmin and vmax #TODO is this true?"""

    def n_out(self, n_in: int):
        """Number of output frames for ``n_in`` in frames. Defaults to 1:1"""
        return n_in

    @abstractmethod
    def run(self, src: Stream) -> Stream:
        # TODO return iterable or generator?
        """Generator: consume (t, frame) pairs from src, yield new ones."""

    # ---- Saving and Loading ----
    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.name, **{k: _plain(v) for k, v in asdict(self).items()}}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Stage:
        name = d.pop("stage", None)

        if name not in STAGES:
            # TODO why name!r, not just name?
            raise ValueError(f"unkown stage name {name!r}, known: {STAGES}")

        cls = STAGES[name]
        allowed = {f.name for f in fields(cls)}

        if unknown := set(d) - allowed:
            raise ValueError(f"{name}: unkown settings {unknown}; allowed settings: {allowed}")
        return cls(**d)


def _plain(v):
    """numpy scalars/arrays -> plain Python, so json.dumps works."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


# ---------- stage subclasses -------------
@dataclass
class Normalize(Stage):
    name = "normalize"
    vmin: float | None = None
    vmax: float | None = None
    curve: str = "linear"

    def __post_init__(self):
        if self.curve not in ("linear", "log", "sqrt"):  # TODO can I make this a class constant list
            raise ValueError(f"curve must be linear, log, or swrt, not {self.curve!r}")

    def apply_norm_to_frame():
        ...  # TODO implement
        # TODO should this be a static or class method?

    def fit(self, frames, times):
        if self.vmin is None or self.vmax is None:
            norm = compute_norm(frames, times, 1)
            if self.vmin is None:
                self.vmin = norm.vmin

            if self.vmax is None:
                self.vmax = norm.vmax

        return self

    def run(self, src):
        for t, frame in src:
            yield t, self.apply_norm_to_frame(frame)


@dataclass
def SpatialMedian(Stage):
    name = "spatial_median"
    k: int = 3

    def __post_init__(self):
        if self.k < 3 or self.k % 2 == 0:
            raise ValueError(f"k must be odd and >= 3. k = {self.k}")

    def run(self, src):
        for t, frame in src:
            ...  # TODO move apply_spatial_median to here


@dataclass
class SlidingTemporalMedian(Stage):
    name = "sliding_temporal_medial"
    window: int = 5

    def __post_init__(self):
        if self.window % 2 == 0:
            raise ValueError(f"window must be odd. window = {self.window}")  # TODO is this true

    def n_out(self, n_in):
        return n_in - self.window + 1

    def run(self, src):
        k = self.window // 2
        ring = None
        times = deque(maxlen=self.window)
        for n, (t, frame) in enumerate(src, start=0):
            if ring is None:
                ring = np.empty((self.window,) + frame.shape, dtype=np.float32)

            ring[n % self.window] = frame  # TODO n or n-1?
            times.append(t)
            if n >= self.window -1:
                yield times[k], np.partition(ring, k, axis=0)[k]


@dataclass
class BinnedTemporalAverage(Stage):
    bin_size: int = 15

    def __post_init__(self):
        if self.bin_size < 1:
            raise ValueError("bin_size must be >= 1")

    def n_out(self, n_in):
        return n_in // self.bin_size

    def run(self, src):
        frame_sum, t_ref, dt_sum, counter = None, 0.0, 0.0, 0

        for t, frame in src:
            if frame_sum is None:
                frame_sum = np.zeros(frame.shape, dtype=np.float64)
                t_ref = t

            frame_sum += frame
            dt_sum += t - t_ref
            counter += 1

            if counter == self.bin_size:
                yield t_ref + (dt_sum / self.bin_size), frame_sum / self.bin_size
                frame_sum, dt_sum, counter = None, 0.0, 0


@dataclass
class Wiener(Stage):
    name = "wiener"
    size: int | None = None
    noise: float | None = None

    # def fit(self, frames, times): # TODO estimate noise from sample?

    def run(self, src):
        for t, frame in src:
            yield t, wiener(frame, self.size, self.noise)


@dataclass
class ToUint16(Stage):
    name = "to_uint16"
    low: float = 0.0
    high: float = 1.0
    out_max: int = 255

    def __post_init__(self):
        if self.hi <= self.low:
            raise ValueError(f"high: {self.high} must be greater than low: {self.low}")

    def run(self, src):
        scale = self.out_max / (self.high - self.low)

        for t, frame in src:
            clipped = np.clip((frame - self.low) * scale, 0, self.out_max)
            yield t, np.rint(clipped).astype(np.uint16)
