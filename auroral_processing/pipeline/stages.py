from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Any, ClassVar, Iterable
from dataclasses import dataclass, asdict, fields
from itertools import chain as _chain, islice
from auroral_processing.utils.norm import compute_norm
import numpy as np
from collections import deque
import cv2
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import wiener

Stream = Iterable[tuple[float, np.ndarray]]
STAGES: dict[str, type[Stage]] = {}

# keys a chain entry may carry that are about the chain, not the stage's settings
META_KEYS = ("id", "enabled")


@dataclass
class Stage(ABC):
    """One step of the processing chain. Each step type is a subclass of Stage"""

    name: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        STAGES[cls.name] = cls

    def needs_fit(self) -> bool:
        """True if ``fit`` must estimate settings from the source frames before running."""
        return False

    def fit(self, frames, times) -> Stage:
        """Estimate settings from the source frames. Default: nothing to estimate."""
        return self

    def n_out(self, n_in: int):
        """Number of output frames for ``n_in`` in frames. Defaults to 1:1"""
        return n_in

    def label(self) -> str:
        """Short tag for automatic output names, e.g. "bin30". Empty to leave the stage out."""
        return self.name

    @abstractmethod
    def run(self, src: Stream) -> Stream:
        """Generator: consume (t, frame) pairs from src, yield new ones."""

    # ---- Saving and Loading ----
    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.name, **{k: _plain(v) for k, v in asdict(self).items()}}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Stage:
        """
        Build a stage from ``{"stage": name, setting: value, ...}``. The dict is not modified.

        Values are cast to the setting's type, so YAML strings such as "1e-3" work. The chain keys
        ``id`` and ``enabled`` are ignored here (see ``config.py``).
        """
        d = {k: v for k, v in d.items() if k not in META_KEYS}
        name = d.pop("stage", None)

        if name not in STAGES:
            raise ValueError(f"unknown stage name {name!r}; known: {sorted(STAGES)}")

        cls = STAGES[name]
        by_name = {f.name: f for f in fields(cls)}

        if unknown := set(d) - set(by_name):
            raise ValueError(f"{name}: unknown settings {sorted(unknown)}; allowed settings: {sorted(by_name)}")
        return cls(**{k: _cast(name, by_name[k], v) for k, v in d.items()})


def _plain(v):
    """numpy scalars/arrays -> plain Python, so json.dumps works."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _cast(stage, f, v):
    """Cast ``v`` to the type annotation of field ``f`` (annotations are strings, e.g. "float | None")."""
    types = [t.strip() for t in str(f.type).split("|")]
    if v is None or (isinstance(v, str) and v.lower() in ("none", "null") and "None" in types):
        if "None" not in types:
            raise ValueError(f"{stage}.{f.name} can't be null")
        return None
    v = _plain(v)
    for t in types:
        try:
            if t == "int":
                if isinstance(v, bool):
                    raise ValueError
                x = float(v)
                if x != int(x):
                    raise ValueError
                return int(x)
            if t == "float":
                if isinstance(v, bool):
                    raise ValueError
                return float(v)
            if t == "str":
                return str(v)
        except (TypeError, ValueError):
            continue
    raise ValueError(f"{stage}.{f.name}: {v!r} is not a valid {f.type}")


# ---------- stage subclasses -------------
@dataclass
class Normalize(Stage):
    """Clip to [vmin, vmax] and map to 0-1. Unset limits are the 1st/99th percentiles of the source."""

    name = "normalize"
    CURVES: ClassVar[tuple[str, ...]] = ("linear", "log", "sqrt")
    vmin: float | None = None
    vmax: float | None = None
    curve: str = "linear"

    def __post_init__(self):
        if self.curve not in self.CURVES:
            raise ValueError(f"curve must be one of {self.CURVES}, not {self.curve!r}")
        self._check_limits()

    def _check_limits(self):
        if self.vmin is not None and self.vmax is not None and self.vmax <= self.vmin:
            raise ValueError(f"normalize: vmax ({self.vmax}) must be greater than vmin ({self.vmin})")
        if self.vmin is not None:
            if self.curve == "log" and self.vmin <= 0:
                raise ValueError(f"normalize: curve='log' needs vmin > 0, got {self.vmin}")
            if self.curve == "sqrt" and self.vmin < 0:
                raise ValueError(f"normalize: curve='sqrt' needs vmin >= 0, got {self.vmin}")

    def needs_fit(self):
        return self.vmin is None or self.vmax is None

    def label(self):
        return "norm" if self.curve == "linear" else f"norm{self.curve}"

    def apply_norm_to_frame(self, frame):
        frame = np.clip(frame, self.vmin, self.vmax).astype(np.float32)
        if self.curve == "linear":
            return (frame - self.vmin) / (self.vmax - self.vmin)
        elif self.curve == "log":
            return (np.log(frame) - np.log(self.vmin)) / (np.log(self.vmax) - np.log(self.vmin))
        elif self.curve == "sqrt":
            return (np.sqrt(frame) - np.sqrt(self.vmin)) / (np.sqrt(self.vmax) - np.sqrt(self.vmin))

    def fit(self, frames, times):
        if self.needs_fit():
            norm = compute_norm(frames, times, 1)
            if self.vmin is None:
                self.vmin = _plain(norm.vmin)
            if self.vmax is None:
                self.vmax = _plain(norm.vmax)
            self._check_limits()
        return self

    def run(self, src):
        for t, frame in src:
            yield t, self.apply_norm_to_frame(frame)


@dataclass
class SpatialMedian(Stage):
    """k x k median filter on each frame."""

    name = "spatial_median"
    k: int = 3

    def __post_init__(self):
        if self.k < 3 or self.k % 2 == 0:
            raise ValueError(f"k must be odd and >= 3. k = {self.k}")

    def label(self):
        return f"med{self.k}"

    def run(self, src):
        for t, frame in src:
            frame = frame.astype(np.float32)
            if self.k <= 5:  # OpenCV only supports k = 3 or 5 for float32
                yield t, cv2.medianBlur(frame, self.k)
            else:  # same border handling as OpenCV (replicate edge pixels)
                yield t, median_filter(frame, size=self.k, mode="nearest")


@dataclass
class SlidingTemporalMedian(Stage):
    """Running median over ``window`` frames, stamped with the centre frame's time."""

    name = "sliding_temporal_median"
    window: int = 5

    def __post_init__(self):
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError(f"window must be odd and >= 1. window = {self.window}")

    def label(self):
        return f"tmed{self.window}"

    def n_out(self, n_in):
        return max(0, n_in - self.window + 1)

    def run(self, src):
        k = self.window // 2
        ring = None
        times = deque(maxlen=self.window)
        for n, (t, frame) in enumerate(src):
            if ring is None:
                ring = np.empty((self.window,) + frame.shape, dtype=np.float32)

            ring[n % self.window] = frame  # order in the ring doesn't matter for a median
            times.append(t)
            if n >= self.window - 1:
                yield times[k], np.partition(ring, k, axis=0)[k]


STAGES["sliding_temporal_medial"] = SlidingTemporalMedian  # old misspelt name, still accepted


@dataclass
class BinnedTemporalAverage(Stage):
    """Average each run of ``bin_size`` frames into one, stamped with their mean time. The partial last bin is dropped."""

    name = "binned_temporal_average"
    bin_size: int = 15

    def __post_init__(self):
        if self.bin_size < 1:
            raise ValueError("bin_size must be >= 1")

    def label(self):
        return f"bin{self.bin_size}"

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


def _local_noise(frame, size):
    """scipy.signal.wiener's own noise estimate: the mean local variance of the frame."""
    frame = frame.astype(np.float64)
    mean = uniform_filter(frame, size, mode="constant")
    return float(np.mean(uniform_filter(frame**2, size, mode="constant") - mean**2))


@dataclass
class Wiener(Stage):
    """
    ``scipy.signal.wiener`` on each frame (window ``size``, default 3).

    noise: null re-estimates the noise power on every frame (scipy's default). A number fixes it.
    "auto" estimates it once, as the median over the first ``AUTO_FRAMES`` frames reaching this stage,
    and records the value, so every frame gets the same smoothing.
    """

    name = "wiener"
    AUTO_FRAMES: ClassVar[int] = 5
    size: int | None = None
    noise: float | str | None = None

    def __post_init__(self):
        if isinstance(self.noise, str) and self.noise != "auto":
            raise ValueError(f"wiener noise must be a number, null or 'auto', not {self.noise!r}")

    def label(self):
        tag = "wiener" if self.size in (None, 3) else f"wiener{self.size}"
        return tag + ("auto" if self.noise == "auto" else "")

    def run(self, src):
        if self.noise == "auto":
            src = iter(src)
            head = list(islice(src, self.AUTO_FRAMES))
            if head:
                self.noise = float(np.median([_local_noise(f, self.size or 3) for _, f in head]))
            src = _chain(head, src)
        for t, frame in src:
            yield t, wiener(frame, self.size, self.noise)


@dataclass
class ToUint16(Stage):
    """Map [low, high] to 0..out_max and store as uint16."""

    name = "to_uint16"
    low: float = 0.0
    high: float = 1.0
    out_max: int = 2**16 - 1  # uint16 max (65535); 2**16 would wrap to 0

    def __post_init__(self):
        if self.high <= self.low:
            raise ValueError(f"high: {self.high} must be greater than low: {self.low}")
        if not 1 <= self.out_max <= 2**16 - 1:
            raise ValueError(f"out_max must be in 1..65535 (uint16), got {self.out_max}")

    def label(self):
        return "" if self.out_max == 2**16 - 1 else f"u{self.out_max}"

    def run(self, src):
        scale = self.out_max / (self.high - self.low)

        for t, frame in src:
            clipped = np.clip((frame - self.low) * scale, 0, self.out_max)
            yield t, np.rint(clipped).astype(np.uint16)
