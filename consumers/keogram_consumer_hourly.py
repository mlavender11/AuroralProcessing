import numpy as np
import math
import datetime
from matplotlib.ticker import FixedLocator, FuncFormatter
import matplotlib.pyplot as plt


def compute_keogram_bins(bin_width_seconds, native_dt=None, hour_span_seconds=3600):
    """
    Number of keogram columns needed to cover a full clock hour, and the
    time width (seconds) each column represents.

    If bin_width_seconds is None, falls back to one column per native frame
    (native_dt = median frame interval), same as the old "no binning" mode.
    """
    if bin_width_seconds is None:
        if native_dt is None:
            raise ValueError("native_dt is required when bin_width_seconds is None")
        bin_width_seconds = native_dt
    n_bins = math.ceil(hour_span_seconds / bin_width_seconds)
    return n_bins, bin_width_seconds


def save_keogram_NS_and_EW(
    *,
    keogram_NS,
    keogram_EW,
    ut_hours,
    frame_height,
    frame_width,
    outfn,
    cmap,
    norm,
    date_str,
    tick_interval_hours=1 / 6,
):
    import math

    fig, (axNS, axEW) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    imNS = axNS.imshow(
        keogram_NS,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[ut_hours[0], ut_hours[-1], 0, frame_height],
        norm=norm,
    )
    axNS.set_ylabel("Y Pixel Index")
    axNS.set_title("North-South Keogram")

    imEW = axEW.imshow(
        keogram_EW,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[ut_hours[0], ut_hours[-1], 0, frame_width],
        norm=norm,
    )
    axEW.set_ylabel("X Pixel Index")
    axEW.set_title("East-West Keogram")
    axEW.set_xlabel("UTC Time")

    # ticks every ten min
    start, end = ut_hours[0], ut_hours[-1]
    step = tick_interval_hours

    first_regular_tick = math.ceil(start / step) * step
    regular_ticks = list(np.arange(first_regular_tick, end, step))

    if regular_ticks and (regular_ticks[0] - start) < step / 5:
        regular_ticks.pop(0)
    if regular_ticks and (end - regular_ticks[-1]) < step / 5:
        regular_ticks.pop()

    ticks = [start] + regular_ticks + [end]

    axEW.xaxis.set_major_locator(FixedLocator(ticks))
    axEW.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f"{math.floor(x) % 24:02d}:{min(round((x % 1) * 60), 59):02d}")
    )
    axEW.tick_params(axis="x", labelrotation=45)

    fig.suptitle(date_str)
    plt.savefig(outfn)
    plt.close(fig)


class KeogramConsumer:  # TODO add type verification for ut
    def __init__(
        self,
        *,
        height,
        width,
        cmap,
        norm,
        outfn,
        hour_start_ut,
        bin_width_seconds=1,
        native_dt=None,
        hour_span_seconds=3600,
    ):
        self.n_bins, self.bin_width_seconds = compute_keogram_bins(
            bin_width_seconds, native_dt=native_dt, hour_span_seconds=hour_span_seconds
        )
        self.hour_start_ut = hour_start_ut
        self.hour_span_seconds = hour_span_seconds

        # sum + count per column, instead of a sequential fill buffer --
        # lets us place each frame directly into the column its timestamp
        # belongs to, so gaps at the start/end (or anywhere) stay empty
        # instead of shifting everything else over.
        self.sum_NS = np.zeros((height, self.n_bins), dtype=float)
        self.sum_EW = np.zeros((width, self.n_bins), dtype=float)
        self.count = np.zeros(self.n_bins, dtype=np.int64)

        self.slice_idx_NS = width // 2
        self.slice_idx_EW = height // 2
        self.norm = norm
        self.outfn = outfn
        self.height = height
        self.width = width
        self.cmap = cmap

    def _bin_for_time(self, time):
        offset = time - self.hour_start_ut
        if offset < 0 or offset >= self.hour_span_seconds:
            return None
        return int(offset // self.bin_width_seconds)

    def update(self, i, frame, time):
        b = self._bin_for_time(time)
        if b is None:
            return  # frame falls outside this clock hour (e.g. clipped by start/end_time)
        self.sum_NS[:, b] += frame[:, self.slice_idx_NS]
        self.sum_EW[:, b] += frame[self.slice_idx_EW, :]
        self.count[b] += 1

    def finalize(self):
        with np.errstate(invalid="ignore"):
            keogram_NS = self.sum_NS / self.count
            keogram_EW = self.sum_EW / self.count
        empty = self.count == 0
        keogram_NS[:, empty] = np.nan
        keogram_EW[:, empty] = np.nan

        hour_start_hours = (self.hour_start_ut % 86400) / 3600.0
        hour_end_hours = hour_start_hours + self.hour_span_seconds / 3600.0
        ut_hours = np.array([hour_start_hours, hour_end_hours])

        date_str = datetime.datetime.fromtimestamp(self.hour_start_ut, datetime.timezone.utc).strftime("%Y-%m-%d")

        cmap = self.cmap.copy()
        cmap.set_bad(color="black")  # empty columns render black, not stretched

        save_keogram_NS_and_EW(
            outfn=self.outfn,
            keogram_NS=keogram_NS,
            keogram_EW=keogram_EW,
            ut_hours=ut_hours,
            frame_height=self.height,
            frame_width=self.width,
            cmap=cmap,
            norm=self.norm,
            date_str=date_str,
        )
