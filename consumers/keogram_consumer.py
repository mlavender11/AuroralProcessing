import numpy as np
from .. import hdf_utils
from ..config import BIN_WIDTH_SECONDS
import math
import datetime
from zoneinfo import ZoneInfo


def compute_keogram_bins(n_frames, ut):
    print(f"binwidthseconds = {BIN_WIDTH_SECONDS}")  # TODO remove
    if BIN_WIDTH_SECONDS is None:
        return n_frames, 1
    num_seconds = ut[-1] - ut[0]
    n_bins = max(1, int(num_seconds / BIN_WIDTH_SECONDS))
    frames_per_bin = max(1, n_frames // n_bins)
    n_bins = math.ceil(n_frames / frames_per_bin)  # actual columns
    return n_bins, frames_per_bin


class KeogramConsumer:  # TODO add type verification for ut
    def __init__(self, height, width, n_bins, frames_per_bin, cmap, norm, outfn, ut):
        self.frames_per_bin = frames_per_bin
        self.n_bins = n_bins
        self.keogram_NS = np.zeros((height, n_bins), dtype=float)
        self.keogram_EW = np.zeros((width, n_bins), dtype=float)
        self.NS_buffer = np.empty((frames_per_bin, height))
        self.EW_buffer = np.empty((frames_per_bin, width))
        self.slice_idx_NS = width // 2
        self.slice_idx_EW = height // 2
        self.norm = norm
        self.buffer_idx = 0
        self.bin_idx = 0
        self.outfn = outfn
        self.height = height
        self.width = width
        self.cmap = cmap
        self.ut = ut

    def update(self, i, frame, time):
        self.NS_buffer[self.buffer_idx] = frame[:, self.slice_idx_NS]
        self.EW_buffer[self.buffer_idx] = frame[self.slice_idx_EW, :]
        self.buffer_idx += 1

        if self.buffer_idx == self.frames_per_bin:
            self.keogram_NS[:, self.bin_idx] = self.NS_buffer.mean(axis=0)
            self.keogram_EW[:, self.bin_idx] = self.EW_buffer.mean(axis=0)
            self.bin_idx += 1
            self.buffer_idx = 0

    def finalize(self):
        print("keogram")
        print("extra buffer")
        if self.buffer_idx > 0 and self.bin_idx < self.n_bins:
            self.keogram_NS[:, self.bin_idx] = self.NS_buffer[: self.buffer_idx].mean(axis=0)
            self.keogram_EW[:, self.bin_idx] = self.EW_buffer[: self.buffer_idx].mean(axis=0)

        ut_hours = (self.ut % 86400) / 3600.0

        date_str = datetime.datetime.fromtimestamp(self.ut[int(len(self.ut) // 2)], datetime.timezone.utc).strftime(
            "%Y-%m-%d"
        )

        print("saving")
        hdf_utils.save_keogram_NS_and_EW(
            outfn=self.outfn,
            keogram_NS=self.keogram_NS,
            keogram_EW=self.keogram_EW,
            ut_hours=ut_hours,
            frame_height=self.height,
            frame_width=self.width,
            cmap=self.cmap,
            norm=self.norm,
            date_str=date_str,
        )
